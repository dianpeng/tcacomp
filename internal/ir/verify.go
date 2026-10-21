package ir

import (
	"fmt"
	"slices"
)

// Violation is one chain-consistency failure reported by Verify (PASS-06
// substrate). NodeID and LastTouchedBy name the responsible node and the
// last pass that touched it (structural, via D-17); Invariant names the
// checker that fired; Msg carries human detail.
type Violation struct {
	NodeID        uint64
	LastTouchedBy string
	Invariant     string
	Msg           string
}

// String renders the canonical diagnostic form:
//
//	verify: node <id> (last touched by '<pass>'): invariant '<name>': <msg>
func (v Violation) String() string {
	return fmt.Sprintf("verify: node %d (last touched by '%s'): invariant '%s': %s",
		v.NodeID, v.LastTouchedBy, v.Invariant, v.Msg)
}

// Invariant names for the two chain directions checked on every graph.
const (
	invUseDef = "use-def" // every operand handle resolves to a live def of a known kind
	invDefUse = "def-use" // every def's ref list matches the operand slots that own its records
)

// Verify runs both chain-consistency checkers over g and returns every
// violation found, in deterministic walk order: the level-aware roots (params
// plus the end node at Level 1, params plus statement order at Level 2),
// then per-node operand order — never map iteration (RESEARCH Pitfall 1).
// A nil or empty result means the chains are consistent. Verify needs no
// Materialize: the Level-1 roots cover the whole constructed graph.
func Verify(g *Graph) []Violation {
	if g == nil {
		return nil
	}
	var vs []Violation
	nodes := walkCheckUseDef(g, &vs)
	checkDefUse(nodes, &vs)
	checkConvention(g, &vs)
	return vs
}

// nodeViolation builds a Violation stamped with n's identity; n may be nil
// (corrupted chains can break back-pointers), in which case only Msg and
// Invariant are set.
func nodeViolation(n Node, invariant, msg string) Violation {
	v := Violation{Invariant: invariant, Msg: msg}
	if n != nil {
		nb := n.BasePtr()
		v.NodeID = nb.id
		v.LastTouchedBy = nb.lastTouchedBy
	}
	return v
}

func nodeIDStr(n Node) string {
	if n == nil {
		return "nil"
	}
	return fmt.Sprintf("%d", n.BasePtr().id)
}

// asImpl narrows n to the internal node surface. Every node the package
// constructs satisfies nodeImpl; a foreign Node implementation simply gets
// no operand checks rather than a panic.
func asImpl(n Node) nodeImpl {
	impl, _ := n.(nodeImpl)
	return impl
}

// walkCheckUseDef walks the graph from the level-aware roots (via Walk,
// directive 3 — params plus the end node at Level 1, params plus block
// statements at Level 2), checking the use-def direction at every node: each
// operand slot's handle must resolve to a live (not tombstoned) def of a kind
// <
// KindEnd. Zero handles and unwired records report the construction-time
// "nil def" message; killed records are the handle model's new corruption
// class, flagged by slot. It returns the visited nodes in walk order for the
// def-use pass. Lambda bodies are walked as edges alongside operand uses.
// R2's structural check rides the same walk: a second live KindReturn node
// past the first is flagged as a multiple-end-nodes violation.
func walkCheckUseDef(g *Graph, vs *[]Violation) []Node {
	var nodes []Node
	var firstEnd Node
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		nodes = append(nodes, n)
		checkNodeUseDef(n, vs)
		if n.KindOf() == KindReturn {
			if firstEnd == nil {
				firstEnd = n
			} else {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("multiple end nodes: node %d is a return, but node %d is already the graph's end node (R2: exactly one)",
						n.BasePtr().id, firstEnd.BasePtr().id)))
			}
		}
		return Descend
	}))
	return nodes
}

// checkNodeUseDef runs the per-node use-def checks for one visited node:
// erased-reachability, surplus operands (WR-01), per-slot operand
// resolution, and per-use payload resolution (parameter references,
// iteration dims/bounds, lambda params, as-orders, store targets — review
// P1R-01 makes every reference an SSAUse, so the use-def invariant covers
// payload uses exactly like operand slots). Kept as a separate function so
// the walk control stays trivial.
func checkNodeUseDef(n Node, vs *[]Violation) {
	nb := n.BasePtr()
	if nb.erased {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"erased node is still reachable from the roots"))
	}
	if nb.surplus > 0 {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("node constructed with %d surplus operands (arity %d)",
				nb.surplus, fixedArity(nb.kind))))
	}
	impl := asImpl(n) // foreign nodes get no operand checks, not a panic
	if impl == nil {
		return
	}
	slot := 0
	impl.forEachOperand(func(u SSAUse) {
		checkOneUse(n, u, fmt.Sprintf("operand slot %d", slot), vs)
		slot++
	})
	if pb, ok := n.(payloadBearer); ok {
		idx := 0
		pb.forEachPayloadUse(func(u SSAUse) {
			checkOneUse(n, u, fmt.Sprintf("payload use %d", idx), vs)
			idx++
		})
	}
	checkNodeShape(n, vs)
}

// checkNodeShape enforces the per-kind operand-shape invariants of the
// memory and loop node suites, fail-closed: a bounded dimvar binds a
// dimvar; an iter's operand 0 is a bounded dimvar and operand 1, when
// present, an iter; a dimvar's operand 0 is a dim InputParam; a
// paramRef's operand 0 is an InputParam; a cluster's dim operands LEAD its
// input list; HIR GMEM load/store tensors bind a RoleTensor InputParam;
// LIR SMEM accesses bind a MemNewSMem buffer; LIR FRAG accesses bind a
// FragNew buffer; a for-loop binds a dimvar, with trailing operands beyond
// (iv, bound, body) pinned to KeyedAttribute annotations (M1,
// quick-261021-jay); an intrinsic map query holds exactly one const-u32
// DoSoC operand and a field in {index, dimension} (§5 companion 3); an
// IterClusterProject's operand 0 is an itercluster; an InductionVariable's operand 0 is an
// iter; a TensorDimOf binds a RoleTensor tensor and a const axis; an As
// binds dim InputParams; a Reduce's operand 1 is its KeyedAttribute init;
// and a Map/Reduce's optional As is unique and last, with the map's
// value-operand count matching its lambda parameter count
// (quick-261019-ktm). Nil operands are left to the generic use-def slot
// check.
func checkNodeShape(n Node, vs *[]Violation) {
	switch t := n.(type) {
	case *boundedDimVarNode:
		if iv := operandAt(t, 0); iv != nil {
			if _, ok := iv.(*dimVarNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"bounded dimvar operand 0 must be a dimvar"))
			}
		}
	case *iterNode:
		ops := OperandsOf(t)
		if len(ops) < 1 || len(ops) > 2 {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"iter must hold [biv] or [biv, parent]"))
			break
		}
		if b := operandAt(t, 0); b != nil {
			if _, ok := b.(*boundedDimVarNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"iter operand 0 must be a bounded dimvar"))
			}
		}
		if p := operandAt(t, 1); p != nil {
			if _, ok := p.(*iterNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"iter operand 1 must be the parental iter"))
			}
		}
	case *dimVarNode:
		if d := operandAt(t, 0); d != nil && !isDimParam(d) {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"dimvar operand 0 must be a dim InputParam"))
		}
	case *paramRefNode:
		if p := operandAt(t, 0); p != nil {
			if _, ok := p.(*inputParamNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"param operand 0 must be an InputParam node"))
			}
		}
	case *iterClusterNode:
		// The cluster's dim operands must LEAD the input list: once a
		// non-dim operand appears, no dim may follow (D14 read-face
		// contract: ClusterDims/ClusterInputs split at that boundary).
		seenInput := false
		for _, op := range OperandsOf(t) {
			if op == nil {
				continue
			}
			if isDimParam(op) {
				if seenInput {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						"cluster dims must lead the input list"))
					break
				}
				continue
			}
			seenInput = true
		}
	case *memLoadNode:
		checkTensorParam(n, operandAt(t, 0), vs)
	case *memStoreNode:
		checkTensorParam(n, operandAt(t, 1), vs)
	case *memLoadSMemNode:
		checkSMemBuffer(n, operandAt(t, 0), vs)
	case *memStoreSMemNode:
		checkSMemBuffer(n, operandAt(t, 1), vs)
	case *fragLoadNode:
		checkFragBuffer(n, operandAt(t, 0), vs)
	case *fragStoreNode:
		checkFragBuffer(n, operandAt(t, 1), vs)
	case *forLoopNode:
		if iv := operandAt(t, 0); iv != nil {
			if _, ok := iv.(*dimVarNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"for-loop operand 0 must be a dimvar"))
			}
		}
		// M1 (quick-261021-jay): trailing operands beyond (iv, bound, body)
		// are KeyedAttribute annotations — the tile-lower pass's soc/dosoc/
		// loc materialization metadata. Anything else is a shape violation.
		for i, op := range OperandsOf(t) {
			if i < 3 || op == nil {
				continue // the generic slot check owns nil operands
			}
			if _, ok := op.(*keyedAttributeNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"for-loop trailing operands must be KeyedAttribute annotations (M1)"))
				break
			}
		}
	case *intrinsicNode:
		// §5 companion 3 (quick-261021-jay): exactly one operand — the DoSoC
		// const u32; the field payload must be a known query selector.
		if len(OperandsOf(t)) != 1 {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"intrinsic map query must hold exactly one operand (the DoSoC const)"))
			break
		}
		if d := operandAt(t, 0); d != nil {
			c, ok := d.(*constNode)
			if !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"intrinsic map query operand 0 must be a const-u32 DoSoC"))
			} else if typ := c.typ; typ == nil || !typ.Equal(InternScalar(TU32)) {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"intrinsic map query operand 0 must be a const-u32 DoSoC"))
			}
		}
		switch t.field {
		case "index", "dimension":
		default:
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("intrinsic map query field %q must be \"index\" or \"dimension\"", t.field)))
		}
	case *iterClusterProjectNode:
		if cl := operandAt(t, 0); cl != nil {
			if _, ok := cl.(*iterClusterNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"iterclusterproject operand 0 must be an itercluster"))
			}
		}
	case *inductionVariableNode:
		if it := operandAt(t, 0); it != nil {
			if _, ok := it.(*iterNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"inductionvariable operand 0 must be an iter"))
			}
		}
	case *tensorDimOfNode:
		checkTensorParam(n, operandAt(t, 0), vs)
		if ax := operandAt(t, 1); ax != nil {
			if _, ok := ax.(*constNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"tensordimof operand 1 must be a const axis index"))
			}
		}
	case *asNode:
		for i, op := range OperandsOf(t) {
			if op == nil {
				continue // the generic slot check owns nil operands
			}
			if ip, ok := op.(*inputParamNode); !ok || ip.role != RoleDim {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("as operand %d must be a dim InputParam", i)))
			}
		}
	case *reduceNode:
		if attr := operandAt(t, 1); attr != nil {
			if _, ok := attr.(*keyedAttributeNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"reduce operand 1 must be a KeyedAttribute (init)"))
			}
		}
		checkAsPlacement(t, vs)
	case *mapNode:
		checkAsPlacement(t, vs)
		// ops = [value operands..., As?, lambda] (D15): strip the trailing
		// lambda node and the As layout marker to count value operands.
		valueOps := OperandsOf(t)
		if len(valueOps) > 0 && valueOps[len(valueOps)-1] != nil &&
			valueOps[len(valueOps)-1].KindOf() == KindLambda {
			valueOps = valueOps[:len(valueOps)-1]
		}
		if len(valueOps) > 0 && valueOps[len(valueOps)-1] != nil &&
			valueOps[len(valueOps)-1].KindOf() == KindAs {
			valueOps = valueOps[:len(valueOps)-1]
		}
		// The positional binding contract (Input[i] ↔ param i, D4) holds
		// while parameters remain declared. A FULLY INLINED map — the
		// output of simplify-inline (quick-261019-ktm) — declares no
		// params; its inputs are plain SSA data flow, so the count check
		// is vacuous there.
		var lamParams int
		if lam := lambdaNodeOf(t); lam != nil {
			lamParams = len(lambdaNodeParams(lam))
		}
		if lamParams > 0 && len(valueOps) != lamParams {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("map input count must equal lambda parameter count (%d inputs, %d params)",
					len(valueOps), lamParams)))
		}
	}
}

// checkAsPlacement enforces D1's optional-but-unique-and-last layout: when a
// Map/Reduce Input list carries an As node, at most one exists and it is
// the last operand BEFORE the trailing lambda node (D15: the lambda closes
// the Input list).
func checkAsPlacement(n Node, vs *[]Violation) {
	ops := OperandsOf(n)
	if lam := lambdaNodeOf(n); lam != nil && len(ops) > 0 {
		ops = ops[:len(ops)-1] // the trailing lambda is a layout marker, not a value input
	}
	last := len(ops) - 1
	for i, op := range ops {
		if op == nil || op.KindOf() != KindAs {
			continue
		}
		if i != last {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"as must be the last input"))
			return // one report per stage; duplicate As also trips this
		}
	}
}

// checkTensorParam pins an HIR GMEM access's tensor operand to a RoleTensor
// InputParam node (the kernel I/O symbol itself — the access axes are the
// dim operands that follow).
func checkTensorParam(n Node, tensor Node, vs *[]Violation) {
	if tensor == nil {
		return
	}
	ip, ok := tensor.(*inputParamNode)
	if !ok || ip.role != RoleTensor {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"tensor operand must be a RoleTensor InputParam node"))
	}
}

// checkSMemBuffer pins an LIR SMEM access's buffer operand to a MemNewSMem
// node (the lowered suite promotes the buffer into ordinary SSA data flow;
// the buffer's def must be the allocation itself).
func checkSMemBuffer(n Node, buf Node, vs *[]Violation) {
	if buf == nil {
		return
	}
	if _, ok := buf.(*memNewSMemNode); !ok {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"smem buffer operand must be a MemNewSMem node"))
	}
}

// checkFragBuffer pins an LIR fragment access's buffer operand to a FragNew
// node (the lowered suite promotes the buffer into ordinary SSA data flow;
// the buffer's def must be the allocation itself).
func checkFragBuffer(n Node, buf Node, vs *[]Violation) {
	if buf == nil {
		return
	}
	if _, ok := buf.(*fragNewNode); !ok {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"frag buffer operand must be a FragNew node"))
	}
}

// checkOneUse applies the use-def resolution checks to one use (operand slot
// or payload use): zero handles and unwired records report the
// construction-time "nil def" message; killed records are the handle model's
// corruption class; erased or unknown-kind defs are flagged. Nil-safe
// throughout.
func checkOneUse(n Node, u SSAUse, where string, vs *[]Violation) {
	var def Node
	if u.rec != nil && !u.rec.killed && u.rec.def != nil {
		def = u.rec.def.node
	}
	switch {
	case u.rec == nil || (u.rec.def == nil && !u.rec.killed):
		// zero handle, or a record unwired by SetDef(_, nil): the same
		// message construction-time unwired slots report
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s has nil def (unwired operand)", where)))
	case u.rec.killed:
		// the handle model's corruption class: a removed record
		// still referenced from a live slot
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s holds a killed use record (removed but still referenced)", where)))
	case def == nil:
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s has nil def (unwired operand)", where)))
	case def.BasePtr().erased:
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s points at erased node %d", where, def.BasePtr().id)))
	case def.KindOf() == KindInvalid || def.KindOf() >= KindEnd:
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s points at node %d of unknown kind %d",
				where, def.BasePtr().id, def.KindOf())))
	}
}

// checkDefUse checks the def-use direction for every visited node: the
// node's SSAVar ref list must contain exactly the records whose slots point
// back at it, counting only users the walk actually visited. Over-links (a
// listed record pointing elsewhere, or a listed record no walked operand
// slot owns) and under-links (an operand record missing from the def's list)
// are both reported — corrupting either direction is caught by this walk.
// Level-1 note: the sea-of-nodes walk roots at the params and the end node,
// so a dead-but-not-yet-DCE'd user is unvisited; its records in a live
// def's ref list are skipped here (they are the unreachable subgraph's
// bookkeeping, killed when DCE erases the user), not flagged as over-links.
func checkDefUse(nodes []Node, vs *[]Violation) {
	expected := make(map[*SSAVar][]*ssaRefRecord)
	walked := make(map[Node]bool, len(nodes))
	for _, n := range nodes {
		walked[n] = true
		if impl := asImpl(n); impl != nil {
			impl.forEachOperand(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					expected[u.rec.def] = append(expected[u.rec.def], u.rec)
				}
			})
		}
		if pb, ok := n.(payloadBearer); ok {
			pb.forEachPayloadUse(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					expected[u.rec.def] = append(expected[u.rec.def], u.rec)
				}
			})
		}
	}
	for _, n := range nodes {
		nb := n.BasePtr()
		v := nb.ssa
		if v == nil {
			continue // a foreign node without a var has no ref list to check
		}
		found := make(map[*ssaRefRecord]bool, len(expected[v]))
		for _, r := range v.refs {
			if r == nil || r.killed {
				continue
			}
			if r.user != nil && !walked[r.user] {
				continue // unreachable user's record: not part of the verified subgraph
			}
			if r.def != v {
				*vs = append(*vs, nodeViolation(r.user, invDefUse,
					fmt.Sprintf("use-list of node %d contains a use pointing at node %s instead",
						nb.id, nodeIDStr(varNode(r.def)))))
			} else if !slices.Contains(expected[v], r) {
				*vs = append(*vs, nodeViolation(r.user, invDefUse,
					fmt.Sprintf("use-list of node %d contains a use that no operand slot owns",
						nb.id)))
			}
			found[r] = true
		}
		for _, r := range expected[v] {
			if !found[r] {
				*vs = append(*vs, nodeViolation(r.user, invDefUse,
					fmt.Sprintf("operand use of node %d is missing from node %d's use-list",
						nb.id, nb.id)))
			}
		}
	}
}

// varNode resolves a var's owning node, nil-safely.
func varNode(v *SSAVar) Node {
	if v == nil {
		return nil
	}
	return v.node
}
