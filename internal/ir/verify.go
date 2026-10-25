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

// Invariant names for the two chain directions checked on every graph, plus
// the LIR loop-structure invariants (quick-261021-o61 T4).
const (
	invUseDef = "use-def" // every operand handle resolves to a live def of a known kind
	invDefUse = "def-use" // every def's ref list matches the operand slots that own its records

	// invLoopStep is the positive-step invariant: a live ForLoop's dimvar
	// step must be a positive constant (HI-04's IR-level backstop —
	// tiling-verify checks the tiling records; this is the graph itself).
	invLoopStep = "loop-step"
	// invScopeDominance is the R-I scope invariant (fix3 D-l): every
	// AST-placed memory/coordinate consumer must be dominated by the loops
	// its operand closure references — the `dimvar ... has no enclosing
	// loop in scope` failure class moved from the backend emitter into the
	// verifier, so the PASS that breaks the invariant is the one that
	// fails. Active POST-SCHEDULE only (it reads SchedBlock stamps over
	// the attached AST; the emitter's own gate stays as the backstop).
	invScopeDominance = "scope-dominance"
	// (invLoopScope deleted — fix1.md D-c ruling: with no body operand,
	// the containment-based scoping check is vacuous. The tiling §8 SCOPE
	// check (coordinates reference only enclosing loops) guards the
	// placement-level hazard; membership is via IV references.)
)

// Verify runs both chain-consistency checkers plus the loop-structure
// invariants over g and returns every violation found, in deterministic walk
// order: the level-aware roots (params plus the end node at Level 1, params
// plus statement order at Level 2), then per-node operand order — never map
// iteration (RESEARCH Pitfall 1). A nil or empty result means the chains are
// consistent. Verify needs no Materialize: the Level-1 roots cover the whole
// constructed graph.
func Verify(g *Graph) []Violation {
	if g == nil {
		return nil
	}
	var vs []Violation
	nodes := walkCheckUseDef(g, &vs)
	checkDefUse(nodes, &vs)
	checkLoopInvariants(nodes, &vs)
	checkScopeDominance(g, &vs)
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
// SMEM-class accesses bind an SMEM-class MemNew buffer; FRAG-class accesses bind a
// FRAG-class MemNew buffer; a for-loop binds a dimvar, with trailing operands beyond
// (iv, bound, body) pinned to KeyedAttribute annotations (M1,
// quick-261021-jay); an LIR store's trailing KeyedAttribute operands (the
// fill annotation convention, LO-01/quick-261021-o61 T5) are pinned
// well-formed — at most one per key, each with a const value; a runtime map
// query holds either exactly one const-u32 DoSoC operand and a field in
// {index, dimension}, or the zero-operand FLAT form (field "index" only —
// the flat-index() ruling 2026-10-23) (§5 companion 3); the
// runtime-object family's P-3 shape cross-checks (KernelInfo operand-free,
// mapping portion grammar, reducer (init, lambda) pair, closed method/
// attribute vocabularies with their object domains); an
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
			// An InputParam (kernel/dim/tensor/scalar/lambda symbol) or a
			// MemNew allocation: pkg.Builder.Local backs `let`-locals with
			// GMEM allocations, and a map operand references the local's
			// def through a paramRef exactly like a parameter (syntax-
			// design.md §6 Task 4).
			if _, ok := p.(*inputParamNode); !ok {
				if _, ok := p.(*memNewNode); !ok {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						"param operand 0 must be an InputParam node or a local MemNew allocation"))
				}
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
		switch t.class {
		case StorageGMEM:
			checkTensorParam(n, operandAt(t, 0), vs)
		case StorageSMem:
			checkSMemBuffer(n, operandAt(t, 0), vs)
		case StorageFrag:
			checkFragBuffer(n, operandAt(t, 0), vs)
		}
	case *memStoreNode:
		switch t.class {
		case StorageGMEM:
			checkTensorParam(n, operandAt(t, 1), vs)
		case StorageSMem:
			checkSMemBuffer(n, operandAt(t, 1), vs)
			checkStoreAnnotations(n, vs)
		case StorageFrag:
			checkFragBuffer(n, operandAt(t, 1), vs)
			checkStoreAnnotations(n, vs)
		}
	case *forLoopNode:
		if iv := operandAt(t, 0); iv != nil {
			if _, ok := iv.(*dimVarNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"for-loop operand 0 must be a dimvar"))
			}
		}
		// M1 (quick-261021-jay): trailing operands beyond (iv, bound)
		// are KeyedAttribute annotations — the tile-lower pass's soc/dosoc/
		// loc materialization metadata. Anything else is a shape violation.
		// (No body operand: sea-of-nodes ruling, fix1.md §0.)
		for i, op := range OperandsOf(t) {
			if i < 2 || op == nil {
				continue // the generic slot check owns nil operands
			}
			if _, ok := op.(*keyedAttributeNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					"for-loop trailing operands must be KeyedAttribute annotations (M1)"))
				break
			}
		}
	case *runtimeQueryNode:
		// §5 companion 3 (quick-261021-jay, renamed LR-6), extended by the
		// flat-index() ruling (2026-10-23): EITHER exactly one operand — the
		// DoSoC const u32 — OR the ZERO-OPERAND flat form, legal ONLY for
		// field "index" (the absolute flattened 1-D rank). The field payload
		// must be a known query selector.
		switch {
		case len(OperandsOf(t)) == 0:
			// The flat form: index-only ("dimension" demands the DoSoC).
			if t.field != "index" {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("runtime map query field %q carries no DoSoC operand — the zero-operand flat form is legal only for field \"index\"", t.field)))
			}
		case len(OperandsOf(t)) == 1:
			if d := operandAt(t, 0); d != nil {
				c, ok := d.(*constNode)
				if !ok {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						"runtime map query operand 0 must be a const-u32 DoSoC"))
				} else if typ := c.typ; typ == nil || !typ.Equal(InternScalar(TU32)) {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						"runtime map query operand 0 must be a const-u32 DoSoC"))
				}
			}
		default:
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"runtime map query must hold exactly one operand (the DoSoC const) or none (the flat index form)"))
		}
		switch t.field {
		case "index", "dimension":
		default:
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("runtime map query field %q must be \"index\" or \"dimension\"", t.field)))
		}
	case *runtimeKernelInfoNode:
		// LR-4 (quick-261022-l1o): the kernel information object carries NO
		// operands — any operand is a shape violation (the fixed-arity-0
		// surplus check also fires on the raw NewNode path).
		if len(OperandsOf(t)) != 0 {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"runtimekernelinfo must carry no operands"))
		}
	case *runtimeMappingNode:
		// P-3 (quick-261022-l1o): ops = [kernel_info, ndim const TI32,
		// portion consts TI32 ...] with len(portions) == ndim >= 1, each
		// portion >= 1, and -1 allowed ONLY as the last portion (the D3
		// remainder rule).
		checkRuntimeMappingShape(n, t, vs)
	case *runtimeReducerNode:
		// LR-3/P-3 + R1/R2 (2026-10-23): the sequential specialization
		// carries exactly (typed init value, KindLambda); the warp/global
		// specializations carry exactly (typed init, KindLambda,
		// RuntimeThreadMapping, staging SMEM-class MemNew). Swapped, missing,
		// or mis-typed operands are shape violations. A missing operand
		// leaves an unwired slot (the generic nil-def check also fires);
		// the count check reads OperandsOf, whose slot count covers
		// unwired slots, so probe nil explicitly for the named message.
		want := 2
		if n.KindOf() == KindRuntimeWarpReducer || n.KindOf() == KindRuntimeGlobalReducer {
			want = 4
		}
		ops := OperandsOf(t)
		if len(ops) != want {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("%s must carry exactly %d operands, got %d", n.KindOf(), want, len(ops))))
			break
		}
		for i, op := range ops {
			if op == nil {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("%s operand %d is unwired", n.KindOf(), i)))
				break
			}
		}
		if init := operandAt(t, 0); init != nil && TypeOf(init) == nil && ConstTypeOf(init) == nil {
			// "Typed" reads either the computed cell (the real flow:
			// typeinfer ran long before final-lower) or a const's interned
			// payload type (the fixture vocabulary). Constructions only
			// exist after final-lower, so real graphs always satisfy the
			// first; the second keeps hand-built test fixtures verifiable.
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"reducer construction operand 0 must be a typed init value"))
		}
		if lam := operandAt(t, 1); lam != nil && lam.KindOf() != KindLambda {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"reducer construction operand 1 must be the combiner KindLambda node"))
		}
		if want == 4 {
			if tm := operandAt(t, 2); tm != nil && tm.KindOf() != KindRuntimeThreadMapping {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("%s operand 2 must be the owning LNT's RuntimeThreadMapping (R2), got %s", n.KindOf(), tm.KindOf())))
			}
			if st := operandAt(t, 3); st != nil {
				if sc, _ := StorageClassOf(st); st.KindOf() != KindMemNew || sc != StorageSMem {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						fmt.Sprintf("%s operand 3 must be the staging SMEM-class MemNew (R1), got %s", n.KindOf(), st.KindOf())))
				}
			}
		}
	case *runtimeCallNode:
		// P-3: the method vocabulary is closed {Init, Reduce, Combine} and
		// applies to a REDUCER CONSTRUCTION object only (the sole method
		// domain of v0's runtime surface).
		switch t.method {
		case "Init", "Reduce", "Combine":
		default:
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("runtimecallmethod payload %q must be one of \"Init\", \"Reduce\", \"Combine\"", t.method)))
		}
		if obj := operandAt(t, 0); obj != nil && !isRuntimeReducerKind(obj.KindOf()) {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("runtimecallmethod operand 0 must be a reducer construction object, got %s", obj.KindOf())))
		}
	case *runtimeGetAttributeNode:
		// P-3: the attribute vocabulary is closed; "total_*" applies to a
		// KernelInfo object with NO extra args; "index"/"dimension" to a
		// mapping object with EXACTLY ONE const-u32 arg.
		switch t.attribute {
		case "total_cta", "total_thread_per_cta":
			if obj := operandAt(t, 0); obj != nil && obj.KindOf() != KindRuntimeKernelInfo {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("runtimegetattribute %q requires a KernelInfo object, got %s", t.attribute, obj.KindOf())))
			}
			if extra := len(OperandsOf(t)) - 1; extra != 0 {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("runtimegetattribute %q carries %d extra args, want 0", t.attribute, extra)))
			}
		case "index", "dimension":
			if obj := operandAt(t, 0); obj != nil && !isRuntimeMappingKind(obj.KindOf()) {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("runtimegetattribute %q requires a mapping object, got %s", t.attribute, obj.KindOf())))
			}
			extra := len(OperandsOf(t)) - 1
			// The flat-index() ruling (2026-10-23): attribute "index" admits
			// the ZERO-ARG form (the flattened 1-D rank); "dimension" keeps
			// exactly one const-u32 DoSoC.
			if t.attribute == "index" && extra == 0 {
				break
			}
			if extra != 1 {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("runtimegetattribute %q carries %d extra args, want exactly 1 (the DoSoC const)", t.attribute, extra)))
				break
			}
			if d := operandAt(t, 1); d != nil {
				if _, ok := constValueOf(d); !ok {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						fmt.Sprintf("runtimegetattribute %q arg must be a const DoSoC", t.attribute)))
				} else if ct := ConstTypeOf(d); ct == nil || !ct.Equal(InternScalar(TU32)) {
					*vs = append(*vs, nodeViolation(n, invUseDef,
						fmt.Sprintf("runtimegetattribute %q arg must be a const-u32 DoSoC", t.attribute)))
				}
			}
		default:
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("runtimegetattribute payload %q must be one of \"total_cta\", \"total_thread_per_cta\", \"index\", \"dimension\"", t.attribute)))
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

// checkStoreAnnotations pins the tiered store annotation convention (LO-01,
// quick-261021-o61 T5): an SMEM/FRAG-class MemStore's trailing operands
// beyond the fixed [value, dst] slots are its coordinates / fill round bases
// plus at most one KeyedAttribute per key, and every such annotation must
// carry a const value — a non-const or duplicated key is a codegen-facing
// contract break, not a style issue. (The retired v0-era loader-selection
// attribute left the IR with the fix3 D-h unification — loader selection is
// derived at codegen from the storage class; no annotation rides the fill.)
func checkStoreAnnotations(n Node, vs *[]Violation) {
	seen := make(map[string]bool)
	for i, op := range OperandsOf(n) {
		if i < 2 || op == nil {
			continue // [value, dst] slots; nil left to the generic slot check
		}
		attr, ok := op.(*keyedAttributeNode)
		if !ok {
			continue // a biv coordinate / round base: not an annotation
		}
		if seen[attr.key] {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("%s trailing operand %d repeats KeyedAttribute key %q", n.KindOf(), i, attr.key)))
			continue
		}
		seen[attr.key] = true
		if v := operandAt(attr, 0); v != nil {
			if _, ok := v.(*constNode); !ok {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("%s trailing KeyedAttribute %q must carry a const value", n.KindOf(), attr.key)))
			}
		}
	}
}

// checkRuntimeMappingShape pins the P-3 mapping constructor encoding
// (quick-261022-l1o): ops = [kernel_info, ndim const TI32, portion consts
// TI32 ...] with len(portions) == ndim >= 1, each portion >= 1, and -1 (the
// remainder) allowed ONLY as the last portion (the D3 remainder rule). The
// dedicated kind is what makes this verifier-pinnable (LR-4).
func checkRuntimeMappingShape(n Node, t *runtimeMappingNode, vs *[]Violation) {
	ops := OperandsOf(t)
	if len(ops) < 2 {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s must carry (kernel_info, ndim, portions...), got %d operands", n.KindOf(), len(ops))))
		return
	}
	if ki := ops[0]; ki != nil && ki.KindOf() != KindRuntimeKernelInfo {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s operand 0 must be a RuntimeKernelInfo node, got %s", n.KindOf(), ki.KindOf())))
	}
	// The ndim const: const, TI32, >= 1.
	nd, ok := ConstValueOf(ops[1])
	if !ok {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s operand 1 (ndim) must be a const", n.KindOf())))
		return
	}
	if ct := ConstTypeOf(ops[1]); ct == nil || !ct.Equal(InternScalar(TI32)) {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s operand 1 (ndim) must be a const-i32", n.KindOf())))
		return
	}
	if nd < 1 {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s ndim must be >= 1, got %v", n.KindOf(), nd)))
		return
	}
	portions := ops[2:]
	if float64(len(portions)) != nd {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			fmt.Sprintf("%s lists %d portions but declares ndim %v", n.KindOf(), len(portions), nd)))
		return
	}
	for i, p := range portions {
		v, ok := ConstValueOf(p)
		if !ok {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("%s portion %d must be a const", n.KindOf(), i)))
			continue
		}
		if ct := ConstTypeOf(p); ct == nil || !ct.Equal(InternScalar(TI32)) {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("%s portion %d must be a const-i32", n.KindOf(), i)))
			continue
		}
		if v == -1 {
			if i != len(portions)-1 {
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("%s portion %d is -1 (the remainder): only the LAST portion may be -1 (the D3 remainder rule)", n.KindOf(), i)))
			}
			continue // the legal last-position remainder
		}
		if v < 1 {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				fmt.Sprintf("%s portion %d must be >= 1 (or -1 as the last portion), got %v", n.KindOf(), i, v)))
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
// dim operands that follow) or to a MemNew allocation (a pkg.Builder.Local
// `let`-local — conceptually GMEM, so its allocation is a legitimate access
// target, syntax-design.md §6 Task 4).
func checkTensorParam(n Node, tensor Node, vs *[]Violation) {
	if tensor == nil {
		return
	}
	ip, ok := tensor.(*inputParamNode)
	if !ok {
		if _, ok := tensor.(*memNewNode); !ok {
			*vs = append(*vs, nodeViolation(n, invUseDef,
				"tensor operand must be a RoleTensor InputParam node or a local MemNew allocation"))
		}
		return
	}
	if ip.role != RoleTensor {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"tensor operand must be a RoleTensor InputParam node"))
	}
}

// checkSMemBuffer pins an SMEM-class access's buffer operand to an
// SMEM-class MemNew node (the tiered constructors promote the buffer into
// ordinary SSA data flow; the buffer's def must be the allocation itself,
// and the classes must agree).
func checkSMemBuffer(n Node, buf Node, vs *[]Violation) {
	if buf == nil {
		return
	}
	b, ok := buf.(*memNewNode)
	if !ok || b.class != StorageSMem {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"smem buffer operand must be an SMEM-class MemNew node"))
	}
}

// checkFragBuffer pins a FRAG-class access's buffer operand to a FRAG-class
// MemNew node (the tiered constructors promote the buffer into ordinary SSA
// data flow; the buffer's def must be the allocation itself, and the
// classes must agree).
func checkFragBuffer(n Node, buf Node, vs *[]Violation) {
	if buf == nil {
		return
	}
	b, ok := buf.(*memNewNode)
	if !ok || b.class != StorageFrag {
		*vs = append(*vs, nodeViolation(n, invUseDef,
			"frag buffer operand must be a FRAG-class MemNew node"))
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

// checkLoopInvariants enforces the LIR loop invariants over every live
// ForLoop the walk visited, in walk order (deterministic):
//
//   - loop-step (HI-04's IR-level half): the dimvar's step operand must be a
//     constant > 0 — a zero or negative stride is an infinite or backwards
//     loop, and a non-constant step has no materializable round stride.
//
// (loop-scope deleted by fix1.md D-c ruling: with no body operand, the
// containment-based scoping check is vacuous. The tiling §8 SCOPE check
// guards the placement-level hazard; membership is via IV references.)
//
// Users the walk never reached are skipped (dead-subgraph bookkeeping, the
// checkDefUse precedent), as are erased/detached users.
func checkLoopInvariants(nodes []Node, vs *[]Violation) {
	for _, n := range nodes {
		if n.KindOf() != KindForLoop {
			continue
		}
		checkLoopStep(n, vs)
	}
}

// forLoopIV resolves loop's IV dimvar node, or nil when the operand is
// missing or not a dimvar (the generic slot and shape checks own those).
func forLoopIV(loop Node) *dimVarNode {
	iv := operandAt(loop, 0)
	if iv == nil {
		return nil
	}
	dv, _ := iv.(*dimVarNode)
	return dv
}

// (checkLoopScope deleted — fix1.md D-c ruling: with no body operand, the
// containment-based scoping check is vacuous. The tiling §8 SCOPE check
// guards the placement-level hazard; membership is via IV references.)

// checkLoopStep enforces loop-step for one live ForLoop: its dimvar's step
// operand must be a positive constant.
func checkLoopStep(loop Node, vs *[]Violation) {
	dv := forLoopIV(loop)
	if dv == nil {
		return
	}
	step := operandAt(dv, 2)
	if step == nil {
		return // unwired step: the generic use-def slot check owns it
	}
	v, ok := constValueOf(step)
	if !ok {
		*vs = append(*vs, nodeViolation(loop, invLoopStep,
			fmt.Sprintf("for-loop node %d's dimvar node %d carries a non-constant step (node %s) — a lowered loop's induction step must be a positive constant",
				loop.BasePtr().id, dv.BasePtr().id, nodeIDStr(step))))
		return
	}
	if v <= 0 {
		*vs = append(*vs, nodeViolation(loop, invLoopStep,
			fmt.Sprintf("for-loop node %d's dimvar node %d carries step %v — a lowered loop's induction step must be a positive constant (zero or negative strides are infinite or backwards loops)",
				loop.BasePtr().id, dv.BasePtr().id, v)))
	}
}

// checkScopeDominance enforces the R-I scope invariant (fix3 D-l) over the
// attached AST — POST-SCHEDULE position only: a graph with no AST carries
// no SchedBlock stamps and no lexical positions, so the check is inactive
// (the sea's membership is IV-reference based, never positional; fix1.md
// D-c). For every AST item node, every DimVar in its operand closure must
// be defined at a block DOMINATING the item's block (ancestor-or-self):
// a re-placed or hoisted consumer that escapes a loop its coordinates
// read would render as an out-of-scope iterator reference at codegen —
// exactly the class the emitter's fail-closed gate catches late (expr.go:
// `dimvar ... has no enclosing loop in scope`; that gate stays as the
// backstop). A loop block's BOUND expression evaluates in the PARENT
// scope, so its closure's dimvars are checked against the parent's
// ancestry (never the loop's own IV). Unscheduled dimvars (nil SchedBlock)
// are skipped: their placement is not yet decided, and the pre-schedule
// world has no positions at all.
func checkScopeDominance(g *Graph, vs *[]Violation) {
	k := g.AST()
	if k == nil || k.Body == nil {
		return
	}
	// blockDominates reports whether maybeAncestor is needle or an ancestor
	// of it on the AST's parent chain.
	var parentOf func(blk, needle *ASTBlock) *ASTBlock
	parentOf = func(blk, needle *ASTBlock) *ASTBlock {
		for _, it := range blk.Items() {
			if it.Block != nil {
				if it.Block == needle {
					return blk
				}
				if p := parentOf(it.Block, needle); p != nil {
					return p
				}
			}
		}
		return nil
	}
	dominates := func(needle, maybeAncestor *ASTBlock) bool {
		for b := needle; b != nil; b = parentOf(k.Body, b) {
			if b == maybeAncestor {
				return true
			}
		}
		return false
	}
	reportItem := func(n Node, blk *ASTBlock) {
		switch n.KindOf() {
		case KindReturn, KindKeyedAttribute:
			// Non-emitting carriers: the return renders nothing in the
			// kernel body (the launcher owns outputs; its operands are the
			// output bindings' stores, whose coordinate closures are the
			// STORES' concern, checked at their own positions), and an
			// annotation carrier rides its host's operand list.
			return
		}
		for d := range operandClosureOf(n) {
			if d.KindOf() != KindDimVar {
				continue
			}
			def := d.BasePtr().SchedBlock()
			if def == nil {
				continue // unscheduled: no position to dominate
			}
			if !dominates(blk, def) {
				*vs = append(*vs, nodeViolation(n, invScopeDominance,
					fmt.Sprintf("%s node %d (block %s) references dimvar node %d whose loop does not dominate its position — the coordinate is not in scope",
						n.KindOf(), n.BasePtr().id, blk.Kind(), d.BasePtr().id)))
			}
		}
	}
	var walk func(blk *ASTBlock)
	walk = func(blk *ASTBlock) {
		parent := parentOf(k.Body, blk)
		// A loop block's header expressions (the bound's max) evaluate in
		// the PARENT scope.
		if l := blk.Loop(); l != nil && parent != nil {
			if bops := OperandsOf(blk.Bound()); len(bops) == 2 && bops[1] != nil {
				for d := range operandClosureOf(bops[1]) {
					if d.KindOf() != KindDimVar {
						continue
					}
					def := d.BasePtr().SchedBlock()
					if def == nil || dominates(parent, def) {
						continue
					}
					*vs = append(*vs, nodeViolation(l, invScopeDominance,
						fmt.Sprintf("forloop node %d's bound references dimvar node %d whose loop does not dominate the loop header's scope",
							l.BasePtr().id, d.BasePtr().id)))
				}
			}
		}
		for _, it := range blk.Items() {
			if it.Node != nil {
				reportItem(it.Node, blk)
				continue
			}
			if it.Block != nil {
				walk(it.Block)
			}
		}
	}
	walk(k.Body)
}

// operandClosureOf collects n's transitive operand-edge closure (self
// included, live nodes only) — the verifier-side twin of the passes'
// operandClosure helpers.
func operandClosureOf(n Node) map[Node]bool {
	set := make(map[Node]bool)
	var visit func(x Node)
	visit = func(x Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || set[x] {
			return
		}
		set[x] = true
		for _, d := range OperandsOf(x) {
			visit(d)
		}
	}
	visit(n)
	return set
}

// childClosure returns the set of nodes reachable from start over the walk's
// child relation — live operand defs in slot order, then live payload-use
// defs (the ImpactBy effect chains included) — plus start itself, skipping
// erased nodes. It is the Level-1 sea's structural-containment carrier:
// "inside a loop" means reachable from the loop's operand/payload edges,
// never materialized statement order. Iterative with a visited set, so
// malformed (even cyclic) graphs cannot overflow the stack (T-1.1-02).
func childClosure(start Node) map[Node]bool {
	seen := make(map[Node]bool)
	if start == nil {
		return seen
	}
	stack := []Node{start}
	push := func(n Node) {
		if n != nil && !seen[n] {
			stack = append(stack, n)
		}
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[n] || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		seen[n] = true
		if impl := asImpl(n); impl != nil {
			impl.forEachOperand(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					push(u.rec.def.node)
				}
			})
		}
		if pb, ok := n.(payloadBearer); ok {
			pb.forEachPayloadUse(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					push(u.rec.def.node)
				}
			})
		}
	}
	return seen
}

// useSlotOf names where user holds a use of def: the first operand slot that
// resolves at def, else the first payload use, else "an unknown slot" (a
// corrupted record can dangle from both directions). It fills the offending
// use's slot into the loop-scope diagnostic.
func useSlotOf(user, def Node) string {
	where := ""
	if impl := asImpl(user); impl != nil {
		slot := 0
		impl.forEachOperand(func(u SSAUse) {
			if where != "" {
				return
			}
			if d := u.Def(); d != nil && d.BasePtr() == def.BasePtr() {
				where = fmt.Sprintf("operand slot %d", slot)
			}
			slot++
		})
	}
	if pb, ok := user.(payloadBearer); ok && where == "" {
		idx := 0
		pb.forEachPayloadUse(func(u SSAUse) {
			if where != "" {
				return
			}
			if d := u.Def(); d != nil && d.BasePtr() == def.BasePtr() {
				where = fmt.Sprintf("payload use %d", idx)
			}
			idx++
		})
	}
	if where == "" {
		where = "an unknown slot"
	}
	return where
}
