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
// violation found, in deterministic order: basic-block statement order,
// then per-node operand order — never map iteration (RESEARCH Pitfall 1).
// A nil or empty result means the chains are consistent.
func Verify(g *Graph) []Violation {
	if g == nil {
		return nil
	}
	var vs []Violation
	nodes := walkCheckUseDef(g, &vs)
	checkDefUse(nodes, &vs)
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

// walkCheckUseDef walks the graph from the block's statements (via Walk,
// directive 3), checking the use-def direction at every node: each operand
// slot's handle must resolve to a live (not tombstoned) def of a kind <
// KindEnd. Zero handles and unwired records report the construction-time
// "nil def" message; killed records are the handle model's new corruption
// class, flagged by slot. It returns the visited nodes in walk order for the
// def-use pass. Lambda bodies are walked as edges alongside operand uses.
func walkCheckUseDef(g *Graph, vs *[]Violation) []Node {
	var nodes []Node
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		nodes = append(nodes, n)
		checkNodeUseDef(n, vs)
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
			"erased node is still reachable from the block"))
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
// back at it. Over-links (a listed record pointing elsewhere, or a listed
// record no operand slot owns) and under-links (an operand record missing
// from the def's list) are both reported — corrupting either direction is
// caught by this walk.
func checkDefUse(nodes []Node, vs *[]Violation) {
	expected := make(map[*SSAVar][]*ssaRefRecord)
	for _, n := range nodes {
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
