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
	invUseDef = "use-def" // every Use points at a live def of a known kind
	invDefUse = "def-use" // every def's use-list matches the uses pointing at it
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

// walkCheckUseDef walks the graph from the block's statements, checking the
// use-def direction at every node: each operand Use must point at a non-nil,
// live (not tombstoned) def of a kind < KindEnd. It returns the visited
// nodes in walk order for the def-use pass. Lambda bodies are walked as
// edges alongside operand uses. The walk is iterative with a visited set, so
// malformed (cyclic) graphs cannot blow the stack (T-1-01-02).
func walkCheckUseDef(g *Graph, vs *[]Violation) []Node {
	seen := make(map[Node]bool)
	var queue []Node
	enqueue := func(n Node) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		queue = append(queue, n)
	}
	for _, s := range g.block.statements {
		enqueue(s.def)
		enqueue(s.rhs)
	}
	var nodes []Node
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		nodes = append(nodes, n)
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
		slot := 0
		asImpl(n).forEachOperand(func(u *Use) {
			switch {
			case u.Val == nil:
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("operand slot %d has nil def (unwired operand)", slot)))
			case u.Val.BasePtr().erased:
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("operand slot %d points at erased node %d", slot, u.Val.BasePtr().id)))
			case u.Val.KindOf() == KindInvalid || u.Val.KindOf() >= KindEnd:
				*vs = append(*vs, nodeViolation(n, invUseDef,
					fmt.Sprintf("operand slot %d points at node %d of unknown kind %d",
						slot, u.Val.BasePtr().id, u.Val.KindOf())))
			}
			enqueue(u.Val)
			slot++
		})
		if lb, ok := n.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { enqueue(lam.Body) })
		}
	}
	return nodes
}

// checkDefUse checks the def-use direction for every visited node: the
// node's intrusive use-list must contain exactly the Use records whose Val
// points back at it. Over-links (a listed use pointing elsewhere, or a
// listed use no operand slot owns) and under-links (an operand use missing
// from the def's list) are both reported — corrupting either direction is
// caught by this walk.
func checkDefUse(nodes []Node, vs *[]Violation) {
	expected := make(map[Node][]*Use)
	for _, n := range nodes {
		asImpl(n).forEachOperand(func(u *Use) {
			if u.Val != nil {
				expected[u.Val] = append(expected[u.Val], u)
			}
		})
	}
	for _, n := range nodes {
		nb := n.BasePtr()
		found := make(map[*Use]bool, len(expected[n]))
		for u := nb.firstUse; u != nil; u = u.next {
			if u.Val != n {
				*vs = append(*vs, nodeViolation(u.User, invDefUse,
					fmt.Sprintf("use-list of node %d contains a use pointing at node %s instead",
						nb.id, nodeIDStr(u.Val))))
			} else if !slices.Contains(expected[n], u) {
				*vs = append(*vs, nodeViolation(u.User, invDefUse,
					fmt.Sprintf("use-list of node %d contains a use that no operand slot owns",
						nb.id)))
			}
			found[u] = true
		}
		for _, u := range expected[n] {
			if !found[u] {
				*vs = append(*vs, nodeViolation(u.User, invDefUse,
					fmt.Sprintf("operand use of node %d is missing from node %d's use-list",
						nb.id, nb.id)))
			}
		}
	}
}
