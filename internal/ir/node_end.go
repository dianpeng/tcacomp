package ir

// node_end.go holds the graph's end node (KindReturn, D-3). R2 (landed in
// quick-261018-bl0 task 2) makes the return THE graph's single end node: it
// carries the kernel's return values AND, via its ImpactBy payload, the final
// state of every mem-param's effect chain — the two-root-set DCE model's
// (R3) second root beside the RoleTensor params themselves.

// returnNode: the graph's end node; ops = the returned values. Return is the
// none Value (directive 2): it satisfies Value so it can sit in a statement
// RHS slot, and IsNone() reports that it binds no SSA name. R2 adds the
// effectIns ImpactBy payload — the per-tensor final-chain-state edges wiring
// the end node into every mem-param's effect chain (task 2).
type returnNode struct {
	*Base
	operandList
	effectIns []SSAUse // R2: one ImpactBy edge per mem-param's final chain state (task 2)
}

// IsNone reports the none Value semantics of the return root.
func (*returnNode) IsNone() bool { return true }

// forEachPayloadUse implements payloadBearer: the ImpactBy effect inputs
// (R2 — the end node's mem-param final chain states).
func (n *returnNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.effectIns {
		fn(u)
	}
}

// impactBy implements effectBearer: a copy of the end node's ordering edges.
func (n *returnNode) impactBy() []SSAUse { return append([]SSAUse(nil), n.effectIns...) }

// NewReturn creates the graph's end node over the returned values. The FIRST
// return created on a graph becomes its end node (R2); the graph's effect
// chain wiring (NewReturnChained) and the two-root-set DCE root there.
func (g *Graph) NewReturn(origin string, values ...Node) Node {
	return g.NewNode(KindReturn, origin, values...)
}
