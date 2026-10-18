package ir

// node_end.go holds the graph's end node (KindReturn, D-3). R2 (landed in
// quick-261018-bl0 task 2) makes the return THE graph's single end node: it
// carries the kernel's return values AND, via its ImpactBy payload, the final
// state of every mem-param's effect chain — the two-root-set DCE model's
// (R3) second root beside the RoleTensor params themselves. Effectful nodes
// are NOT roots: stores stay live only through the end node's operand and
// ImpactBy paths (or are unreachable building debris, wiped by Level-1 DCE).

// returnNode: the graph's end node; ops = the returned values. Return is the
// none Value (directive 2): it satisfies Value so it can sit in a statement
// RHS slot, and IsNone() reports that it binds no SSA name. R2's effectIns
// ImpactBy payload carries one edge per mem-param's final chain state (wired
// by NewReturnChained).
type returnNode struct {
	*Base
	operandList
	effectIns []SSAUse // R2: one ImpactBy edge per mem-param's final chain state
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
// return created on a graph becomes its end node (R2 — Graph.End); later
// returns allocate ordinary nodes, which the verifier rejects under its
// multiple-end-nodes check. The end node's liveness wiring — ImpactBy edges
// carrying every mem-param's final chain state — arrives through
// NewReturnChained.
func (g *Graph) NewReturn(origin string, values ...Node) Node {
	n := g.NewNode(KindReturn, origin, values...)
	if g.end == nil {
		g.end = n
	}
	return n
}

// End returns the graph's single end node (R2), or nil before any return has
// been created. It is the sea-of-nodes' second root beside the RoleTensor
// params (R3's two-root set).
func (g *Graph) End() Node {
	if g == nil {
		return nil
	}
	return g.end
}

// NewReturnChained creates the graph's end node exactly like NewReturn and
// additionally wires it into the construction environment's per-tensor
// effect chains: for each tensor in tensors, the end node's ImpactBy payload
// points at that tensor's CURRENT chain head (one SSAUse per tensor, heads
// only — never intermediate links). R2: the end node thereby observes every
// mem-param's final chain state, so the whole per-tensor store chain stays
// live through the end node alone; a tensor with no recorded access
// contributes no edge. The returned values stay ordinary operand slots.
func (g *Graph) NewReturnChained(origin string, env *Env, tensors []Node, values ...Node) Node {
	n := g.NewReturn(origin, values...).(*returnNode)
	for _, t := range tensors {
		if head := env.ChainHead(t); head != nil {
			n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
		}
	}
	return n
}
