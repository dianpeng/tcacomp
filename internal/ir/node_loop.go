package ir

// node_loop.go holds the loop-structure nodes: the HIR abstract iteration
// suite (KindIter/KindIterCluster) and the LIR explicit loop nest
// (KindForLoop, D-3). The LIR induction-variable VALUE objects (KindDimVar/
// KindBoundedDimVar) are value-level material, not loop-nest structure — they
// live with their relations in node_dimvar.go.

// iterNode: abstract induction variable; ops = [init, step]; dim/bound are
// payload uses of dim InputParam nodes (bound unwired = unbounded).
type iterNode struct {
	*Base
	operandList
	dimUse   SSAUse
	boundUse SSAUse
}

// forEachPayloadUse implements payloadBearer: dim then bound.
func (n *iterNode) forEachPayloadUse(fn func(SSAUse)) {
	fn(n.dimUse)
	fn(n.boundUse)
}

// NewIter creates an abstract induction-variable node over the dim
// InputParam node dim, bounded by the InputParam node bound (nil = the
// unbounded form @x), with init and step operands.
func (g *Graph) NewIter(origin string, dim, bound Node, init, step Node) Node {
	n := g.NewNode(KindIter, origin, init, step).(*iterNode)
	n.dimUse = wirePayloadUse(n, dim)
	n.boundUse = wirePayloadUse(n, bound)
	return n
}

// iterClusterNode: abstract compound header; ops = [body]; dims are payload
// uses of dim InputParam nodes.
type iterClusterNode struct {
	*Base
	operandList
	dims []SSAUse
}

// forEachPayloadUse implements payloadBearer: cluster dims in order.
func (n *iterClusterNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.dims {
		fn(u)
	}
}

// NewIterCluster creates an abstract compound iteration header over dim
// InputParam nodes with body as its single operand.
func (g *Graph) NewIterCluster(origin string, dims []Node, body Node) Node {
	n := g.NewNode(KindIterCluster, origin, body).(*iterClusterNode)
	n.dims = make([]SSAUse, len(dims))
	for i, d := range dims {
		n.dims[i] = wirePayloadUse(n, d)
	}
	return n
}

// forLoopNode: lowered loop nest; ops = [iv, bound, body] — the bounded
// induction variable, its bound, and the body subtree (an LIR memory store, a
// nested forloop, or another effect-carrying stage).
type forLoopNode struct {
	*Base
	operandList
}

// NewForLoop creates an explicit lowered loop over the bounded induction
// variable iv, bounded by bound, with body as the loop body subtree. The
// verifier's shape check enforces the iv operand kind.
func (g *Graph) NewForLoop(origin string, iv, bound, body Node) Node {
	n := g.allocNode(KindForLoop, origin, 3).(*forLoopNode)
	wireOperands(n, iv, bound, body)
	return n
}
