package ir

// node_lambda.go holds the HIR lambda-stage nodes (KindMap/KindReduce, D-3)
// and the Lambda payload record. "Pipe" survives only as BUILDER terminology
// (the fluent pkg chain): the IR itself models a stage's piped input as an
// ordinary operand — the map's operand 0 and the reduce's operand 1 — so the
// sea-of-nodes' data-dependency edges cover piping with zero special cases
// (tinycudaopto lambda.go: pipes were removed as IR nodes; the edges stay).

// Lambda is the payload of map/reduce stage nodes: lambda(l, r): <body>.
// It is a record, not a Kind. Its params are SSAUses of lambda InputParam
// nodes — the review's IR model: lambda parameters are InputParams, inlined
// at lowering.
type Lambda struct {
	params []SSAUse
	Body   Node
}

// Params returns the lambda's parameter uses in declaration order.
func (l Lambda) Params() []SSAUse { return append([]SSAUse(nil), l.params...) }

// NewLambda builds a lambda record over the given lambda-InputParam nodes,
// wiring one payload use per parameter. The record's use back-pointers are
// fixed up to the owning stage node when the lambda is attached by
// NewMap/NewReduce (the node does not exist yet at Lambda construction).
func NewLambda(params []Node, body Node) Lambda {
	lam := Lambda{Body: body}
	for _, p := range params {
		lam.params = append(lam.params, wirePayloadUse(nil, p))
	}
	return lam
}

// mapNode: abstract map stage; ops = operand values (piped input first);
// lam/as are payload records.
type mapNode struct {
	*Base
	operandList
	lam Lambda
	as  []SSAUse
}

// forEachLambda implements lambdaBearer.
func (n *mapNode) forEachLambda(fn func(*Lambda)) { fn(&n.lam) }

// forEachPayloadUse implements payloadBearer: lambda params, then the
// as-order uses.
func (n *mapNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.lam.params {
		fn(u)
	}
	for _, u := range n.as {
		fn(u)
	}
}

// NewMap creates an abstract map stage over operand values ops, with lambda
// lam and explicit result-axis order as (InputParam nodes for the axes).
func (g *Graph) NewMap(origin string, ops []Node, lam Lambda, as []Node) Node {
	n := g.NewNode(KindMap, origin, ops...).(*mapNode)
	fixupLambdaUsers(n, &lam)
	n.lam = lam
	n.as = make([]SSAUse, len(as))
	for i, d := range as {
		n.as[i] = wirePayloadUse(n, d)
	}
	return n
}

// reduceNode: abstract reduce stage; ops = [init, piped]; combiner is payload.
type reduceNode struct {
	*Base
	operandList
	combiner Lambda
}

// forEachLambda implements lambdaBearer.
func (n *reduceNode) forEachLambda(fn func(*Lambda)) { fn(&n.combiner) }

// forEachPayloadUse implements payloadBearer: combiner params.
func (n *reduceNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.combiner.params {
		fn(u)
	}
}

// NewReduce creates an abstract reduce stage with init and piped operands
// and combiner lambda.
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node {
	n := g.NewNode(KindReduce, origin, init, piped).(*reduceNode)
	fixupLambdaUsers(n, &combiner)
	n.combiner = combiner
	return n
}
