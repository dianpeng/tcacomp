package ir

// node_lambda.go holds the HIR lambda-stage nodes (KindMap/KindReduce, D-3)
// and the Lambda payload record. "Pipe" survives only as BUILDER terminology
// (the fluent pkg chain): the IR itself models a stage's piped input as an
// ordinary operand — the map's operand 0 and the reduce's operand 0 — so the
// sea-of-nodes' data-dependency edges cover piping with zero special cases
// (tinycudaopto lambda.go: pipes were removed as IR nodes; the edges stay).
//
// Input-list restructure (quick-261019-ktm, D1/D2/D9): EVERY incoming
// reference of a Map/Reduce rides its operand list. The map's as-order is a
// trailing KindAs node appended by NewMap; the reduce's initializer rides
// operand 1 as a KindKeyedAttribute ("init") wrapping the value; a lambda
// parameter binds positionally to the i-th Input-list VALUE binding, where
// an attribute element contributes its wrapped value operand and a trailing
// As binds nothing. The accessors AsOf/ReduceInitOf read the layout back
// without hidden pointer state.

// Lambda is the payload of map/reduce stage nodes: lambda(l, r): <body>.
// It is a record, not a Kind. Its params are the parameter DECLARATIONS
// (ordered, Walk-anchored SSAUses of lambda InputParam nodes — D8: the
// declarations stay; the BINDING is the Input-list correspondence, not a
// hidden pointer). Lambda bodies are inlined at lowering
// (internal/opt/simplify).
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

// mapNode: abstract map stage; ops = [value operands..., As?] — the
// as-order rides the operand list as a trailing KindAs node (D1); lam is
// the payload record (parameter declarations + body).
type mapNode struct {
	*Base
	operandList
	lam Lambda
}

// forEachLambda implements lambdaBearer.
func (n *mapNode) forEachLambda(fn func(*Lambda)) { fn(&n.lam) }

// forEachPayloadUse implements payloadBearer: the lambda parameter
// declarations (the as-order left the payload with D1 — it lives on the
// Input list now).
func (n *mapNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.lam.params {
		fn(u)
	}
}

// NewMap creates an abstract map stage over operand values ops, with lambda
// lam and explicit result-axis order as (InputParam nodes for the axes).
// When as is non-empty a trailing As node is allocated (before the map, so
// allocation order is deterministic: wrapper first, then the stage) and
// appended to the wired operand list. The signature is unchanged — the
// legacy builder path compiles as-is.
func (g *Graph) NewMap(origin string, ops []Node, lam Lambda, as []Node) Node {
	wired := ops
	if len(as) > 0 {
		asNode := g.NewAs(origin, as...)
		wired = make([]Node, 0, len(ops)+1)
		wired = append(wired, ops...)
		wired = append(wired, asNode)
	}
	n := g.NewNode(KindMap, origin, wired...).(*mapNode)
	fixupLambdaUsers(n, &lam)
	n.lam = lam
	return n
}

// reduceNode: abstract reduce stage; ops = [piped, KeyedAttribute("init",
// init), As?] — the initializer rides operand 1 as a keyed attribute (D2),
// an optional trailing As may follow (D1); combiner is the payload record.
type reduceNode struct {
	*Base
	operandList
	combiner Lambda
}

// forEachLambda implements lambdaBearer.
func (n *reduceNode) forEachLambda(fn func(*Lambda)) { fn(&n.combiner) }

// forEachPayloadUse implements payloadBearer: combiner parameter
// declarations.
func (n *reduceNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.combiner.params {
		fn(u)
	}
}

// NewReduce creates an abstract reduce stage with init and piped operands
// and combiner lambda. The signature is unchanged (zero call-site churn);
// internally the operand list is [piped, KeyedAttribute("init", init)] with
// the attribute allocated BEFORE the reduce so allocation order is
// deterministic (wrapper first, then the stage — the NewMap rule).
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node {
	attr := g.NewKeyedAttribute(origin, "init", init)
	n := g.NewNode(KindReduce, origin, piped, attr).(*reduceNode)
	fixupLambdaUsers(n, &combiner)
	n.combiner = combiner
	return n
}

// AsOf returns the trailing as-order node of a Map (or Reduce, when one
// arrives): the LAST KindAs operand of n's Input list, or nil when n
// carries none (D1: the layout contract is optional-but-unique-and-last;
// the verifier enforces the placement, this accessor reads it back without
// hidden pointer state).
func AsOf(n Node) Node {
	if n == nil {
		return nil
	}
	ops := OperandsOf(n)
	for i := len(ops) - 1; i >= 0; i-- {
		if ops[i] != nil && ops[i].KindOf() == KindAs {
			return ops[i]
		}
	}
	return nil
}

// ReduceInitOf unwraps a reduce's initializer: operand 1 must be a
// KeyedAttribute (D2 — the verifier enforces the shape fail-closed); the
// attribute's value operand is the init. Any other node kind, a missing
// operand, or a malformed attribute yields nil.
func ReduceInitOf(n Node) Node {
	red, ok := n.(*reduceNode)
	if !ok {
		return nil
	}
	attr, ok := operandAt(red, 1).(*keyedAttributeNode)
	if !ok {
		return nil
	}
	return operandAt(attr, 0)
}
