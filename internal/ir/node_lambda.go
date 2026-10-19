package ir

// node_lambda.go holds the HIR lambda-stage nodes (KindMap/KindReduce, D-3)
// and the lambda expression node (KindLambda, quick-261019-ktm revision
// D15). "Pipe" survives only as BUILDER terminology (the fluent pkg chain):
// the IR itself models a stage's piped input as an ordinary operand — the
// map's and the reduce's operand 0 — so the sea-of-nodes' data-dependency
// edges cover piping with zero special cases (tinycudaopto lambda.go: pipes
// were removed as IR nodes; the edges stay).
//
// Input/ImpactBy-only convention (quick-261019-ktm revision): Map and
// Reduce store NO lambda record — they carry a KindLambda NODE as their
// TRAILING operand. The lambda node's own Input list is
// [param declarations..., body]: the parameter atoms (RoleLambda
// InputParams) in binding order, then the body expression. Every reference
// — as-order (trailing As), initializer (operand 1 KeyedAttribute), lambda
// parameters, and the body — rides an ordinary operand slot, so Walk,
// Verify, DCE, GVN, and the printers see stage inputs as plain SSA edges.
// Simplification (internal/opt/simplify) still inlines bodies and strips
// parameters; until then the body is simply graph-visible data flow.
//
// Lambda binding rule (D9, unchanged): lambda param i binds to the i-th
// Input-list VALUE binding of the Map/Reduce, where a KeyedAttribute
// element contributes its wrapped value operand and trailing As/lambda
// nodes bind nothing.

// Lambda is the TRANSIENT construction record the builder API hands to
// NewMap/NewReduce (D15): parameter atoms in binding order plus the body.
// It holds PLAIN NODES, wires nothing, and no IR node ever stores one —
// NewMap/NewReduce materialize a KindLambda node from it. (The pre-revision
// record stored SSAUse params and a raw Body pointer inside the stage —
// exactly the hidden state the Input/ImpactBy-only convention bans.)
type Lambda struct {
	params []Node
	Body   Node
}

// Params returns the lambda's parameter atoms in declaration order.
func (l Lambda) Params() []Node { return append([]Node(nil), l.params...) }

// NewLambda builds the transient lambda construction record over the given
// lambda-InputParam nodes and body. It wires nothing.
func NewLambda(params []Node, body Node) Lambda {
	return Lambda{params: append([]Node(nil), params...), Body: body}
}

// lambdaNode: the lambda expression node; ops = [params..., body]. The
// parameter atoms lead in binding order, the body expression closes the
// list. Carried by Map/Reduce as their trailing operand; pure
// (FamilyAbstract) and GVN-eligible (two lambdas over the same atoms and
// body node are one value).
type lambdaNode struct {
	*Base
	operandList
}

// NewLambdaNode creates a lambda expression node over the parameter atoms
// (in binding order) and the body expression. The verifier permits only
// NewMap/NewReduce to carry it; standalone lambdas are construction debris.
func (g *Graph) NewLambdaNode(origin string, params []Node, body Node) Node {
	ops := make([]Node, 0, len(params)+1)
	ops = append(ops, params...)
	ops = append(ops, body)
	return g.NewNode(KindLambda, origin, ops...)
}

// mapNode: abstract map stage; ops = [value operands..., As?, lambda] — the
// as-order rides the operand list as a trailing As node (D1) and the lambda
// node closes the list (D15); NO payload records.
type mapNode struct {
	*Base
	operandList
}

// NewMap creates an abstract map stage over operand values ops, with lambda
// lam and explicit result-axis order as (InputParam nodes for the axes).
// When as is non-empty a trailing As node is allocated, then the lambda
// node (wrapper nodes before the stage, so allocation order is
// deterministic), and both are appended to the wired operand list. The
// signature is unchanged — the legacy builder path compiles as-is.
func (g *Graph) NewMap(origin string, ops []Node, lam Lambda, as []Node) Node {
	wired := make([]Node, 0, len(ops)+2)
	wired = append(wired, ops...)
	if len(as) > 0 {
		wired = append(wired, g.NewAs(origin, as...))
	}
	wired = append(wired, g.NewLambdaNode(origin, lam.params, lam.Body))
	return g.NewNode(KindMap, origin, wired...)
}

// reduceNode: abstract reduce stage; ops = [piped, KeyedAttribute("init",
// init), As?, lambda] — the initializer rides operand 1 as a keyed
// attribute (D2), the lambda node closes the list (D15); NO payload
// records.
type reduceNode struct {
	*Base
	operandList
}

// NewReduce creates an abstract reduce stage with init and piped operands
// and combiner lambda. The signature is unchanged (zero call-site churn);
// internally the operand list is [piped, KeyedAttribute("init", init),
// lambda] with the wrappers allocated BEFORE the reduce so allocation order
// is deterministic.
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node {
	attr := g.NewKeyedAttribute(origin, "init", init)
	lam := g.NewLambdaNode(origin, combiner.params, combiner.Body)
	return g.NewNode(KindReduce, origin, piped, attr, lam)
}

// lambdaNodeOf returns n's lambda node — the TRAILING operand when it is a
// KindLambda — or nil. Map and Reduce always carry one (until a future pass
// strips it); the read is derived, never stored.
func lambdaNodeOf(n Node) Node {
	ops := OperandsOf(n)
	if len(ops) == 0 {
		return nil
	}
	last := ops[len(ops)-1]
	if last != nil && last.KindOf() == KindLambda {
		return last
	}
	return nil
}

// LambdaOf returns the lambda node carried by a Map or Reduce (its trailing
// operand), or nil for any other shape (D15 read face).
func LambdaOf(n Node) Node {
	if n == nil {
		return nil
	}
	switch n.KindOf() {
	case KindMap, KindReduce:
		return lambdaNodeOf(n)
	}
	return nil
}

// lambdaNodeParams returns the lambda node's leading parameter operands
// (all but the trailing body).
func lambdaNodeParams(lam Node) []Node {
	ops := OperandsOf(lam)
	if len(ops) <= 1 {
		return nil
	}
	return ops[:len(ops)-1]
}

// lambdaNodeBody returns the lambda node's trailing body operand, or nil.
func lambdaNodeBody(lam Node) Node {
	ops := OperandsOf(lam)
	if len(ops) == 0 {
		return nil
	}
	return ops[len(ops)-1]
}

// AsOf returns the trailing as-order node of a Map (or Reduce, when one
// arrives): the last KindAs operand of n's Input list BEFORE the trailing
// lambda operand, or nil when n carries none (D1: the layout contract is
// optional-but-unique-and-last among the value inputs; the verifier
// enforces the placement, this accessor reads it back without hidden
// pointer state).
func AsOf(n Node) Node {
	if n == nil {
		return nil
	}
	ops := OperandsOf(n)
	if lam := lambdaNodeOf(n); lam != nil && len(ops) > 0 {
		ops = ops[:len(ops)-1] // the As rides before the lambda
	}
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
