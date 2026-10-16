package ir

// This file is the exported graph-construction surface: the ONLY way to
// create graphs, params, statements, and nodes from outside package ir
// (pkg/ builder, pass tests, visualization tests). Construction is exempt
// from MutContext (RESEARCH Pattern 2) but records PassOrigin so verifier
// diagnostics are meaningful from birth (D-17).
//
// Constructors deliberately do NOT validate kinds or operands: kind
// acceptance is the legality window's job (IR-03) and operand wiring is the
// verifier's job (PASS-06). This lets cross-package tests construct
// out-of-window or unwired graphs to prove rejection.

// Lambda is the payload of map/reduce stage nodes: lambda(l, r): <body>.
// It is a record, not a Kind.
type Lambda struct {
	Params []Param
	Body   Node
}

// Access is a structural tensor access: lhs[@m][@n].
type Access struct {
	Tensor Param
	Axes   []Param
}

// Concrete node types (D-15: Go interface families with a shared embedded
// *Base). Every operand edge is an SSAUse slot in the node's operandList; the
// kind lives in Base. Payload fields are set once by the typed constructors
// and are immutable after construction.

type constNode struct {
	*Base
	operandList
	typ Type
	val float64
}

func (*constNode) scalarExprNode() {}

type paramRefNode struct {
	*Base
	operandList
	param Param
}

func (*paramRefNode) scalarExprNode() {}

type binOpNode struct {
	*Base
	operandList
	op BinOpCode
}

func (*binOpNode) scalarExprNode() {}

type unaryOpNode struct {
	*Base
	operandList
	op UnaryOpCode
}

func (*unaryOpNode) scalarExprNode() {}

// selectNode: ops = [cond, a, b].
type selectNode struct {
	*Base
	operandList
}

func (*selectNode) scalarExprNode() {}

// castNode: ops = [x].
type castNode struct {
	*Base
	operandList
	typ Type
}

func (*castNode) scalarExprNode() {}

// callNode: ops = args.
type callNode struct {
	*Base
	operandList
	name string
}

func (*callNode) scalarExprNode() {}

// iterNode: abstract induction variable; ops = [init, step]; dim/bound are params.
type iterNode struct {
	*Base
	operandList
	dim   Param
	bound Param
}

// iterClusterNode: abstract compound header; ops = [body]; dims are params.
type iterClusterNode struct {
	*Base
	operandList
	dims []Param
}

// mapNode: abstract map stage; ops = operand values (piped input first);
// lam/as are payload records.
type mapNode struct {
	*Base
	operandList
	lam Lambda
	as  []Param
}

func (n *mapNode) forEachLambda(fn func(*Lambda)) { fn(&n.lam) }

// reduceNode: abstract reduce stage; ops = [init, piped]; combiner is payload.
type reduceNode struct {
	*Base
	operandList
	combiner Lambda
}

func (n *reduceNode) forEachLambda(fn func(*Lambda)) { fn(&n.combiner) }

// storeNode: abstract store; ops = [value]; target is payload.
type storeNode struct {
	*Base
	operandList
	target Access
}

// returnNode: abstract kernel return; ops = [value]. Return is the none
// Value (directive 2): it satisfies Value so it can sit in a statement RHS
// slot, and IsNone() reports that it binds no SSA name.
type returnNode struct {
	*Base
	operandList
}

// IsNone reports the none Value semantics of the return root.
func (*returnNode) IsNone() bool { return true }

// rawNode is the generic fallback for KindInvalid and any kind without a
// dedicated struct (including kinds >= KindEnd constructed by rejection
// tests). Its operand count is exactly the constructor's operand count.
type rawNode struct {
	*Base
	operandList
}

// lambdaBearer is implemented by nodes carrying Lambda payloads; the
// verifier walks lambda bodies as graph edges alongside operand uses.
type lambdaBearer interface {
	forEachLambda(fn func(*Lambda))
}

// fixedArity returns the operand-slot count for fixed-arity kinds, or -1
// for variadic kinds (call, map) and the raw fallback.
func fixedArity(k Kind) int {
	switch k {
	case KindConst, KindParam:
		return 0
	case KindUnaryOp, KindIterCluster, KindCast, KindStore, KindReturn:
		return 1
	case KindIter, KindBinOp, KindReduce:
		return 2
	case KindSelect:
		return 3
	default:
		return -1 // KindCall, KindMap, raw/unknown kinds: variadic
	}
}

// allocNode creates the concrete node for kind k with exactly nSlots
// zero-valued operand slots; the ID comes only from g.NewID() and the
// PassOrigin is stamped from origin. Every node gets its SSAVar at
// allocation (1:1, ID == node ID — IR-05's single NewID source, so statement
// numbering can never drift from the node IDs).
func (g *Graph) allocNode(k Kind, origin string, nSlots int) nodeImpl {
	b := &Base{id: g.NewID(), kind: k, loc: Loc{PassOrigin: origin}, graph: g}
	var n nodeImpl
	switch k {
	case KindConst:
		n = &constNode{Base: b}
	case KindParam:
		n = &paramRefNode{Base: b}
	case KindBinOp:
		n = &binOpNode{Base: b}
	case KindUnaryOp:
		n = &unaryOpNode{Base: b}
	case KindSelect:
		n = &selectNode{Base: b}
	case KindCast:
		n = &castNode{Base: b}
	case KindCall:
		n = &callNode{Base: b}
	case KindIter:
		n = &iterNode{Base: b}
	case KindIterCluster:
		n = &iterClusterNode{Base: b}
	case KindMap:
		n = &mapNode{Base: b}
	case KindReduce:
		n = &reduceNode{Base: b}
	case KindStore:
		n = &storeNode{Base: b}
	case KindReturn:
		n = &returnNode{Base: b}
	default:
		n = &rawNode{Base: b}
	}
	def, _ := n.(Value) // total for in-repo nodes: every concrete node embeds *Base
	b.ssa = &SSAVar{id: b.id, node: n, define: def}
	for i := 0; i < nSlots; i++ {
		n.addSlot()
	}
	return n
}

// slotCount returns the node's operand-slot count.
func slotCount(n nodeImpl) int {
	count := 0
	n.forEachOperand(func(SSAUse) { count++ })
	return count
}

// wireOperands wires each operand into its slot as an SSAUse handle over a
// fresh ref record in the def's SSAVar, exactly as the AddUse choke point
// does. A nil operand leaves the slot a zero handle (unwired) for the
// verifier to flag. Operands beyond a fixed-arity kind's slot count are NOT
// wired, but they are COUNTED and returned as the surplus so NewNode can
// record them: silent drops are fail-open, and construction in this design
// is fail-closed — the verifier flags any surplus under the use-def
// invariant (WR-01).
func wireOperands(n nodeImpl, operands ...Node) int {
	slots := slotCount(n)
	surplus := 0
	for i, val := range operands {
		if i >= slots {
			surplus++
			continue
		}
		if val == nil || val.BasePtr().ssa == nil {
			continue // leave the zero handle (foreign nodes without a var stay unwired)
		}
		rec := val.BasePtr().ssa.addRef(n)
		n.setSlot(i, SSAUse{rec: rec})
	}
	return surplus
}

// NewGraph creates an empty named graph holding exactly one basic block.
func NewGraph(name string) *Graph {
	return &Graph{name: name, block: &Block{}, nextID: 1}
}

// AddDimParam appends a dim parameter (a runtime shape symbol) to the graph.
func (g *Graph) AddDimParam(name string) Param {
	p := Param{name: name, role: roleDim}
	g.params = append(g.params, p)
	return p
}

// AddTensorParam appends a tensor parameter of the given interned type.
func (g *Graph) AddTensorParam(name string, t Type) Param {
	p := Param{name: name, typ: t, role: roleTensor}
	g.params = append(g.params, p)
	return p
}

// AddScalarParam appends a scalar parameter of the given interned type.
func (g *Graph) AddScalarParam(name string, t Type) Param {
	p := Param{name: name, typ: t, role: roleScalar}
	g.params = append(g.params, p)
	return p
}

// AddStatement appends a statement binding rhs to the graph's single block;
// the statement ID comes from g.NewID() (D-04). The rhs stays Node-typed —
// see the Statement contract in node.go for why (the Value classification
// lives in Statement.Value()).
func (g *Graph) AddStatement(rhs Node) *Statement {
	s := &Statement{id: g.NewID(), graph: g, rhs: rhs}
	g.block.statements = append(g.block.statements, s)
	return s
}

// ReserveStatementID returns the graph's next monotonic ID and advances the
// counter, letting a builder pin a statement's ID before the statement's RHS
// nodes exist. The doc-pinned oracle prints dense statement numbers
// (`statement #1`, `#2`), but a fluent builder constructs the statement's
// chain nodes before it appends the statement, which would otherwise push
// statement IDs behind every chain node. Reservation moves the allocation to
// chain-entry time; IDs still come only from the graph's monotonic counter
// (IR-05) and must be consumed by AddStatementAs in append order.
func (g *Graph) ReserveStatementID() uint64 {
	id := g.nextID
	g.nextID++
	return id
}

// AddStatementAs appends a statement carrying a builder-reserved ID (see
// ReserveStatementID). Unlike AddStatement the ID is not drawn at append
// time; the statement still joins the block in append order.
func (g *Graph) AddStatementAs(id uint64, rhs Node) *Statement {
	s := &Statement{id: id, graph: g, rhs: rhs}
	g.block.statements = append(g.block.statements, s)
	return s
}

// NewNode is the low-level constructor: it creates a node of ANY kind —
// including KindInvalid and kinds >= KindEnd — with operands wired as Uses.
// It deliberately performs no kind validation (see file comment), but
// operands supplied beyond a fixed-arity kind's slot count are recorded as
// surplus on the node so the verifier can reject the graph (fail-closed,
// WR-01); variadic kinds (call, map) and the raw fallback wire every operand.
func (g *Graph) NewNode(k Kind, origin string, operands ...Node) Node {
	arity := fixedArity(k)
	nSlots := arity
	if nSlots < 0 {
		nSlots = len(operands)
	}
	n := g.allocNode(k, origin, nSlots)
	if arity >= 0 {
		n.BasePtr().surplus = wireOperands(n, operands...)
	} else {
		wireOperands(n, operands...)
	}
	return n
}

// NewConst creates a typed scalar constant node.
func (g *Graph) NewConst(origin string, t Type, v float64) Node {
	n := g.NewNode(KindConst, origin).(*constNode)
	n.typ, n.val = t, v
	return n
}

// NewLambdaParam creates a lambda-parameter symbol: the named, typed
// placeholder for a map/reduce lambda argument (lambda(l, r): ...). Lambda
// params are NOT kernel I/O — they are not registered in the graph's param
// list and carry no kernel role.
func NewLambdaParam(name string, t Type) Param {
	return Param{name: name, typ: t}
}

// NewParamRef creates a scalar lambda-parameter reference node.
func (g *Graph) NewParamRef(origin string, p Param) Node {
	n := g.NewNode(KindParam, origin).(*paramRefNode)
	n.param = p
	return n
}

// NewBinOp creates a binary scalar expression node with operands a, b.
func (g *Graph) NewBinOp(origin string, op BinOpCode, a, b Node) Node {
	n := g.NewNode(KindBinOp, origin, a, b).(*binOpNode)
	n.op = op
	return n
}

// NewUnaryOp creates a unary scalar expression node with operand x.
func (g *Graph) NewUnaryOp(origin string, op UnaryOpCode, x Node) Node {
	n := g.NewNode(KindUnaryOp, origin, x).(*unaryOpNode)
	n.op = op
	return n
}

// NewSelect creates a typed conditional (cond ? a : b) node.
func (g *Graph) NewSelect(origin string, cond, a, b Node) Node {
	return g.NewNode(KindSelect, origin, cond, a, b)
}

// NewCast creates an explicit cast-to-t node with operand x.
func (g *Graph) NewCast(origin string, t Type, x Node) Node {
	n := g.NewNode(KindCast, origin, x).(*castNode)
	n.typ = t
	return n
}

// NewCall creates a registered pure builtin call node with the given args.
func (g *Graph) NewCall(origin string, name string, args ...Node) Node {
	n := g.NewNode(KindCall, origin, args...).(*callNode)
	n.name = name
	return n
}

// NewIterCluster creates an abstract compound iteration header over dims
// with body as its single operand.
func (g *Graph) NewIterCluster(origin string, dims []Param, body Node) Node {
	n := g.NewNode(KindIterCluster, origin, body).(*iterClusterNode)
	n.dims = append([]Param(nil), dims...)
	return n
}

// NewIter creates an abstract induction-variable node over dim, bounded by
// bound, with init and step operands.
func (g *Graph) NewIter(origin string, dim Param, bound Param, init, step Node) Node {
	n := g.NewNode(KindIter, origin, init, step).(*iterNode)
	n.dim, n.bound = dim, bound
	return n
}

// NewMap creates an abstract map stage over operand values ops, with lambda
// lam and explicit result-axis order as.
func (g *Graph) NewMap(origin string, ops []Node, lam Lambda, as []Param) Node {
	n := g.NewNode(KindMap, origin, ops...).(*mapNode)
	n.lam, n.as = lam, append([]Param(nil), as...)
	return n
}

// NewReduce creates an abstract reduce stage with init and piped operands
// and combiner lambda.
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node {
	n := g.NewNode(KindReduce, origin, init, piped).(*reduceNode)
	n.combiner = combiner
	return n
}

// NewStore creates an abstract structural store node writing value through
// the target access.
func (g *Graph) NewStore(origin string, target Access, value Node) Node {
	n := g.NewNode(KindStore, origin, value).(*storeNode)
	n.target = target
	return n
}

// NewReturn creates an abstract kernel-return node with the returned value.
func (g *Graph) NewReturn(origin string, value Node) Node {
	return g.NewNode(KindReturn, origin, value)
}
