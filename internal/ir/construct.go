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
//
// Every reference in the graph is an SSAUse (review P1R-01): operand slots
// hold SSAUse handles, and payload fields (lambda params, store target
// access, iteration dims/bounds, map as-order, parameter refs) hold SSAUse
// records wired exactly like operand uses — a fresh ref record in the def's
// SSAVar, owned by the referencing node. Names appear ONLY as InputParam
// debug payload and are never used for resolution or identity.

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

// access is a structural tensor access: lhs[@m][@n]. Every component is an
// SSAUse — the tensor InputParam and one dim InputParam per axis — never a
// name (review P1R-01).
type access struct {
	tensorUse SSAUse
	axes      []SSAUse
}

// Tensor returns the access's tensor parameter node, or nil when unwired.
func (a access) Tensor() Node { return a.tensorUse.Def() }

// Axes returns the access's axis parameter nodes in subscript order; unwired
// slots yield nil entries.
func (a access) Axes() []Node {
	out := make([]Node, len(a.axes))
	for i, u := range a.axes {
		out[i] = u.Def()
	}
	return out
}

// Concrete node types (D-15: Go interface families with a shared embedded
// *Base). Every operand edge is an SSAUse slot in the node's operandList;
// the kind lives in Base. Payload fields are set once by the typed
// constructors and are immutable after construction.

type constNode struct {
	*Base
	operandList
	typ Type
	val float64
}

func (*constNode) scalarExprNode() {}

// inputParamNode is a kernel or lambda input parameter (review P1R-01): one
// graph-owned node per parameter, referenced exclusively through SSAUse
// records. name/typ/role are debug/diagnostic payload — NEVER resolution or
// identity. ops is empty (arity 0); the SSAVar IS the parameter's def.
type inputParamNode struct {
	*Base
	operandList
	name string
	typ  Type // nil for dim params (dims are not typed values in v0)
	role ParamRole
}

// paramRefNode: scalar-expr reference to a (lambda or scalar) InputParam;
// the reference IS the payload use.
type paramRefNode struct {
	*Base
	operandList
	use SSAUse
}

func (*paramRefNode) scalarExprNode() {}

// forEachPayloadUse implements payloadBearer: the parameter reference.
func (n *paramRefNode) forEachPayloadUse(fn func(SSAUse)) { fn(n.use) }

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

// storeNode: abstract store; ops = [value]; target is a payload access of
// InputParam uses.
type storeNode struct {
	*Base
	operandList
	target access
}

// forEachPayloadUse implements payloadBearer: target tensor, then axes.
func (n *storeNode) forEachPayloadUse(fn func(SSAUse)) {
	fn(n.target.tensorUse)
	for _, u := range n.target.axes {
		fn(u)
	}
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

// payloadBearer is implemented by nodes carrying non-operand SSAUse payloads
// (parameter refs, iteration dims/bounds, lambda params, as-orders, store
// targets). The verifier treats payload uses exactly like operand slots in
// both chain directions; Walk follows them as graph edges.
type payloadBearer interface {
	forEachPayloadUse(fn func(SSAUse))
}

// fixedArity returns the operand-slot count for fixed-arity kinds, or -1
// for variadic kinds (call, map) and the raw fallback.
func fixedArity(k Kind) int {
	switch k {
	case KindConst, KindParam, KindInputParam:
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
	case KindInputParam:
		n = &inputParamNode{Base: b}
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

// wirePayloadUse links val as a payload (non-operand-slot) reference of
// user, creating a fresh ref record in the def's SSAVar exactly as operand
// wiring does. A nil val (or a foreign node without a var) leaves a zero
// handle for the verifier to flag. The user back-pointer may be nil when the
// owning node does not exist yet (Lambda records); NewMap/NewReduce fix the
// records up to the stage node when attaching the lambda.
func wirePayloadUse(user Node, val Node) SSAUse {
	if val == nil || val.BasePtr().ssa == nil {
		return SSAUse{}
	}
	return SSAUse{rec: val.BasePtr().ssa.addRef(user)}
}

// fixupLambdaUsers retargets a freshly-built lambda record's parameter use
// records at their owning stage node (they were created with a nil user
// before the node existed).
func fixupLambdaUsers(n Node, lam *Lambda) {
	for i := range lam.params {
		if r := lam.params[i].rec; r != nil {
			r.user = n
		}
	}
}

// NewGraph creates an empty named graph holding exactly one basic block.
func NewGraph(name string) *Graph {
	g := &Graph{name: name, nextID: 1}
	g.cfg = &CFG{entry: &BasicBlock{graph: g}}
	return g
}

// NewInputParam creates one graph-owned kernel input parameter (dim, tensor,
// or scalar) and appends it to the graph's param list. The returned node IS
// the parameter's SSADef pointee: every reference to it elsewhere in the
// graph is an SSAUse record; the name is debug payload only (review P1R-01).
func (g *Graph) NewInputParam(origin string, role ParamRole, name string, typ Type) Node {
	n := g.allocNode(KindInputParam, origin, 0).(*inputParamNode)
	n.name, n.typ, n.role = name, typ, role
	g.params = append(g.params, n)
	return n
}

// NewLambdaInputParam creates a lambda-bound InputParam node (role
// RoleLambda). Like kernel parameters it is an SSA def referenced only
// through SSAUse records, but it is NOT registered in the graph's param
// list — lambda params are graph-external by design (review P1R-01; they are
// inlined at lowering).
func (g *Graph) NewLambdaInputParam(origin, name string, typ Type) Node {
	n := g.allocNode(KindInputParam, origin, 0).(*inputParamNode)
	n.name, n.typ, n.role = name, typ, RoleLambda
	return n
}

// AddStatement appends a statement binding rhs to the graph's single block;
// the statement ID comes from g.NewID() (D-04). The rhs stays Node-typed —
// see the Statement contract in cfg.go for why (the Value classification
// lives in Statement.Value()).
func (g *Graph) AddStatement(rhs Node) *Statement {
	s := &Statement{id: g.NewID(), kind: classifyStmt(rhs), rhs: rhs}
	if rhs != nil {
		rhs.BasePtr().stmt = s
		if s.kind == StmtAssign {
			s.def = SSADef{ssa: rhs.BasePtr().ssa}
		}
	}
	g.cfg.entry.Append(s)
	return s
}

// ReserveStatementID returns the graph's next monotonic ID and advances the
// counter, letting a builder pin a statement's ID before the statement's RHS
// nodes exist. A fluent builder constructs the statement's chain nodes
// before it appends the statement, which would otherwise push statement IDs
// behind every chain node. Reservation moves the allocation to chain-entry
// time; IDs still come only from the graph's monotonic counter (IR-05) and
// must be consumed by AddStatementAs in append order.
func (g *Graph) ReserveStatementID() uint64 {
	id := g.nextID
	g.nextID++
	return id
}

// AddStatementAs appends a statement carrying a builder-reserved ID (see
// ReserveStatementID). Unlike AddStatement the ID is not drawn at append
// time; the statement still joins the block in append order.
func (g *Graph) AddStatementAs(id uint64, rhs Node) *Statement {
	s := &Statement{id: id, kind: classifyStmt(rhs), rhs: rhs}
	if rhs != nil {
		rhs.BasePtr().stmt = s
		if s.kind == StmtAssign {
			s.def = SSADef{ssa: rhs.BasePtr().ssa}
		}
	}
	g.cfg.entry.Append(s)
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

// NewParamRef creates a scalar parameter-reference node: a payload SSAUse
// pointing at an InputParam node (a lambda parameter or a captured kernel
// scalar).
func (g *Graph) NewParamRef(origin string, p Node) Node {
	n := g.NewNode(KindParam, origin).(*paramRefNode)
	n.use = wirePayloadUse(n, p)
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

// NewIter creates an abstract induction-variable node over the dim
// InputParam node dim, bounded by the InputParam node bound (nil = the
// unbounded form @x), with init and step operands.
func (g *Graph) NewIter(origin string, dim, bound Node, init, step Node) Node {
	n := g.NewNode(KindIter, origin, init, step).(*iterNode)
	n.dimUse = wirePayloadUse(n, dim)
	n.boundUse = wirePayloadUse(n, bound)
	return n
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

// NewReduce creates an abstract reduce stage with init and piped operands
// and combiner lambda.
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node {
	n := g.NewNode(KindReduce, origin, init, piped).(*reduceNode)
	fixupLambdaUsers(n, &combiner)
	n.combiner = combiner
	return n
}

// NewStore creates an abstract structural store node writing value through
// the target access: tensor is the tensor InputParam node and axes its dim
// InputParam nodes (nil/empty for a whole-tensor store).
func (g *Graph) NewStore(origin string, tensor Node, axes []Node, value Node) Node {
	n := g.NewNode(KindStore, origin, value).(*storeNode)
	n.target.tensorUse = wirePayloadUse(n, tensor)
	if len(axes) > 0 {
		n.target.axes = make([]SSAUse, len(axes))
		for i, ax := range axes {
			n.target.axes[i] = wirePayloadUse(n, ax)
		}
	}
	return n
}

// NewReturn creates an abstract kernel-return node with the returned value.
func (g *Graph) NewReturn(origin string, value Node) Node {
	return g.NewNode(KindReturn, origin, value)
}
