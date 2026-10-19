package ir

// node_expr.go holds the HIR scalar-expression node suite (D-3: Const, Param,
// BinOp, UnaryOp, Select, Cast, Call — FamilyScalarExpr) and their operator
// opcode tables. These are the value-level leaves every stage's lambda body
// and operand list is built from; lowering never rewrites them (they survive
// to codegen as scalar expressions).

// BinOpCode is the binary operator table, mirroring doc/syntax.md §7.6
// precedence levels from loosest to tightest binding.
type BinOpCode int

const (
	BinOpInvalid BinOpCode = iota
	BinOpLOr               // ||
	BinOpLAnd              // &&
	BinOpOr                // |
	BinOpXor               // ^
	BinOpAnd               // &
	BinOpEq                // ==
	BinOpNE                // !=
	BinOpLT                // <
	BinOpLE                // <=
	BinOpGT                // >
	BinOpGE                // >=
	BinOpShL               // <<
	BinOpShR               // >>
	BinOpAdd               // +
	BinOpSub               // -
	BinOpMul               // *
	BinOpDiv               // /
	BinOpMod               // %
)

// String returns the canonical source spelling of the operator.
func (op BinOpCode) String() string {
	switch op {
	case BinOpLOr:
		return "||"
	case BinOpLAnd:
		return "&&"
	case BinOpOr:
		return "|"
	case BinOpXor:
		return "^"
	case BinOpAnd:
		return "&"
	case BinOpEq:
		return "=="
	case BinOpNE:
		return "!="
	case BinOpLT:
		return "<"
	case BinOpLE:
		return "<="
	case BinOpGT:
		return ">"
	case BinOpGE:
		return ">="
	case BinOpShL:
		return "<<"
	case BinOpShR:
		return ">>"
	case BinOpAdd:
		return "+"
	case BinOpSub:
		return "-"
	case BinOpMul:
		return "*"
	case BinOpDiv:
		return "/"
	case BinOpMod:
		return "%"
	default:
		return "?"
	}
}

// UnaryOpCode is the unary operator table, mirroring doc/syntax.md §7.6.
type UnaryOpCode int

const (
	UnaryOpInvalid UnaryOpCode = iota
	UnaryOpPos                 // +
	UnaryOpNeg                 // -
	UnaryOpLNot                // !
	UnaryOpNot                 // ~
)

// String returns the canonical source spelling of the operator.
func (op UnaryOpCode) String() string {
	switch op {
	case UnaryOpPos:
		return "+"
	case UnaryOpNeg:
		return "-"
	case UnaryOpLNot:
		return "!"
	case UnaryOpNot:
		return "~"
	default:
		return "?"
	}
}

// constNode: typed scalar constant; ops = [].
type constNode struct {
	*Base
	operandList
	typ Type
	val float64
}

func (*constNode) scalarExprNode() {}

// NewConst creates a typed scalar constant node.
func (g *Graph) NewConst(origin string, t Type, v float64) Node {
	n := g.NewNode(KindConst, origin).(*constNode)
	n.typ, n.val = t, v
	return n
}

// paramRefNode: scalar-expr reference to a (lambda or scalar) InputParam.
// The reference IS operand 0 (Input/ImpactBy-only convention,
// quick-261019-ktm revision: the parameter target is a data dependency, so
// it rides the Input list — no payload field).
type paramRefNode struct {
	*Base
	operandList
}

func (*paramRefNode) scalarExprNode() {}

// NewParamRef creates a scalar parameter-reference node: operand 0 points
// at an InputParam node (a lambda parameter or a captured kernel scalar).
// The verifier's shape check pins the operand kind.
func (g *Graph) NewParamRef(origin string, p Node) Node {
	return g.NewNode(KindParam, origin, p)
}

// binOpNode: binary scalar expression; ops = [a, b].
type binOpNode struct {
	*Base
	operandList
	op BinOpCode
}

func (*binOpNode) scalarExprNode() {}

// NewBinOp creates a binary scalar expression node with operands a, b.
func (g *Graph) NewBinOp(origin string, op BinOpCode, a, b Node) Node {
	n := g.NewNode(KindBinOp, origin, a, b).(*binOpNode)
	n.op = op
	return n
}

// unaryOpNode: unary scalar expression; ops = [x].
type unaryOpNode struct {
	*Base
	operandList
	op UnaryOpCode
}

func (*unaryOpNode) scalarExprNode() {}

// NewUnaryOp creates a unary scalar expression node with operand x.
func (g *Graph) NewUnaryOp(origin string, op UnaryOpCode, x Node) Node {
	n := g.NewNode(KindUnaryOp, origin, x).(*unaryOpNode)
	n.op = op
	return n
}

// selectNode: typed conditional; ops = [cond, a, b].
type selectNode struct {
	*Base
	operandList
}

func (*selectNode) scalarExprNode() {}

// NewSelect creates a typed conditional (cond ? a : b) node.
func (g *Graph) NewSelect(origin string, cond, a, b Node) Node {
	return g.NewNode(KindSelect, origin, cond, a, b)
}

// castNode: explicit cast; ops = [x].
type castNode struct {
	*Base
	operandList
	typ Type
}

func (*castNode) scalarExprNode() {}

// NewCast creates an explicit cast-to-t node with operand x.
func (g *Graph) NewCast(origin string, t Type, x Node) Node {
	n := g.NewNode(KindCast, origin, x).(*castNode)
	n.typ = t
	return n
}

// callNode: registered pure builtin call; ops = args.
type callNode struct {
	*Base
	operandList
	name string
}

func (*callNode) scalarExprNode() {}

// NewCall creates a registered pure builtin call node with the given args.
func (g *Graph) NewCall(origin string, name string, args ...Node) Node {
	n := g.NewNode(KindCall, origin, args...).(*callNode)
	n.name = name
	return n
}
