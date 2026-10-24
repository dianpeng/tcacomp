package ast

import "tcacomp/internal/syntax/lexer"

// --- expressions (design §3.1; Expr is an interface) ---

// Expr is one scalar expression node. The set is closed: literals, name
// references, binary/unary operators, the conditional, builtin calls, and
// explicit casts. There is no ParenExpr — precedence is resolved during
// parsing and parens never survive into the AST (design §3.2).
type Expr interface {
	Node
	exprNode()
}

// LitKind is the literal family of a BasicLit (design §3.1: Int|Uint|Float|Bool).
type LitKind int

const (
	LitInt LitKind = iota
	LitUint
	LitFloat
	LitBool
)

// BasicLit is one scalar literal: intLit, uintLit, floatLit, true/false, or
// the predefined inf_f32 constant (rendered as LitFloat with Value "inf_f32",
// design §2.1). Value is the exact source spelling; Val the parsed value
// (int32 / uint32 / float32 / bool).
type BasicLit struct {
	pos   lexer.Pos
	Kind  LitKind
	Value string
	Val   any
}

func (n *BasicLit) astNode()       {}
func (n *BasicLit) exprNode()      {}
func (n *BasicLit) Pos() lexer.Pos { return n.pos }

// NewBasicLit builds a BasicLit at pos.
func NewBasicLit(pos lexer.Pos, kind LitKind, value string, val any) *BasicLit {
	return &BasicLit{pos: pos, Kind: kind, Value: value, Val: val}
}

// NameRef is a bare identifier inside an expression. Resolution order is
// fixed by semacheck (design §3.1): lambda param > capture > const scalar let.
type NameRef struct {
	pos  lexer.Pos
	Name string
}

func (n *NameRef) astNode()       {}
func (n *NameRef) exprNode()      {}
func (n *NameRef) Pos() lexer.Pos { return n.pos }

// NewNameRef builds a NameRef at pos.
func NewNameRef(pos lexer.Pos, name string) *NameRef {
	return &NameRef{pos: pos, Name: name}
}

// PipeRef is a use-site reference to the innermost enclosing stage's pipe
// slot (param 0 of a piped map, or the element param of a reduce). Position,
// not spelling, identifies the slot — the def-site is the stage's ParamSpec,
// whether spelled %0 or named (design §7.3). Semacheck rejects a PipeRef in
// a stage that has no pipe slot (root-map body, reduce init).
type PipeRef struct {
	pos lexer.Pos
}

func (n *PipeRef) astNode()       {}
func (n *PipeRef) exprNode()      {}
func (n *PipeRef) Pos() lexer.Pos { return n.pos }

// NewPipeRef builds a PipeRef at pos.
func NewPipeRef(pos lexer.Pos) *PipeRef { return &PipeRef{pos: pos} }

// BinOp is a binary operator spelling, 1:1 with the IR BinOpCode table
// (design §2.7/§3.1): "||" "&&" "|" "^" "&" "==" "!=" "<" "<=" ">" ">="
// "<<" ">>" "+" "-" "*" "/" "%".
type BinOp string

// UnaryOp is a unary operator spelling, 1:1 with the IR UnaryOpCode table
// (design §2.7/§3.1): "+" "-" "!" "~".
type UnaryOp string

// BinaryExpr is one binary operation L op R.
type BinaryExpr struct {
	pos  lexer.Pos
	Op   BinOp
	L, R Expr
}

func (n *BinaryExpr) astNode()       {}
func (n *BinaryExpr) exprNode()      {}
func (n *BinaryExpr) Pos() lexer.Pos { return n.pos }

// NewBinaryExpr builds a BinaryExpr at pos.
func NewBinaryExpr(pos lexer.Pos, op BinOp, l, r Expr) *BinaryExpr {
	return &BinaryExpr{pos: pos, Op: op, L: l, R: r}
}

// UnaryExpr is one unary operation op X.
type UnaryExpr struct {
	pos lexer.Pos
	Op  UnaryOp
	X   Expr
}

func (n *UnaryExpr) astNode()       {}
func (n *UnaryExpr) exprNode()      {}
func (n *UnaryExpr) Pos() lexer.Pos { return n.pos }

// NewUnaryExpr builds a UnaryExpr at pos.
func NewUnaryExpr(pos lexer.Pos, op UnaryOp, x Expr) *UnaryExpr {
	return &UnaryExpr{pos: pos, Op: op, X: x}
}

// SelectExpr is the lazy conditional cond ? then : else (design §4.6 rule 18:
// C++ conditional evaluation; the frontend must not constant-fold a branch
// away).
type SelectExpr struct {
	pos              lexer.Pos
	Cond, Then, Else Expr
}

func (n *SelectExpr) astNode()       {}
func (n *SelectExpr) exprNode()      {}
func (n *SelectExpr) Pos() lexer.Pos { return n.pos }

// NewSelectExpr builds a SelectExpr at pos.
func NewSelectExpr(pos lexer.Pos, cond, then, els Expr) *SelectExpr {
	return &SelectExpr{pos: pos, Cond: cond, Then: then, Else: els}
}

// CallExpr is one builtin call identifier '(' args ')' — registry-resolved by
// semacheck against ir.Builtin (design §4.6 rule 20).
type CallExpr struct {
	pos  lexer.Pos
	Name string
	Args []Expr
}

func (n *CallExpr) astNode()       {}
func (n *CallExpr) exprNode()      {}
func (n *CallExpr) Pos() lexer.Pos { return n.pos }

// NewCallExpr builds a CallExpr at pos.
func NewCallExpr(pos lexer.Pos, name string, args []Expr) *CallExpr {
	return &CallExpr{pos: pos, Name: name, Args: args}
}

// CastExpr is the explicit conversion scalarType '(' expr ')' (design §4.6
// rule 19; admitted pairs come from the shared cast registry).
type CastExpr struct {
	pos    lexer.Pos
	Target Elem
	X      Expr
}

func (n *CastExpr) astNode()       {}
func (n *CastExpr) exprNode()      {}
func (n *CastExpr) Pos() lexer.Pos { return n.pos }

// NewCastExpr builds a CastExpr at pos.
func NewCastExpr(pos lexer.Pos, target Elem, x Expr) *CastExpr {
	return &CastExpr{pos: pos, Target: target, X: x}
}
