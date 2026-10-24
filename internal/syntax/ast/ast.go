// Package ast is the TCA frontend's untyped syntax tree (doc/syntax-design.md
// §3): Go interfaces with concrete structs per node and a typed-switch
// visitor, exactly the go/ast shape (project decision (a) in CLAUDE.md).
//
// Layering gate: this package must NOT import the IR package (internal/ir).
// The design names ir.ElemType in two node fields; per the gate the AST
// instead carries the canonical source spelling (ast.Elem, e.g. "f32"), and
// semacheck — which may import ir — owns the 1:1 conversion. BinOp/UnaryOp
// are likewise the operator's source spelling, mapped onto the IR opcode
// tables by semacheck.
//
// Every node carries its lexer.Pos. The tree is untyped at parse time;
// semacheck fills the omitted annotation cells IN PLACE (Param.Role,
// ParamSpec.Type, Lambda.RetType, Stage As lists, LetDecl.Type — design §5)
// and returns diagnostics; there is no parallel typed-AST hierarchy.
package ast

import "tcacomp/internal/syntax/lexer"

// Node is every AST node's interface (design §3.1).
type Node interface {
	Pos() lexer.Pos
	astNode()
}

// --- module (design §3.1) ---

// Module is one parsed compilation unit: the include requests, the optional
// runtime annotation, and the kernel declarations, in source order.
type Module struct {
	Includes []*Include
	Runtime  *RuntimeAnn // nil when the module has no @runtime
	Kernels  []*Kernel
}

func (*Module) astNode() {}

// Pos returns the module's opening position: its first include, runtime
// annotation, or kernel (zero Pos for an empty module).
func (m *Module) Pos() lexer.Pos {
	switch {
	case len(m.Includes) > 0:
		return m.Includes[0].Pos()
	case m.Runtime != nil:
		return m.Runtime.Pos()
	case len(m.Kernels) > 0:
		return m.Kernels[0].Pos()
	}
	return lexer.Pos{}
}

// Include is one '@include' stringLit ';' request. Path is the literal text;
// Resolved is set by the module loader to the resolved include path.
type Include struct {
	pos      lexer.Pos
	Path     string
	Resolved string
}

func (n *Include) astNode()       {}
func (n *Include) Pos() lexer.Pos { return n.pos }

// NewInclude builds an Include at pos for the literal path.
func NewInclude(pos lexer.Pos, path string) *Include {
	return &Include{pos: pos, Path: path}
}

// RuntimeAnn is the module-global '@runtime' '(' keyedConstList ')' ';'
// declaration (design §2.3; at most one per module, rule 3).
type RuntimeAnn struct {
	pos  lexer.Pos
	Args []*KeyedConst
}

func (n *RuntimeAnn) astNode()       {}
func (n *RuntimeAnn) Pos() lexer.Pos { return n.pos }

// NewRuntimeAnn builds a RuntimeAnn at pos.
func NewRuntimeAnn(pos lexer.Pos, args []*KeyedConst) *RuntimeAnn {
	return &RuntimeAnn{pos: pos, Args: args}
}

// Annotation is one '@' identifier '(' keyedConstList ')' attachable
// declaration (design §2.3). The parser records annotations as pending and
// attaches them to the next kernel; a pending annotation followed by anything
// else is a semantic error (rule: annotation @x does not precede a kernel).
type Annotation struct {
	pos  lexer.Pos
	Name string
	Args []*KeyedConst
}

func (n *Annotation) astNode()       {}
func (n *Annotation) Pos() lexer.Pos { return n.pos }

// NewAnnotation builds an Annotation at pos.
func NewAnnotation(pos lexer.Pos, name string, args []*KeyedConst) *Annotation {
	return &Annotation{pos: pos, Name: name, Args: args}
}

// KeyedConst is one identifier '=' constValue entry of an annotation's or
// runtime declaration's keyed list (design §2.3).
type KeyedConst struct {
	pos   lexer.Pos
	Key   string
	Value ConstValue
}

func (n *KeyedConst) astNode()       {}
func (n *KeyedConst) Pos() lexer.Pos { return n.pos }

// NewKeyedConst builds a KeyedConst at pos.
func NewKeyedConst(pos lexer.Pos, key string, value ConstValue) *KeyedConst {
	return &KeyedConst{pos: pos, Key: key, Value: value}
}

// ConstKind is the constant-value family of a KeyedConst (design §2.3
// constValue: intLit | uintLit | floatLit | stringLit | 'true' | 'false').
type ConstKind int

const (
	ConstInt ConstKind = iota
	ConstUint
	ConstFloat
	ConstString
	ConstBool
)

// ConstValue is one parsed constant: its family, exact source text, and
// parsed value (int32 / uint32 / float32 / string / bool).
type ConstValue struct {
	Kind ConstKind
	Text string
	Val  any
}
