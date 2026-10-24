package ast

import "tcacomp/internal/syntax/lexer"

// --- kernel (design §3.1) ---

// Kernel is one 'kernel' identifier '(' paramList ')' '->' type '{' stmtList
// '}' declaration with its attached annotations.
type Kernel struct {
	pos    lexer.Pos
	Anns   []*Annotation // pending annotations attached at parse (§7.2)
	Name   string
	Params []*Param
	Ret    TypeRef
	Body   []Stmt
}

func (n *Kernel) astNode()       {}
func (n *Kernel) Pos() lexer.Pos { return n.pos }

// NewKernel builds a Kernel at pos.
func NewKernel(pos lexer.Pos, anns []*Annotation, name string, params []*Param, ret TypeRef, body []Stmt) *Kernel {
	return &Kernel{pos: pos, Anns: anns, Name: name, Params: params, Ret: ret, Body: body}
}

// ParamRole distinguishes tensor parameters from launch-uniform scalar
// parameters. The parser leaves Role unset (ParamRoleUnknown); semacheck
// fills it from the declared type (design §3.1: set by semacheck).
type ParamRole int

const (
	ParamRoleUnknown ParamRole = iota
	TensorParam
	ScalarParam
)

// Param is one kernel parameter: identifier ':' paramType. Type is a
// *TensorType or *ScalarType; Role is filled by semacheck.
type Param struct {
	pos  lexer.Pos
	Name string
	Type TypeRef
	Role ParamRole
}

func (n *Param) astNode()       {}
func (n *Param) Pos() lexer.Pos { return n.pos }

// NewParam builds a Param at pos with unknown role.
func NewParam(pos lexer.Pos, name string, typ TypeRef) *Param {
	return &Param{pos: pos, Name: name, Type: typ}
}
