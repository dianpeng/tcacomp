package ast

import "tcacomp/internal/syntax/lexer"

// --- stage chain nodes (design §3.1) ---

// Stage is one pipe stage: *MapStage or *ReduceStage.
type Stage interface {
	Node
	stageNode()
}

// MapStage is 'map' 'on' '(' bindingList ')' [ 'as' '(' dimList ')' ] lambda
// (design §2.6). As is nil when the clause is omitted (infer; §7.4) — nil vs
// non-nil is load-bearing for resolution (design §3.2).
type MapStage struct {
	pos      lexer.Pos
	Bindings []*Binding
	As       []string // nil = omitted (infer)
	Lambda   *Lambda
}

func (n *MapStage) astNode()       {}
func (n *MapStage) stageNode()     {}
func (n *MapStage) Pos() lexer.Pos { return n.pos }

// NewMapStage builds a MapStage at pos.
func NewMapStage(pos lexer.Pos, bindings []*Binding, as []string, lambda *Lambda) *MapStage {
	return &MapStage{pos: pos, Bindings: bindings, As: as, Lambda: lambda}
}

// ReduceStage is 'reduce' '(' keyedInitList ')' 'on' '(' paramSpecList ')'
// [ 'as' '(' dimList ')' ] '->' scalarType ':' expr (design §2.6). OnParams
// holds exactly the accumulator then the piped element (rule 13). As is nil
// when omitted (infer, §7.4).
type ReduceStage struct {
	pos      lexer.Pos
	Init     Expr // nil when init = is omitted (design allows [ 'init' '=' initExpr ])
	OnParams []*ParamSpec
	As       []string // nil = omitted (infer)
	RetType  ScalarType
	Body     Expr
}

func (n *ReduceStage) astNode()       {}
func (n *ReduceStage) stageNode()     {}
func (n *ReduceStage) Pos() lexer.Pos { return n.pos }

// NewReduceStage builds a ReduceStage at pos.
func NewReduceStage(pos lexer.Pos, init Expr, onParams []*ParamSpec, as []string, retType ScalarType, body Expr) *ReduceStage {
	return &ReduceStage{pos: pos, Init: init, OnParams: onParams, As: as, RetType: retType, Body: body}
}

// Binding is one stage operand: identifier { '[' identifier ']' } — a tensor
// subscripted by dimvars, structurally position-bound (design §2.6).
type Binding struct {
	pos    lexer.Pos
	Tensor string
	Dims   []string
}

func (n *Binding) astNode()       {}
func (n *Binding) Pos() lexer.Pos { return n.pos }

// NewBinding builds a Binding at pos.
func NewBinding(pos lexer.Pos, tensor string, dims []string) *Binding {
	return &Binding{pos: pos, Tensor: tensor, Dims: dims}
}

// Lambda is [ captureList ] '(' paramSpecList ')' [ '->' scalarType ] ':'
// expr (design §2.6). RetType is nil when the return annotation is omitted
// (infer; §5.1). The body is exactly one expr (§2.7).
type Lambda struct {
	pos      lexer.Pos
	Captures []string
	Params   []*ParamSpec
	RetType  *ScalarType // nil = omitted (infer)
	Body     Expr
}

func (n *Lambda) astNode()       {}
func (n *Lambda) Pos() lexer.Pos { return n.pos }

// NewLambda builds a Lambda at pos.
func NewLambda(pos lexer.Pos, captures []string, params []*ParamSpec, retType *ScalarType, body Expr) *Lambda {
	return &Lambda{pos: pos, Captures: captures, Params: params, RetType: retType, Body: body}
}

// ParamSpec is one stage parameter spec: '%' digits [ ':' scalarType ] or
// identifier ':' scalarType (design §2.6). Pipe is true exactly when the spec
// was spelled %0 (the only legal placeholder in v0, §7.3); Type is nil when
// the annotation is omitted (infer; checked against the pipe/operand type).
type ParamSpec struct {
	pos  lexer.Pos
	Pipe bool
	Name string
	Type *ScalarType // nil = omitted (infer)
}

func (n *ParamSpec) astNode()       {}
func (n *ParamSpec) Pos() lexer.Pos { return n.pos }

// NewParamSpec builds a ParamSpec at pos.
func NewParamSpec(pos lexer.Pos, pipe bool, name string, typ *ScalarType) *ParamSpec {
	return &ParamSpec{pos: pos, Pipe: pipe, Name: name, Type: typ}
}
