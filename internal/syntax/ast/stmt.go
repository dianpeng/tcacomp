package ast

import "tcacomp/internal/syntax/lexer"

// --- statements (design §3.1; Stmt is an interface) ---

// Stmt is one kernel-body statement: *DimVarDecl, *LetDecl, or *ReturnStmt.
type Stmt interface {
	Node
	stmtNode()
}

// DimVarDecl is 'dimvar' identifierList ';' — the declaration of bounded
// logical iteration axes (design §2.5). A dimvar is not a scalar value: it
// may appear only in bindings and as lists.
type DimVarDecl struct {
	pos   lexer.Pos
	Names []string
}

func (n *DimVarDecl) astNode()       {}
func (n *DimVarDecl) stmtNode()      {}
func (n *DimVarDecl) Pos() lexer.Pos { return n.pos }

// NewDimVarDecl builds a DimVarDecl at pos.
func NewDimVarDecl(pos lexer.Pos, names []string) *DimVarDecl {
	return &DimVarDecl{pos: pos, Names: names}
}

// LetDecl is 'let' identifier [':' type] '=' stageChain ';' (design §2.5).
// Type is nil when the annotation is omitted (resolution infers and fills
// it, design §5.1 step 5).
type LetDecl struct {
	pos   lexer.Pos
	Name  string
	Type  TypeRef // nil = omitted (infer)
	Chain *StageChain
}

func (n *LetDecl) astNode()       {}
func (n *LetDecl) stmtNode()      {}
func (n *LetDecl) Pos() lexer.Pos { return n.pos }

// NewLetDecl builds a LetDecl at pos.
func NewLetDecl(pos lexer.Pos, name string, typ TypeRef, chain *StageChain) *LetDecl {
	return &LetDecl{pos: pos, Name: name, Type: typ, Chain: chain}
}

// ReturnStmt is 'return' identifier ';' — exactly one per kernel, the last
// statement, naming a let-bound tensor (design §2.5, rule 25).
type ReturnStmt struct {
	pos  lexer.Pos
	Name string
}

func (n *ReturnStmt) astNode()       {}
func (n *ReturnStmt) stmtNode()      {}
func (n *ReturnStmt) Pos() lexer.Pos { return n.pos }

// NewReturnStmt builds a ReturnStmt at pos.
func NewReturnStmt(pos lexer.Pos, name string) *ReturnStmt {
	return &ReturnStmt{pos: pos, Name: name}
}

// StageChain is stage { '|' stage } — one let's pipe (design §2.6). The
// parser has already applied the pipe-escape rule (§7.1), so no bitwise-or
// survives at the top level here.
type StageChain struct {
	pos    lexer.Pos
	Stages []Stage
}

func (n *StageChain) astNode()       {}
func (n *StageChain) Pos() lexer.Pos { return n.pos }

// NewStageChain builds a StageChain at pos.
func NewStageChain(pos lexer.Pos, stages []Stage) *StageChain {
	return &StageChain{pos: pos, Stages: stages}
}
