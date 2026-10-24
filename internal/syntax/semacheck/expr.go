package semacheck

import (
	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
)

// Expression typing (design §4.6 rules 17-22), mirroring pkg/expr.go's
// homogeneous rules exactly — the same operator classes, the same shared
// cast registry (ir.CastPairOK), the same builtin registry (ir.Builtin).
// A violation records one diagnostic and yields a poison type so checking
// continues (every node typed, no fixed point needed: the expr AST is a
// tree, §5.1 step 2).

// exprScope is the name environment of one expression: lambda params,
// captures (map lambdas only), and the pipe slot. Reduce-init expressions
// instead seed params with the kernel scalar params directly.
type exprScope struct {
	params       map[string]ir.ElemType
	captures     map[string]ir.ElemType
	usedCaptures map[string]bool
	hasPipe      bool
	pipeElem     ir.ElemType
}

func newExprScope() *exprScope {
	return &exprScope{
		params:       map[string]ir.ElemType{},
		captures:     map[string]ir.ElemType{},
		usedCaptures: map[string]bool{},
	}
}

// typeExpr computes e's element type in scope, recording one diagnostic per
// violation and returning a poison type (TF32) after an error.
func (c *checker) typeExpr(e ast.Expr, scope *exprScope) ir.ElemType {
	switch e := e.(type) {
	case *ast.BasicLit:
		switch e.Kind {
		case ast.LitInt:
			return ir.TI32
		case ast.LitUint:
			return ir.TU32
		case ast.LitFloat:
			return ir.TF32
		case ast.LitBool:
			return ir.TBool
		}
		return ir.TF32
	case *ast.NameRef:
		if t, ok := scope.params[e.Name]; ok {
			return t
		}
		if t, ok := scope.captures[e.Name]; ok {
			scope.usedCaptures[e.Name] = true
			return t
		}
		c.errorf(e.Pos(), "unknown name %q in expression (rule 15: only lambda params, captures, and kernel scalar params in init are visible)", e.Name)
		return ir.TF32
	case *ast.PipeRef:
		if !scope.hasPipe {
			c.errorf(e.Pos(), "%%0 has no pipe slot here (only piped maps and reduce bodies carry one)")
			return ir.TF32
		}
		return scope.pipeElem
	case *ast.BinaryExpr:
		return c.typeBinary(e, scope)
	case *ast.UnaryExpr:
		return c.typeUnary(e, scope)
	case *ast.SelectExpr:
		condT := c.typeExpr(e.Cond, scope)
		if condT != ir.TBool {
			c.errorf(e.Cond.Pos(), "select condition must be bool, got %s (rule 18)", elemName(condT))
		}
		thenT := c.typeExpr(e.Then, scope)
		elseT := c.typeExpr(e.Else, scope)
		if thenT != elseT {
			c.errorf(e.Pos(), "select branches must have identical types, got %s vs %s (rule 18)", elemName(thenT), elemName(elseT))
			return thenT
		}
		return thenT
	case *ast.CallExpr:
		return c.typeCall(e, scope)
	case *ast.CastExpr:
		from := c.typeExpr(e.X, scope)
		to, ok := elemOf(e.Target)
		if !ok {
			c.errorf(e.Pos(), "unknown cast target %q", e.Target)
			return ir.TF32
		}
		if from == to {
			c.errorf(e.Pos(), "cast: redundant cast to %s (rule 19)", elemName(to))
			return to
		}
		if !ir.CastPairOK(from, to) {
			c.errorf(e.Pos(), "cast: pair %s -> %s is not in the v0 cast registry (arithmetic-to-arithmetic only, rule 19)", elemName(from), elemName(to))
		}
		return to
	}
	c.errorf(e.Pos(), "unknown expression node %T", e)
	return ir.TF32
}

// typeBinary types one binary operator by class (rule 17).
func (c *checker) typeBinary(e *ast.BinaryExpr, scope *exprScope) ir.ElemType {
	op, ok := binOpOf(e.Op)
	if !ok {
		c.errorf(e.Pos(), "unknown operator %q", e.Op)
		return ir.TF32
	}
	lt := c.typeExpr(e.L, scope)
	rt := c.typeExpr(e.R, scope)
	if lt != rt {
		c.errorf(e.Pos(), "%s: operand types differ (%s vs %s); implicit promotions are not supported (rule 17)", op, elemName(lt), elemName(rt))
		return lt
	}
	switch op {
	case ir.BinOpLOr, ir.BinOpLAnd:
		if lt != ir.TBool {
			c.errorf(e.Pos(), "%s: operands must be bool, got %s (rule 17)", op, elemName(lt))
		}
		return ir.TBool
	case ir.BinOpEq, ir.BinOpNE, ir.BinOpLT, ir.BinOpLE, ir.BinOpGT, ir.BinOpGE:
		return ir.TBool
	case ir.BinOpMod, ir.BinOpShL, ir.BinOpShR, ir.BinOpAnd, ir.BinOpOr, ir.BinOpXor:
		if !isIntElem(lt) {
			c.errorf(e.Pos(), "%s: operands must have the same integer type, got %s (rule 17)", op, elemName(lt))
		}
		return lt
	default: // + - * /
		if !isArithElem(lt) {
			c.errorf(e.Pos(), "%s: operands must have the same arithmetic type, got %s (rule 17)", op, elemName(lt))
		}
		return lt
	}
}

// typeUnary types one unary operator by class (rule 17).
func (c *checker) typeUnary(e *ast.UnaryExpr, scope *exprScope) ir.ElemType {
	op, ok := unaryOpOf(e.Op)
	if !ok {
		c.errorf(e.Pos(), "unknown unary operator %q", e.Op)
		return ir.TF32
	}
	t := c.typeExpr(e.X, scope)
	switch op {
	case ir.UnaryOpLNot:
		if t != ir.TBool {
			c.errorf(e.Pos(), "%s: operand must be bool, got %s (rule 17)", op, elemName(t))
		}
	case ir.UnaryOpNot:
		if !isIntElem(t) {
			c.errorf(e.Pos(), "%s: operand must have an integer type, got %s (rule 17)", op, elemName(t))
		}
	default: // - +
		if !isArithElem(t) {
			c.errorf(e.Pos(), "%s: operand must have an arithmetic type, got %s (rule 17)", op, elemName(t))
		}
	}
	return t
}

// typeCall resolves a pure builtin against the ir.Builtin registry (rule 20):
// registered name, exact arity, arguments of the signature's argument type,
// result of the signature's result type.
func (c *checker) typeCall(e *ast.CallExpr, scope *exprScope) ir.ElemType {
	sig, ok := ir.Builtin(e.Name)
	if !ok {
		c.errorf(e.Pos(), "function %s is not a registered pure scalar builtin (rule 20)", e.Name)
		return ir.TF32
	}
	if len(e.Args) != sig.Arity {
		c.errorf(e.Pos(), "builtin %s expects %d argument(s), got %d (rule 20)", e.Name, sig.Arity, len(e.Args))
	}
	for i, a := range e.Args {
		t := c.typeExpr(a, scope)
		if len(e.Args) == sig.Arity && t != sig.Arg {
			c.errorf(a.Pos(), "call %s: argument %d must be %s, got %s (rule 20)", e.Name, i, elemName(sig.Arg), elemName(t))
		}
	}
	return sig.Result
}
