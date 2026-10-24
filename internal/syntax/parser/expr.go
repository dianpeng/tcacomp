package parser

import (
	"math"

	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Scalar expressions: precedence climbing per design §2.7. Each precedence
// level is one method; left-associative levels loop, the right-associative
// ternary and unary recurse.

// parseExprTop parses a stage-body's top-level expr with the pipe escape
// enabled (§7.1): a '|' followed by the keyword map/reduce terminates the
// stage instead of parsing as bitwise-or.
func (p *Parser) parseExprTop() ast.Expr {
	saved := p.allowPipe
	p.allowPipe = true
	e := p.parseExpr()
	p.allowPipe = saved
	return e
}

// parseExprNested parses an expr in a position where a pipe can never appear
// (paren groups, ternary branches, call arguments, clause groups): '|' is
// always bitwise-or there (§7.1).
func (p *Parser) parseExprNested() ast.Expr {
	saved := p.allowPipe
	p.allowPipe = false
	e := p.parseExpr()
	p.allowPipe = saved
	return e
}

func (p *Parser) parseExpr() ast.Expr { return p.parseTernary() }

// ternary ::= lor [ '?' expr ':' expr ] — C-style, lazy branches (rule 18).
// The branches are part of the stage's top-level expr, so they inherit the
// pipe-escape allowance (§7.1 distinguishes paren groups, not ternaries).
func (p *Parser) parseTernary() ast.Expr {
	x := p.parseLor()
	if !p.at(lexer.Question) {
		return x
	}
	q := p.advance()
	then := p.parseExpr()
	p.expect(lexer.Colon)
	els := p.parseExpr()
	return ast.NewSelectExpr(q.Pos, x, then, els)
}

func (p *Parser) parseLor() ast.Expr {
	x := p.parseLand()
	for p.at(lexer.OrOr) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseLand())
	}
	return x
}

func (p *Parser) parseLand() ast.Expr {
	x := p.parseBitor()
	for p.at(lexer.AndAnd) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseBitor())
	}
	return x
}

// parseBitor is the '|' level — and the pipe-escape point (§7.1). At a
// stage-body top level, a '|' whose next token is the keyword map or reduce
// terminates the stage: the token is left unconsumed for parseStageChain.
// Everywhere else '|' is bitwise-or.
func (p *Parser) parseBitor() ast.Expr {
	x := p.parseBitxor()
	for p.at(lexer.Pipe) {
		if p.allowPipe {
			nk := p.peek(1).Kind
			if nk == lexer.Map || nk == lexer.Reduce {
				return x
			}
		}
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseBitxor())
	}
	return x
}

func (p *Parser) parseBitxor() ast.Expr {
	x := p.parseBitand()
	for p.at(lexer.Caret) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseBitand())
	}
	return x
}

func (p *Parser) parseBitand() ast.Expr {
	x := p.parseEquality()
	for p.at(lexer.Amp) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseEquality())
	}
	return x
}

func (p *Parser) parseEquality() ast.Expr {
	x := p.parseRelational()
	for p.at(lexer.EqEq) || p.at(lexer.Ne) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseRelational())
	}
	return x
}

func (p *Parser) parseRelational() ast.Expr {
	x := p.parseAdditive()
	for {
		var t lexer.Token
		switch p.cur().Kind {
		case lexer.Lt, lexer.Le, lexer.Gt, lexer.Ge:
			t = p.advance()
		default:
			return x
		}
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseAdditive())
	}
}

func (p *Parser) parseAdditive() ast.Expr {
	x := p.parseMultiplicative()
	for p.at(lexer.Plus) || p.at(lexer.Minus) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseMultiplicative())
	}
	return x
}

func (p *Parser) parseMultiplicative() ast.Expr {
	x := p.parseUnary()
	for p.at(lexer.Star) || p.at(lexer.Slash) || p.at(lexer.Percent) {
		t := p.advance()
		x = ast.NewBinaryExpr(t.Pos, ast.BinOp(t.Lit), x, p.parseUnary())
	}
	return x
}

func (p *Parser) parseUnary() ast.Expr {
	switch p.cur().Kind {
	case lexer.Minus, lexer.Plus, lexer.Bang, lexer.Tilde:
		t := p.advance()
		return ast.NewUnaryExpr(t.Pos, ast.UnaryOp(t.Lit), p.parseUnary())
	}
	return p.parsePrimary()
}

// parsePrimary parses literal | identifier | builtinCall | castExpr |
// '(' expr ')' (design §2.7). There is no ParenExpr: parens never survive
// into the AST.
func (p *Parser) parsePrimary() ast.Expr {
	t := p.cur()
	switch {
	case t.Kind == lexer.IntLit || t.Kind == lexer.UintLit || t.Kind == lexer.FloatLit:
		p.advance()
		return ast.NewBasicLit(t.Pos, litKindOf(t.Kind), t.Lit, t.Val)
	case t.Kind == lexer.True || t.Kind == lexer.False:
		p.advance()
		return ast.NewBasicLit(t.Pos, ast.LitBool, t.Lit, t.Val)
	case t.Kind == lexer.InfF32:
		p.advance()
		return ast.NewBasicLit(t.Pos, ast.LitFloat, t.Lit, float32(math.Inf(1)))
	case t.Kind == lexer.Ident:
		if p.peek(1).Kind == lexer.LParen {
			return p.parseCall()
		}
		p.advance()
		return ast.NewNameRef(t.Pos, t.Lit)
	case lexer.IsElemType(t.Kind):
		p.advance()
		p.expect(lexer.LParen)
		x := p.parseExprNested()
		p.expect(lexer.RParen)
		return ast.NewCastExpr(t.Pos, ast.Elem(t.Lit), x)
	case t.Kind == lexer.LParen:
		p.advance()
		x := p.parseExprNested()
		p.expect(lexer.RParen)
		return x
	case t.Kind == lexer.Placeholder:
		// The pipe slot referenced from a stage body (Appendix A: acc + %0).
		// Position, not spelling, identifies the slot: this references param
		// 0 of the innermost enclosing stage, whether that param was spelled
		// %0 or named. Semacheck rejects a PipeRef in a stage that has no
		// pipe slot (a root map body, a reduce init).
		p.advance()
		if t.Val.(int) != 0 {
			p.errorf(t.Pos, "placeholder %s is reserved in v0 (only %%0 is legal, design §7.3)", t.Lit)
		}
		return ast.NewPipeRef(t.Pos)
	default:
		p.errorf(t.Pos, "expected expression, found %s", t.Kind)
		return nil
	}
}

// parseCall parses identifier '(' [ expr { ',' expr } [','] ] ')'.
func (p *Parser) parseCall() ast.Expr {
	name := p.advance()
	p.advance() // '('
	var args []ast.Expr
	for !p.at(lexer.RParen) && !p.at(lexer.EOF) {
		args = append(args, p.parseExprNested())
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	p.expect(lexer.RParen)
	return ast.NewCallExpr(name.Pos, name.Lit, args)
}

func litKindOf(k lexer.Kind) ast.LitKind {
	switch k {
	case lexer.UintLit:
		return ast.LitUint
	case lexer.FloatLit:
		return ast.LitFloat
	default:
		return ast.LitInt
	}
}
