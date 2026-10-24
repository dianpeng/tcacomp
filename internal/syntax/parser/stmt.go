package parser

import (
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Statements, stage chains, and stage clauses (design §2.5, §2.6).

// parseStmtList parses { stmt } up to '}' or EOF (design §2.5).
func (p *Parser) parseStmtList() []ast.Stmt {
	var stmts []ast.Stmt
	for !p.at(lexer.RBrace) && !p.at(lexer.EOF) {
		s := p.parseStmt()
		if s != nil {
			stmts = append(stmts, s)
		}
	}
	return stmts
}

func (p *Parser) parseStmt() ast.Stmt {
	switch p.cur().Kind {
	case lexer.DimVar:
		return p.parseDimVarDecl()
	case lexer.Let:
		return p.parseLetDecl()
	case lexer.Return:
		return p.parseReturnStmt()
	default:
		p.errorf(p.cur().Pos, "expected statement, found %s", p.cur().Kind)
		p.sync()
		return nil
	}
}

// parseDimVarDecl parses 'dimvar' (identifierList | '(' identifierList ')'
// ';' (design §2.5).
func (p *Parser) parseDimVarDecl() ast.Stmt {
	kw := p.advance() // 'dimvar'
	paren := p.at(lexer.LParen)
	if paren {
		p.advance()
	}
	names := p.parseIdentList()
	if paren {
		p.expect(lexer.RParen)
	}
	p.expect(lexer.Semi)
	return ast.NewDimVarDecl(kw.Pos, names)
}

// parseIdentList parses [ identifier { ',' identifier } [','] ].
func (p *Parser) parseIdentList() []string {
	var names []string
	for p.at(lexer.Ident) {
		names = append(names, p.advance().Lit)
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	return names
}

// parseLetDecl parses 'let' identifier [':' type] '=' stageChain ';'.
func (p *Parser) parseLetDecl() ast.Stmt {
	kw := p.advance() // 'let'
	name := p.expect(lexer.Ident)
	var typ ast.TypeRef
	if p.at(lexer.Colon) {
		p.advance()
		typ = p.parseType()
	}
	p.expect(lexer.Assign)
	chain := p.parseStageChain()
	p.expect(lexer.Semi)
	return ast.NewLetDecl(kw.Pos, name.Lit, typ, chain)
}

// parseReturnStmt parses 'return' identifier ';'.
func (p *Parser) parseReturnStmt() ast.Stmt {
	kw := p.advance() // 'return'
	name := p.expect(lexer.Ident)
	p.expect(lexer.Semi)
	return ast.NewReturnStmt(kw.Pos, name.Lit)
}

// --- stage chains (design §2.6; pipe escape §7.1 lives in expr.go) ---

// parseStageChain parses stage { '|' stage }.
func (p *Parser) parseStageChain() *ast.StageChain {
	pos := p.cur().Pos
	var stages []ast.Stage
	for {
		st := p.parseStage()
		if st != nil {
			stages = append(stages, st)
		}
		if !p.at(lexer.Pipe) {
			break
		}
		p.advance() // the pipe token (escape was decided inside parseBitor)
	}
	return ast.NewStageChain(pos, stages)
}

// parseStage parses one mapStage or reduceStage.
func (p *Parser) parseStage() ast.Stage {
	switch p.cur().Kind {
	case lexer.Map:
		return p.parseMapStage()
	case lexer.Reduce:
		return p.parseReduceStage()
	default:
		p.errorf(p.cur().Pos, "expected stage (map or reduce), found %s", p.cur().Kind)
		p.sync()
		return nil
	}
}

// parseMapStage parses 'map' 'on' '(' bindingList ')' [ 'as' '(' dimList ')'
// ] lambda (design §2.6).
func (p *Parser) parseMapStage() ast.Stage {
	kw := p.advance() // 'map'
	p.expect(lexer.On)
	p.expect(lexer.LParen)
	bindings := p.parseBindingList()
	p.expect(lexer.RParen)
	var as []string
	if p.at(lexer.As) {
		as = p.parseDimList()
	}
	lambda := p.parseLambda()
	if lambda == nil {
		return nil
	}
	return ast.NewMapStage(kw.Pos, bindings, as, lambda)
}

// parseReduceStage parses 'reduce' '(' [ 'init' '=' initExpr ] ')' 'on' '('
// paramSpecList ')' [ 'as' '(' dimList ')' ] '->' scalarType ':' expr.
func (p *Parser) parseReduceStage() ast.Stage {
	kw := p.advance() // 'reduce'
	p.expect(lexer.LParen)
	var init ast.Expr
	if p.at(lexer.Init) {
		p.advance()
		p.expect(lexer.Assign)
		init = p.parseExprNested() // inside the clause group: '|' is never a pipe
	}
	p.expect(lexer.RParen)
	p.expect(lexer.On)
	p.expect(lexer.LParen)
	onParams := p.parseParamSpecList()
	p.expect(lexer.RParen)
	var as []string
	if p.at(lexer.As) {
		as = p.parseDimList()
	}
	p.expect(lexer.Arrow)
	retTok := p.cur()
	ret := p.parseScalarTypeOpt()
	if ret == nil {
		p.errorf(retTok.Pos, "reduce requires '->' scalarType, found %s", retTok.Kind)
		ret = ast.NewScalarType(retTok.Pos, ast.EF32)
	}
	p.expect(lexer.Colon)
	body := p.parseExprTop()
	return ast.NewReduceStage(kw.Pos, init, onParams, as, *ret, body)
}

// parseBindingList parses [ binding { ',' binding } [','] ] where binding is
// identifier { '[' identifier ']' } (design §2.6).
func (p *Parser) parseBindingList() []*ast.Binding {
	var bindings []*ast.Binding
	for p.at(lexer.Ident) {
		name := p.advance()
		var dims []string
		for p.at(lexer.LBracket) {
			p.advance()
			d := p.expect(lexer.Ident)
			if d.Pos.IsValid() {
				dims = append(dims, d.Lit)
			}
			p.expect(lexer.RBracket)
		}
		bindings = append(bindings, ast.NewBinding(name.Pos, name.Lit, dims))
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	return bindings
}

// parseDimList parses 'as' '(' [ identifier { ',' identifier } [','] ] ')'.
func (p *Parser) parseDimList() []string {
	p.advance() // 'as'
	p.expect(lexer.LParen)
	names := p.parseIdentList()
	p.expect(lexer.RParen)
	return names
}

// parseLambda parses [ captureList ] '(' paramSpecList ')' [ '->' scalarType ]
// ':' expr (design §2.6).
func (p *Parser) parseLambda() *ast.Lambda {
	pos := p.cur().Pos
	var captures []string
	if p.at(lexer.LBracket) {
		p.advance()
		captures = p.parseIdentList()
		p.expect(lexer.RBracket)
	}
	p.expect(lexer.LParen)
	params := p.parseParamSpecList()
	p.expect(lexer.RParen)
	var retType *ast.ScalarType
	if p.at(lexer.Arrow) {
		p.advance()
		retTok := p.cur()
		retType = p.parseScalarTypeOpt()
		if retType == nil {
			p.errorf(retTok.Pos, "lambda return annotation must be an element type, found %s", retTok.Kind)
			retType = ast.NewScalarType(retTok.Pos, ast.EF32)
		}
	}
	p.expect(lexer.Colon)
	body := p.parseExprTop()
	return ast.NewLambda(pos, captures, params, retType, body)
}

// parseParamSpecList parses [ paramSpec { ',' paramSpec } [','] ] where
// paramSpec is '%' digits [':' scalarType] | identifier ':' scalarType
// (design §2.6; §7.3 placeholder rules).
func (p *Parser) parseParamSpecList() []*ast.ParamSpec {
	var specs []*ast.ParamSpec
	for !p.at(lexer.RParen) && !p.at(lexer.EOF) {
		spec := p.parseParamSpec()
		if spec != nil {
			specs = append(specs, spec)
		}
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	return specs
}

func (p *Parser) parseParamSpec() *ast.ParamSpec {
	t := p.cur()
	switch {
	case t.Kind == lexer.Placeholder:
		p.advance()
		if t.Val.(int) != 0 {
			p.errorf(t.Pos, "placeholder %s is reserved in v0 (only %%0 is legal, design §7.3)", t.Lit)
		}
		var typ *ast.ScalarType
		if p.at(lexer.Colon) {
			p.advance()
			typ = p.parseScalarTypeOpt()
			if typ == nil {
				p.errorf(p.cur().Pos, "placeholder annotation must be an element type, found %s", p.cur().Kind)
			}
		}
		return ast.NewParamSpec(t.Pos, true, t.Lit, typ)
	case t.Kind == lexer.Ident:
		p.advance()
		p.expect(lexer.Colon)
		typ := p.parseScalarTypeOpt()
		if typ == nil {
			p.errorf(p.cur().Pos, "parameter %s requires an element type, found %s", t.Lit, p.cur().Kind)
			typ = ast.NewScalarType(p.cur().Pos, ast.EF32)
		}
		return ast.NewParamSpec(t.Pos, false, t.Lit, typ)
	default:
		p.errorf(t.Pos, "expected stage parameter, found %s", t.Kind)
		return nil
	}
}
