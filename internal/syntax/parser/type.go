package parser

import (
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Type parsing (design §2.4).

// parseType parses tensorType | scalarType: the annotation position after a
// parameter name, a let's ':', or the kernel return arrow.
func (p *Parser) parseType() ast.TypeRef {
	if p.at(lexer.Ident) && p.cur().Lit == "tensor" {
		return p.parseTensorType()
	}
	if st := p.parseScalarTypeOpt(); st != nil {
		return st
	}
	p.errorf(p.cur().Pos, "expected type, found %s", p.cur().Kind)
	return ast.NewScalarType(p.cur().Pos, ast.EF32) // poison placeholder; error recorded
}

// parseScalarTypeOpt parses one element-type keyword as a *ScalarType, or
// returns nil when the current token is not an element type.
func (p *Parser) parseScalarTypeOpt() *ast.ScalarType {
	if !lexer.IsElemType(p.cur().Kind) {
		return nil
	}
	t := p.advance()
	return ast.NewScalarType(t.Pos, ast.Elem(t.Lit))
}

// parseTensorType parses 'tensor' '<' 'dtype' '=' scalarType [ ','
// 'shape' '=' '[' shapeDimList ']' ] [','] '>' (design §2.4). The tensor/
// dtype/shape spellings are contextual identifiers, not keywords.
func (p *Parser) parseTensorType() ast.TypeRef {
	kw := p.advance() // 'tensor'
	p.expect(lexer.Lt)
	p.expectContextual("dtype")
	p.expect(lexer.Assign)
	elemTok := p.cur()
	elem := p.parseScalarTypeOpt()
	if elem == nil {
		p.errorf(elemTok.Pos, "tensor dtype must be an element type, found %s", elemTok.Kind)
		p.sync()
		return ast.NewTensorType(kw.Pos, ast.ScalarType{Elem: ast.EF32}, nil)
	}
	var shape []ast.ShapeDim
	for p.at(lexer.Comma) {
		p.advance()
		if p.at(lexer.Ident) && p.cur().Lit == "shape" {
			p.advance()
			p.expect(lexer.Assign)
			p.expect(lexer.LBracket)
			shape = p.parseShapeDimList()
			p.expect(lexer.RBracket)
		}
	}
	p.expect(lexer.Gt)
	return ast.NewTensorType(kw.Pos, *elem, shape)
}

// expectContextual consumes an identifier with the given spelling, reporting
// an error otherwise (contextual keywords are matched by text, not kind).
func (p *Parser) expectContextual(lit string) lexer.Token {
	if p.at(lexer.Ident) && p.cur().Lit == lit {
		return p.advance()
	}
	p.errorf(p.cur().Pos, "expected %q, found %s", lit, p.describe())
	return lexer.Token{}
}

// describe renders the current token for diagnostics (literal spellings for
// identifiers, the kind name otherwise).
func (p *Parser) describe() string {
	if p.at(lexer.Ident) {
		return "identifier " + p.cur().Lit
	}
	return p.cur().Kind.String()
}

// parseShapeDimList parses [ shapeDim { ',' shapeDim } [','] ] (design §2.4).
func (p *Parser) parseShapeDimList() []ast.ShapeDim {
	var dims []ast.ShapeDim
	for !p.at(lexer.RBracket) && !p.at(lexer.EOF) {
		t := p.cur()
		switch {
		case t.Kind == lexer.UintLit:
			p.advance()
			dims = append(dims, *ast.NewShapeDim(t.Pos, "", uint64(t.Val.(uint32)), false))
		case t.Kind == lexer.Ident:
			p.advance()
			dims = append(dims, *ast.NewShapeDim(t.Pos, t.Lit, 0, true))
		default:
			p.errorf(t.Pos, "expected shape dimension (uint literal or identifier), found %s", t.Kind)
			p.advance() // make progress; the enclosing ']' expectation reports further damage
			return dims
		}
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	return dims
}
