// Package parser is the TCA frontend's recursive-descent parser
// (doc/syntax-design.md §2, §3; CLAUDE.md decision (a)): one method per
// grammar level, precedence climbing for scalar expressions (§2.7), producing
// the untyped ast tree. The pipe-escape rule (§7.1) lives here. Eager
// syntactic checks only — everything semantic is semacheck's (§4).
//
// The parser consumes an eager token slice: the lexer runs to EOF up front
// (its error handler feeds the same diagnostic list), so lookahead is
// indexing, not scanning.
package parser

import (
	"fmt"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Parser is one parsing session over a single lexed file.
type Parser struct {
	f     *lexer.File
	toks  []lexer.Token
	pos   int
	diags []diagnostic.Diagnostic

	// allowPipe enables the §7.1 pipe escape: at the top-level expr of a
	// stage body a '|' followed by the keyword map/reduce terminates the
	// stage; everywhere else (paren groups, ternary branches, call args)
	// '|' is always bitwise-or.
	allowPipe bool
}

// Parse lexes f and parses it as one module, returning the best-effort tree
// and every lexical/syntactic diagnostic. The module is non-nil even with
// errors (declaration granularity recovery); callers must check for
// Error-severity diagnostics before using the tree.
func Parse(f *lexer.File) (*ast.Module, []diagnostic.Diagnostic) {
	p := &Parser{f: f}
	s := lexer.NewScanner(f, func(pos lexer.Pos, msg string) {
		p.diags = append(p.diags, diagnostic.Diagnostic{
			Severity: diagnostic.Error,
			Span:     spanOf(pos),
			Message:  msg,
		})
	})
	for {
		tok := s.Next()
		p.toks = append(p.toks, tok)
		if tok.Kind == lexer.EOF {
			break
		}
	}
	m := p.parseModule()
	return m, p.diags
}

// --- token helpers ---

func (p *Parser) cur() lexer.Token { return p.toks[p.pos] }
func (p *Parser) peek(n int) lexer.Token {
	if p.pos+n < len(p.toks) {
		return p.toks[p.pos+n]
	}
	return p.toks[len(p.toks)-1] // EOF
}
func (p *Parser) at(k lexer.Kind) bool { return p.cur().Kind == k }
func (p *Parser) advance() lexer.Token {
	t := p.toks[p.pos]
	if p.pos < len(p.toks)-1 {
		p.pos++
	}
	return t
}

// expect consumes and returns the current token if it has kind k; otherwise it
// reports one error at the current position and returns the zero token.
func (p *Parser) expect(k lexer.Kind) lexer.Token {
	if p.at(k) {
		return p.advance()
	}
	p.errorf(p.cur().Pos, "expected %s, found %s", k, p.cur().Kind)
	return lexer.Token{}
}

func (p *Parser) errorf(pos lexer.Pos, format string, args ...any) {
	p.diags = append(p.diags, diagnostic.Diagnostic{
		Severity: diagnostic.Error,
		Span:     spanOf(pos),
		Message:  fmt.Sprintf(format, args...),
	})
}

// spanOf converts a lexer position into a diagnostic span.
func spanOf(pos lexer.Pos) diagnostic.Span {
	span := diagnostic.Span{}
	if pos.File != nil {
		span.File = pos.File.Name
	}
	span.Line, span.Col = pos.Line, pos.Col
	return span
}

// sync skips to the next statement/declaration boundary after an error.
func (p *Parser) sync() {
	for !p.at(lexer.EOF) && !p.at(lexer.Semi) && !p.at(lexer.RBrace) && !p.at(lexer.Kernel) && !p.at(lexer.At) {
		p.advance()
	}
	if p.at(lexer.Semi) {
		p.advance()
	}
}

// --- module and declarations (design §2.3) ---

func (p *Parser) parseModule() *ast.Module {
	m := &ast.Module{}
	var pending []*ast.Annotation
	for !p.at(lexer.EOF) {
		switch {
		case p.at(lexer.At):
			if inc, rt, ann, ok := p.parseAtDecl(); ok {
				switch {
				case inc != nil:
					m.Includes = append(m.Includes, inc)
				case rt != nil:
					m.Runtime = rt
				default:
					pending = append(pending, ann)
				}
			}
		case p.at(lexer.Kernel):
			if k := p.parseKernel(pending); k != nil {
				m.Kernels = append(m.Kernels, k)
			}
			pending = nil
		default:
			p.errorf(p.cur().Pos, "expected declaration, found %s", p.cur().Kind)
			p.sync()
		}
	}
	for _, ann := range pending {
		p.errorf(ann.Pos(), "annotation @%s does not precede a kernel", ann.Name)
	}
	return m
}

// parseAtDecl parses one '@' declaration: an @include, the @runtime, or a
// pending annotation. ok is false when a syntax error was reported (recovery
// already performed by the caller's sync).
func (p *Parser) parseAtDecl() (inc *ast.Include, rt *ast.RuntimeAnn, ann *ast.Annotation, ok bool) {
	at := p.expect(lexer.At)
	name := p.expect(lexer.Ident)
	if !name.Pos.IsValid() {
		p.sync()
		return nil, nil, nil, false
	}
	switch name.Lit {
	case "include":
		// includeDecl ::= '@include' stringLit ';' — no keyed-const list.
		path := p.cur()
		if path.Kind != lexer.StringLit {
			p.errorf(path.Pos, "@include expects a single string literal path, found %s", path.Kind)
			p.sync()
			return nil, nil, nil, false
		}
		p.advance()
		p.expect(lexer.Semi)
		return ast.NewInclude(at.Pos, path.Val.(string)), nil, nil, true
	case "runtime":
		if rt != nil {
			p.errorf(at.Pos, "at most one @runtime per module (rule 3)")
		}
		args := p.parseKeyedConstList()
		p.expect(lexer.Semi)
		return nil, ast.NewRuntimeAnn(at.Pos, args), nil, true
	default:
		ann = ast.NewAnnotation(at.Pos, name.Lit, p.parseKeyedConstList())
		return nil, nil, ann, true
	}
}

// parseKeyedConstList parses '(' [ keyedConst { ',' keyedConst } [',' ] ] ')'.
func (p *Parser) parseKeyedConstList() []*ast.KeyedConst {
	p.expect(lexer.LParen)
	var args []*ast.KeyedConst
	for !p.at(lexer.RParen) && !p.at(lexer.EOF) {
		key := p.expect(lexer.Ident)
		if !key.Pos.IsValid() {
			p.sync()
			return args
		}
		p.expect(lexer.Assign)
		v := p.parseConstValue()
		args = append(args, ast.NewKeyedConst(key.Pos, key.Lit, v))
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	p.expect(lexer.RParen)
	return args
}

// parseConstValue parses one constValue (design §2.3).
func (p *Parser) parseConstValue() ast.ConstValue {
	t := p.cur()
	switch t.Kind {
	case lexer.IntLit, lexer.UintLit, lexer.FloatLit, lexer.StringLit:
		p.advance()
		return constOf(t)
	case lexer.True, lexer.False:
		p.advance()
		return ast.ConstValue{Kind: ast.ConstBool, Text: t.Lit, Val: t.Val}
	default:
		p.errorf(t.Pos, "expected constant value, found %s", t.Kind)
		return ast.ConstValue{}
	}
}

func constOf(t lexer.Token) ast.ConstValue {
	switch t.Kind {
	case lexer.IntLit:
		return ast.ConstValue{Kind: ast.ConstInt, Text: t.Lit, Val: t.Val}
	case lexer.UintLit:
		return ast.ConstValue{Kind: ast.ConstUint, Text: t.Lit, Val: t.Val}
	case lexer.FloatLit:
		return ast.ConstValue{Kind: ast.ConstFloat, Text: t.Lit, Val: t.Val}
	default: // StringLit
		return ast.ConstValue{Kind: ast.ConstString, Text: t.Val.(string), Val: t.Val}
	}
}

// --- kernel declaration (design §2.3, §2.4, §2.5) ---

func (p *Parser) parseKernel(anns []*ast.Annotation) *ast.Kernel {
	kw := p.expect(lexer.Kernel)
	name := p.expect(lexer.Ident)
	if !name.Pos.IsValid() {
		p.sync()
		return nil
	}
	p.expect(lexer.LParen)
	params := p.parseParamList()
	p.expect(lexer.RParen)
	p.expect(lexer.Arrow)
	ret := p.parseType()
	p.expect(lexer.LBrace)
	body := p.parseStmtList()
	p.expect(lexer.RBrace)
	return ast.NewKernel(kw.Pos, anns, name.Lit, params, ret, body)
}

// parseParamList parses [ param { ',' param } [','] ] (design §2.4).
func (p *Parser) parseParamList() []*ast.Param {
	var params []*ast.Param
	for !p.at(lexer.RParen) && !p.at(lexer.EOF) {
		name := p.expect(lexer.Ident)
		if !name.Pos.IsValid() {
			p.sync()
			return params
		}
		p.expect(lexer.Colon)
		typ := p.parseType()
		params = append(params, ast.NewParam(name.Pos, name.Lit, typ))
		if p.at(lexer.Comma) {
			p.advance()
			continue
		}
		break
	}
	return params
}

// --- scalar expressions entry points live in expr.go ---
// --- statements and stages live in stmt.go ---
// --- type parsing lives in type.go ---
