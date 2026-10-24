package lexer

import (
	"fmt"
	"strconv"
	"unicode"
	"unicode/utf8"
)

// Scanner is the hand-rolled TCA scanner: the go/scanner STRUCTURE (a
// scanner struct over the source string, next()/peek() rune reading,
// position tracking from the start, error-handler callback) with the TCA
// token set (design §2). Malformed input reports a position-attributed
// error through the handler and the scan of that token stops cleanly at the
// offending point; recovery past it is best-effort so one bad token cannot
// wedge the scan (design §4.8: fail fast, position-attributed).
type Scanner struct {
	file *File
	src  string

	ch       rune // current character (utf8.RuneError at EOF/end)
	offset   int  // byte offset of ch in src
	rdOffset int  // byte offset just past ch (the read-ahead point)
	line     int  // 1-based line of ch
	col      int  // 1-based column of ch

	errCount int
	err      func(Pos, string) // error-handler callback (design §2.1)
}

// NewScanner creates a scanner over f; errHandler (may be nil) receives one
// position-attributed message per lexical error.
func NewScanner(f *File, errHandler func(Pos, string)) *Scanner {
	s := &Scanner{file: f, src: f.Src, err: errHandler, line: 1, col: 1}
	s.next() // prime the lookahead
	return s
}

// ErrCount returns the number of lexical errors reported so far.
func (s *Scanner) ErrCount() int { return s.errCount }

// errorf reports one error at the CURRENT character position.
func (s *Scanner) errorf(format string, args ...any) {
	s.errCount++
	if s.err != nil {
		s.err(Pos{File: s.file, Offset: s.offset, Line: s.line, Col: s.col}, fmt.Sprintf(format, args...))
	}
}

// errorAt reports one error at an explicit position.
func (s *Scanner) errorAt(p Pos, format string, args ...any) {
	s.errCount++
	if s.err != nil {
		s.err(p, fmt.Sprintf(format, args...))
	}
}

// next advances to the next character, tracking line/column from the start.
// At end of input ch becomes utf8.RuneError with offset == len(src).
func (s *Scanner) next() {
	if s.rdOffset >= len(s.src) {
		s.offset = len(s.src)
		s.ch = utf8.RuneError
		return
	}
	s.offset = s.rdOffset
	if s.ch == '\n' {
		s.line++
		s.col = 1
	} else {
		s.col++
	}
	r, w := utf8.DecodeRuneInString(s.src[s.rdOffset:])
	s.rdOffset += w
	s.ch = r
}

// peek returns the character after the current one without consuming it
// (utf8.RuneError at end of input).
func (s *Scanner) peek() rune {
	if s.rdOffset >= len(s.src) {
		return utf8.RuneError
	}
	r, _ := utf8.DecodeRuneInString(s.src[s.rdOffset:])
	return r
}

// posAt builds a Pos for the token starting at byte offset off.
func (s *Scanner) posAt(off int) Pos {
	line, col := 1, 1
	for i := 0; i < off && i < len(s.src); i++ {
		if s.src[i] == '\n' {
			line++
			col = 1
		} else {
			col++
		}
	}
	return Pos{File: s.file, Offset: off, Line: line, Col: col}
}

// Next scans and returns the next token, skipping whitespace and comments.
// At end of input it returns an EOF token on every call.
func (s *Scanner) Next() Token {
	for {
		switch {
		case s.ch == ' ' || s.ch == '\t' || s.ch == '\r' || s.ch == '\n':
			s.next()
			continue
		case s.ch == '/' && (s.peek() == '/' || s.peek() == '*'):
			s.scanComment()
			continue
		}
		break
	}
	off := s.offset
	tok := s.switchToken()
	tok.Pos = s.posAt(off)
	return tok
}

// switchToken dispatches on the current character. Every path returns a
// Token with Kind/Lit/Val set; Pos is stamped by Next.
func (s *Scanner) switchToken() Token {
	switch ch := s.ch; {
	case ch == utf8.RuneError && s.offset >= len(s.src):
		return Token{Kind: EOF}
	case isLetter(ch):
		return s.scanIdentifier()
	case isDigit(ch):
		return s.scanNumber()
	case ch == '"':
		return s.scanString()
	case ch == '%':
		if isDigit(s.peek()) {
			return s.scanPlaceholder()
		}
		s.next()
		return Token{Kind: Percent, Lit: "%"}
	default:
		return s.scanOperator()
	}
}

// scanIdentifier scans an identifier and resolves it against the keyword
// table (design §2.1); identifiers not in the table stay Ident.
func (s *Scanner) scanIdentifier() Token {
	off := s.offset
	for isLetter(s.ch) || isDigit(s.ch) {
		s.next()
	}
	lit := s.src[off:s.offset]
	if k, ok := keywords[lit]; ok {
		if k == True {
			return Token{Kind: True, Lit: lit, Val: true}
		}
		if k == False {
			return Token{Kind: False, Lit: lit, Val: false}
		}
		return Token{Kind: k, Lit: lit}
	}
	return Token{Kind: Ident, Lit: lit}
}

// scanNumber scans intLit/uintLit/floatLit per design §2.1. Numeric parsing
// is strconv-only with explicit base/size; a malformed literal reports one
// position-attributed error and the scan stops cleanly at the bad point.
func (s *Scanner) scanNumber() Token {
	start := s.posAt(s.offset)
	off := s.offset
	isFloat := false
	for isDigit(s.ch) {
		s.next()
	}
	if s.ch == '.' {
		isFloat = true
		s.next()
		if !isDigit(s.ch) {
			s.errorAt(start, "malformed float literal: expected digits after '.'")
		}
		for isDigit(s.ch) {
			s.next()
		}
	}
	if s.ch == 'e' || s.ch == 'E' {
		isFloat = true
		s.next()
		if s.ch == '+' || s.ch == '-' {
			s.next()
		}
		if !isDigit(s.ch) {
			s.errorAt(start, "malformed float literal: exponent has no digits")
		}
		for isDigit(s.ch) {
			s.next()
		}
	}
	lit := s.src[off:s.offset]
	if isFloat {
		if s.ch != 'f' && s.ch != 'F' {
			s.errorAt(start, "float literal %q requires the f/F suffix (design §2.1)", lit)
			return Token{Kind: Illegal, Lit: lit}
		}
		s.next()
		lit = s.src[off:s.offset]
		if isLetter(s.ch) || isDigit(s.ch) { // "1.5fx": one error, scan continues
			for isLetter(s.ch) || isDigit(s.ch) {
				s.next()
			}
			s.errorAt(start, "malformed numeric literal %q", s.src[off:s.offset])
			return Token{Kind: Illegal, Lit: s.src[off:s.offset]}
		}
		v, err := strconv.ParseFloat(lit[:len(lit)-1], 32)
		if err != nil {
			s.errorAt(start, "malformed float literal %q: %v", lit, err)
			return Token{Kind: Illegal, Lit: lit}
		}
		return Token{Kind: FloatLit, Lit: lit, Val: float32(v)}
	}
	if s.ch == 'u' || s.ch == 'U' {
		s.next()
		lit = s.src[off:s.offset]
		if isLetter(s.ch) || isDigit(s.ch) {
			for isLetter(s.ch) || isDigit(s.ch) {
				s.next()
			}
			s.errorAt(start, "malformed numeric literal %q", s.src[off:s.offset])
			return Token{Kind: Illegal, Lit: s.src[off:s.offset]}
		}
		v, err := strconv.ParseUint(lit[:len(lit)-1], 10, 32)
		if err != nil {
			s.errorAt(start, "uint literal %q does not fit u32", lit)
			return Token{Kind: Illegal, Lit: lit}
		}
		return Token{Kind: UintLit, Lit: lit, Val: uint32(v)}
	}
	if isLetter(s.ch) { // "123abc": one error, tail consumed, scan continues
		for isLetter(s.ch) || isDigit(s.ch) {
			s.next()
		}
		s.errorAt(start, "malformed numeric literal %q", s.src[off:s.offset])
		return Token{Kind: Illegal, Lit: s.src[off:s.offset]}
	}
	v, err := strconv.ParseInt(lit, 10, 32)
	if err != nil {
		s.errorAt(start, "int literal %q does not fit i32", lit)
		return Token{Kind: Illegal, Lit: lit}
	}
	return Token{Kind: IntLit, Lit: lit, Val: int32(v)}
}

// scanString scans a string literal: '"' then any char except '"'
// (design §2.1 — no escape sequences). A newline or EOF before the closing
// quote reports one error and terminates the literal.
func (s *Scanner) scanString() Token {
	start := s.posAt(s.offset)
	s.next() // consume '"'
	off := s.offset
	for s.ch != '"' {
		if s.ch == '\n' || (s.ch == utf8.RuneError && s.offset >= len(s.src)) {
			s.errorAt(start, "string literal not terminated")
			return Token{Kind: Illegal, Lit: s.src[off:s.offset]}
		}
		s.next()
	}
	lit := s.src[off:s.offset]
	s.next() // consume '"'
	return Token{Kind: StringLit, Lit: lit, Val: lit}
}

// scanPlaceholder scans '%' digits as one placeholder token (design §2.2,
// §7.3); the parser restricts where it may appear (v0: %0 only).
func (s *Scanner) scanPlaceholder() Token {
	off := s.offset
	s.next() // consume '%'
	for isDigit(s.ch) {
		s.next()
	}
	lit := s.src[off:s.offset]
	n, err := strconv.Atoi(lit[1:])
	if err != nil {
		s.errorf("malformed placeholder %q", lit)
		return Token{Kind: Illegal, Lit: lit}
	}
	return Token{Kind: Placeholder, Lit: lit, Val: n}
}

// scanComment consumes one '//' line comment or one '/* */' block comment.
// An unterminated block comment reports one error at its start.
func (s *Scanner) scanComment() {
	if s.peek() == '/' {
		for s.ch != '\n' && !(s.ch == utf8.RuneError && s.offset >= len(s.src)) {
			s.next()
		}
		return
	}
	start := s.posAt(s.offset)
	s.next() // '/'
	s.next() // '*'
	for {
		if s.ch == utf8.RuneError && s.offset >= len(s.src) {
			s.errorAt(start, "block comment not terminated")
			return
		}
		if s.ch == '*' && s.peek() == '/' {
			s.next()
			s.next()
			return
		}
		s.next()
	}
}

// scanOperator scans the punctuation/operator set (design §2.2), resolving
// the two-character forms (-> == != <= >= && || << >>) by one rune of
// lookahead.
func (s *Scanner) scanOperator() Token {
	ch := s.ch
	start := s.posAt(s.offset)
	s.next()
	two := func(k Kind, spelling string) Token {
		s.next()
		return Token{Kind: k, Lit: spelling}
	}
	switch ch {
	case '@':
		return Token{Kind: At, Lit: "@"}
	case '(':
		return Token{Kind: LParen, Lit: "("}
	case ')':
		return Token{Kind: RParen, Lit: ")"}
	case '[':
		return Token{Kind: LBracket, Lit: "["}
	case ']':
		return Token{Kind: RBracket, Lit: "]"}
	case '{':
		return Token{Kind: LBrace, Lit: "{"}
	case '}':
		return Token{Kind: RBrace, Lit: "}"}
	case ',':
		return Token{Kind: Comma, Lit: ","}
	case ';':
		return Token{Kind: Semi, Lit: ";"}
	case ':':
		return Token{Kind: Colon, Lit: ":"}
	case '?':
		return Token{Kind: Question, Lit: "?"}
	case '~':
		return Token{Kind: Tilde, Lit: "~"}
	case '<':
		if s.ch == '<' {
			return two(Shl, "<<")
		}
		if s.ch == '=' {
			return two(Le, "<=")
		}
		return Token{Kind: Lt, Lit: "<"}
	case '>':
		if s.ch == '>' {
			return two(Shr, ">>")
		}
		if s.ch == '=' {
			return two(Ge, ">=")
		}
		return Token{Kind: Gt, Lit: ">"}
	case '=':
		if s.ch == '=' {
			return two(EqEq, "==")
		}
		return Token{Kind: Assign, Lit: "="}
	case '!':
		if s.ch == '=' {
			return two(Ne, "!=")
		}
		return Token{Kind: Bang, Lit: "!"}
	case '-':
		if s.ch == '>' {
			return two(Arrow, "->")
		}
		return Token{Kind: Minus, Lit: "-"}
	case '&':
		if s.ch == '&' {
			return two(AndAnd, "&&")
		}
		return Token{Kind: Amp, Lit: "&"}
	case '|':
		if s.ch == '|' {
			return two(OrOr, "||")
		}
		return Token{Kind: Pipe, Lit: "|"}
	case '+':
		return Token{Kind: Plus, Lit: "+"}
	case '*':
		return Token{Kind: Star, Lit: "*"}
	case '/':
		return Token{Kind: Slash, Lit: "/"}
	case '^':
		return Token{Kind: Caret, Lit: "^"}
	default:
		s.errorAt(start, "unexpected character %q", ch)
		return Token{Kind: Illegal, Lit: string(ch)}
	}
}

func isLetter(ch rune) bool {
	return ch == '_' || unicode.IsLetter(ch)
}

func isDigit(ch rune) bool { return unicode.IsDigit(ch) && ch < utf8.RuneSelf }
