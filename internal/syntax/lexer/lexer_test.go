package lexer

import (
	"fmt"
	"strings"
	"testing"
)

// scanAll lexes src to EOF, returning the tokens (EOF included) and the
// error messages the handler collected.
func scanAll(t *testing.T, src string) ([]Token, []string) {
	t.Helper()
	var errs []string
	s := NewScanner(&File{Name: "test.tca", Src: src}, func(p Pos, msg string) {
		errs = append(errs, fmt.Sprintf("%d:%d: %s", p.Line, p.Col, msg))
	})
	var toks []Token
	for {
		tok := s.Next()
		toks = append(toks, tok)
		if tok.Kind == EOF {
			break
		}
		if len(toks) > 10000 {
			t.Fatal("scanner did not reach EOF")
		}
	}
	return toks, errs
}

// kinds renders a token stream as "Kind:lit" pairs for compact comparison.
func kinds(toks []Token) string {
	var b strings.Builder
	for i, tok := range toks {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%s:%q", tok.Kind, tok.Lit)
	}
	return b.String()
}

// TestAppendixAHeader lexes the Appendix A prologue (design doc): the
// include, the @runtime keyed-const list, the @name annotation, and the
// kernel signature opening.
func TestAppendixAHeader(t *testing.T) {
	src := `@include "tca-common.tca";

@runtime(
  cxx_path="runtime/tca_runtime.h",
);

@name(cxx="MatMulScale")
kernel matmul_scale(
`
	toks, errs := scanAll(t, src)
	if len(errs) != 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}
	want := `@:"@" identifier:"include" string literal:"tca-common.tca" ;:";" ` +
		`@:"@" identifier:"runtime" (:"(" identifier:"cxx_path" =:"=" string literal:"runtime/tca_runtime.h" ,:"," ):")" ;:";" ` +
		`@:"@" identifier:"name" (:"(" identifier:"cxx" =:"=" string literal:"MatMulScale" ):")" ` +
		`kernel:"kernel" identifier:"matmul_scale" (:"(" EOF:""`
	if got := kinds(toks); got != want {
		t.Fatalf("token stream mismatch\ngot:  %s\nwant: %s", got, want)
	}
}

// TestExpressionFragment exercises every precedence-level operator (design
// §2.7), both comment forms, and every literal family in one stream.
func TestExpressionFragment(t *testing.T) {
	src := `// line comment
let x = -a + +b * !c - ~d / e % f /* block
comment */ | g ^ h & i << 2u >> 1 == j != k < l <= m > n >= o && p || q ? r : s;
let big = 3000000000u;
let pi = 2.5f * 1e2f;
`
	toks, errs := scanAll(t, src)
	if len(errs) != 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}
	want := `let:"let" identifier:"x" =:"=" -:"-" identifier:"a" +:"+" +:"+" identifier:"b" *:"*" !:"!" identifier:"c" ` +
		`-:"-" ~:"~" identifier:"d" /:"/" identifier:"e" %:"%" identifier:"f" ` +
		`|:"|" identifier:"g" ^:"^" identifier:"h" &:"&" identifier:"i" <<:"<<" uint literal:"2u" >>:">>" int literal:"1" ` +
		`==:"==" identifier:"j" !=:"!=" identifier:"k" <:"<" identifier:"l" <=:"<=" identifier:"m" >:">" identifier:"n" >=:">=" identifier:"o" ` +
		`&&:"&&" identifier:"p" ||:"||" identifier:"q" ?:"?" identifier:"r" ::":" identifier:"s" ;:";" ` +
		`let:"let" identifier:"big" =:"=" uint literal:"3000000000u" ;:";" ` +
		`let:"let" identifier:"pi" =:"=" float literal:"2.5f" *:"*" float literal:"1e2f" ;:";" EOF:""`
	if got := kinds(toks); got != want {
		t.Fatalf("token stream mismatch\ngot:  %s\nwant: %s", got, want)
	}
}

// TestPlaceholderVsModulo pins the §2.2/§7.3 rule: '%' + digits is ONE
// placeholder token; a bare '%' is the modulo operator.
func TestPlaceholderVsModulo(t *testing.T) {
	src := `on(acc: f32, %0: f32, %3: f32) | map on() [s] (%0): %0 % 2`
	toks, errs := scanAll(t, src)
	if len(errs) != 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}
	var placeholders []Token
	var percents int
	for _, tok := range toks {
		switch tok.Kind {
		case Placeholder:
			placeholders = append(placeholders, tok)
		case Percent:
			percents++
		}
	}
	if len(placeholders) != 4 {
		t.Fatalf("want 4 placeholder tokens, got %d (%s)", len(placeholders), kinds(toks))
	}
	for i, want := range []struct {
		lit string
		val int
	}{
		{"%0", 0}, {"%3", 3}, {"%0", 0}, {"%0", 0},
	} {
		if placeholders[i].Lit != want.lit || placeholders[i].Val != want.val {
			t.Errorf("placeholder %d = %q/%v, want %q/%d", i, placeholders[i].Lit, placeholders[i].Val, want.lit, want.val)
		}
	}
	if percents != 1 {
		t.Errorf("want exactly 1 modulo Percent, got %d (%s)", percents, kinds(toks))
	}
}

// TestKeywordAndLiteralValues checks the parsed Val cells and keyword
// resolution (design §2.1).
func TestKeywordAndLiteralValues(t *testing.T) {
	src := `kernel let return dimvar map on as reduce init true false inf_f32 bool f32 f16 bf16 i8 i16 i32 u8 u16 u32 7 7u 0.5f`
	toks, errs := scanAll(t, src)
	if len(errs) != 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}
	wantKinds := []Kind{Kernel, Let, Return, DimVar, Map, On, As, Reduce, Init, True, False, InfF32,
		TBool, TF32, TF16, TBF16, TI8, TI16, TI32, TU8, TU16, TU32, IntLit, UintLit, FloatLit}
	if len(toks) != len(wantKinds)+1 { // + EOF
		t.Fatalf("want %d tokens, got %d (%s)", len(wantKinds)+1, len(toks), kinds(toks))
	}
	for i, want := range wantKinds {
		if toks[i].Kind != want {
			t.Errorf("token %d = %s, want %s (%s)", i, toks[i].Kind, want, kinds(toks))
		}
	}
	var lits []Token
	for _, tok := range toks {
		switch tok.Kind {
		case IntLit, UintLit, FloatLit:
			lits = append(lits, tok)
		}
	}
	for i, want := range []struct {
		lit string
		val any
	}{
		{"7", int32(7)}, {"7u", uint32(7)}, {"0.5f", float32(0.5)},
	} {
		tok := lits[i]
		if tok.Lit != want.lit || tok.Val != want.val {
			t.Errorf("literal %q = %v, want %q/%v", tok.Lit, tok.Val, want.lit, want.val)
		}
	}
	if toks[11].Kind != InfF32 {
		t.Errorf("inf_f32 must lex as the predefined-constant keyword, got %s", toks[11].Kind)
	}
}

// TestNegativeLiterals pins the fail-fast lexical errors (design §4.8):
// each malformed input reports exactly one position-attributed error.
func TestNegativeLiterals(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string // substring of the error message
		line int
		col  int
	}{
		{"float suffix required", "let x = 1.5;", "requires the f/F suffix", 1, 9},
		{"exponent suffix required", "let x = 1e3;", "requires the f/F suffix", 1, 9},
		{"overflow i32", "let x = 3000000000;", "does not fit i32", 1, 9},
		{"unterminated string", `let x = "abc;`, "string literal not terminated", 1, 9},
		{"unterminated block comment", "let x = 1; /* oops", "block comment not terminated", 1, 12},
		{"malformed trailing letters", "let x = 12ab;", "malformed numeric literal", 1, 9},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, errs := scanAll(t, tc.src)
			if len(errs) != 1 {
				t.Fatalf("want exactly 1 error, got %d: %v", len(errs), errs)
			}
			if !strings.Contains(errs[0], tc.want) {
				t.Fatalf("error %q does not mention %q", errs[0], tc.want)
			}
			want := fmt.Sprintf("%d:%d:", tc.line, tc.col)
			if !strings.HasPrefix(errs[0], want) {
				t.Fatalf("error position = %q, want prefix %q", errs[0], want)
			}
		})
	}
}

// TestPositions pins 1-based line/column attribution across lines.
func TestPositions(t *testing.T) {
	src := "kernel k(\n  lhs: tensor<dtype=f32>,\n) -> f32"
	toks, errs := scanAll(t, src)
	if len(errs) != 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}
	var f32tok *Token
	for i := range toks {
		if toks[i].Kind == TF32 {
			f32tok = &toks[i]
			break // the dtype=f32 occurrence on line 2
		}
	}
	if f32tok == nil {
		t.Fatal("no f32 token found")
	}
	if f32tok.Pos.Line != 2 || f32tok.Pos.Col != 21 {
		t.Errorf("f32 at %d:%d, want 2:21", f32tok.Pos.Line, f32tok.Pos.Col)
	}
}
