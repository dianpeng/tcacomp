// Package lexer is the TCA frontend's hand-rolled scanner (doc/syntax-design.md
// §2.1–§2.2, CLAUDE.md decision (a)): a scanner struct over the source string
// with next()/peek() rune reading, position tracking from the start, and an
// error-handler callback — the go/scanner STRUCTURE ported to the TCA token
// set (text/scanner and go/scanner themselves are never imported; the grammar
// is C++-style, not Go-literal-oriented).
package lexer

// Kind is the TCA token kind (design §2.1–§2.2). One constant per token
// family: literal kinds, keyword kinds (reserved words and element types),
// the placeholder family, punctuation/operators, and Illegal/EOF sentinels.
type Kind int

const (
	Illegal Kind = iota // a malformed token (the scan stopped cleanly at it)
	EOF                 // end of input
	Ident               // identifier
	IntLit              // decimal intLit  → i32 (design §2.1)
	UintLit             // decimal uintLit → u32 ('u'/'U' suffix)
	FloatLit            // floatLit → f32 ('f'/'F' suffix REQUIRED)
	StringLit           // string literal (annotations only)
	Placeholder         // '%' digits — the §7.3 pipe-slot family (v0: %0 only)

	// Reserved keywords (design §2.1).
	Kernel  // kernel
	Let     // let
	Return  // return
	DimVar  // dimvar
	Map     // map
	On      // on
	As      // as
	Reduce  // reduce
	Init    // init
	True    // true
	False   // false
	InfF32  // inf_f32 — predefined f32 constant (reserved identifier)

	// Element-type keywords (reserved; double as cast operators, §2.7).
	TBool // bool
	TF32  // f32
	TF16  // f16
	TBF16 // bf16
	TI8   // i8
	TI16  // i16
	TI32  // i32
	TU8   // u8
	TU16  // u16
	TU32  // u32

	// Punctuation and operators (design §2.2).
	At       // @
	LParen   // (
	RParen   // )
	LBracket // [
	RBracket // ]
	LBrace   // {
	RBrace   // }
	Lt       // <
	Gt       // >
	Comma    // ,
	Semi     // ;
	Colon    // :
	Assign   // =
	Question // ?
	Pipe     // |
	Amp      // &
	Caret    // ^
	Tilde    // ~
	Bang     // !
	Plus     // +
	Minus    // -
	Star     // *
	Slash    // /
	Percent  // %
	Arrow    // ->
	EqEq     // ==
	Ne       // !=
	Le       // <=
	Ge       // >=
	AndAnd   // &&
	OrOr     // ||
	Shl      // <<
	Shr      // >>
)

// keywords is the reserved-word table (design §2.1): statement/stage
// keywords, the boolean literals, the predefined constant, and the
// element-type keywords. Annotation/attribute keys (dtype, shape, cxx,
// cxx_path) are CONTEXTUAL identifiers and deliberately absent. The table is
// a plain lookup map, never iterated.
var keywords = map[string]Kind{
	"kernel":   Kernel,
	"let":      Let,
	"return":   Return,
	"dimvar":   DimVar,
	"map":      Map,
	"on":       On,
	"as":       As,
	"reduce":   Reduce,
	"init":     Init,
	"true":     True,
	"false":    False,
	"inf_f32":  InfF32,
	"bool":     TBool,
	"f32":      TF32,
	"f16":      TF16,
	"bf16":     TBF16,
	"i8":       TI8,
	"i16":      TI16,
	"i32":      TI32,
	"u8":       TU8,
	"u16":      TU16,
	"u32":      TU32,
}

// keywordText is keywords' inverse: Kind → source spelling, for diagnostics
// and tests. Only keyword kinds appear.
var keywordText = map[Kind]string{
	Kernel: "kernel", Let: "let", Return: "return", DimVar: "dimvar",
	Map: "map", On: "on", As: "as", Reduce: "reduce", Init: "init",
	True: "true", False: "false", InfF32: "inf_f32",
	TBool: "bool", TF32: "f32", TF16: "f16", TBF16: "bf16",
	TI8: "i8", TI16: "i16", TI32: "i32", TU8: "u8", TU16: "u16", TU32: "u32",
}

// IsKeyword reports whether k is a reserved-word kind.
func IsKeyword(k Kind) bool {
	_, ok := keywordText[k]
	return ok
}

// IsElemType reports whether k is an element-type keyword kind (the cast
// operators of design §2.7).
func IsElemType(k Kind) bool {
	switch k {
	case TBool, TF32, TF16, TBF16, TI8, TI16, TI32, TU8, TU16, TU32:
		return true
	}
	return false
}

// String returns the canonical source spelling for punctuation and keyword
// kinds (diagnostics/tests); literal kinds render their family name.
func (k Kind) String() string {
	if s, ok := keywordText[k]; ok {
		return s
	}
	switch k {
	case Illegal:
		return "Illegal"
	case EOF:
		return "EOF"
	case Ident:
		return "identifier"
	case IntLit:
		return "int literal"
	case UintLit:
		return "uint literal"
	case FloatLit:
		return "float literal"
	case StringLit:
		return "string literal"
	case Placeholder:
		return "%N placeholder"
	case At:
		return "@"
	case LParen:
		return "("
	case RParen:
		return ")"
	case LBracket:
		return "["
	case RBracket:
		return "]"
	case LBrace:
		return "{"
	case RBrace:
		return "}"
	case Lt:
		return "<"
	case Gt:
		return ">"
	case Comma:
		return ","
	case Semi:
		return ";"
	case Colon:
		return ":"
	case Assign:
		return "="
	case Question:
		return "?"
	case Pipe:
		return "|"
	case Amp:
		return "&"
	case Caret:
		return "^"
	case Tilde:
		return "~"
	case Bang:
		return "!"
	case Plus:
		return "+"
	case Minus:
		return "-"
	case Star:
		return "*"
	case Slash:
		return "/"
	case Percent:
		return "%"
	case Arrow:
		return "->"
	case EqEq:
		return "=="
	case Ne:
		return "!="
	case Le:
		return "<="
	case Ge:
		return ">="
	case AndAnd:
		return "&&"
	case OrOr:
		return "||"
	case Shl:
		return "<<"
	case Shr:
		return ">>"
	}
	return "?"
}

// File is one source file the scanner reads. The module loader (package
// syntax) owns file creation; the scanner only reads.
type File struct {
	Name string // path as given to the loader (diagnostic attribution)
	Src  string // full source text
}

// Pos is a source position (design §3: every token and AST node carries
// one): a 1-based line/column pair plus the 0-based byte offset into the
// file, with the file itself for attribution.
type Pos struct {
	File   *File
	Offset int // 0-based byte offset
	Line   int // 1-based
	Col    int // 1-based
}

// IsValid reports whether p carries any position information.
func (p Pos) IsValid() bool { return p.Line > 0 }

// Token is one lexed token: its kind, literal text (the exact source
// spelling), parsed value (literals only: int32/uint32/float32/string/bool;
// Placeholder carries the slot index as int), and position.
type Token struct {
	Kind Kind
	Lit  string // exact source spelling ("" for EOF)
	Val  any    // parsed value; nil for non-literals
	Pos  Pos
}
