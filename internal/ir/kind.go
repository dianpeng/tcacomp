package ir

// Kind identifies the operator of an IR node. The enumeration is versioned
// and stable (IR-03): new kinds introduced by later lowering phases append
// before KindEnd; existing constants are never renumbered or reordered, so
// graphs serialized under KindVersion 1 keep their meaning. The pass manager
// rejects any node with Kind >= KindEnd or Kind == KindInvalid.
type Kind uint32

// KindVersion is the IR node-kind schema version. Bump when kinds are added
// (append-only) so downstream consumers can gate on it.
const KindVersion = 1

const (
	KindInvalid     Kind = iota // unused slot; verifier and legality reject it
	KindIter                    // abstract: induction variable @m (name, bound dim, init, step)
	KindIterCluster             // abstract: compound iter(@m, @k, @n) header
	KindMap                     // abstract: map stage (operands, lambda, as-order)
	KindReduce                  // abstract: reduce stage (init, piped input, combiner, annotation)
	KindStore                   // abstract: store(target access, value)
	KindReturn                  // abstract: kernel return
	KindConst                   // scalar-expr: typed constant
	KindParam                   // scalar-expr: lambda parameter reference
	KindBinOp                   // scalar-expr: binary op (opcode table mirrors syntax.md §7.6)
	KindUnaryOp                 // scalar-expr: unary op
	KindSelect                  // scalar-expr: typed conditional ?:
	KindCast                    // scalar-expr: explicit cast to named type
	KindCall                    // scalar-expr: registered pure builtin call
	KindInputParam              // graph-owned kernel/lambda input parameter (SSA def; name is debug-only)
	KindEnd                     // sentinel: all kinds < KindEnd
)

// String returns the lowercase canonical name of the kind.
func (k Kind) String() string {
	switch k {
	case KindInvalid:
		return "invalid"
	case KindIter:
		return "iter"
	case KindIterCluster:
		return "itercluster"
	case KindMap:
		return "map"
	case KindReduce:
		return "reduce"
	case KindStore:
		return "store"
	case KindReturn:
		return "return"
	case KindConst:
		return "const"
	case KindParam:
		return "param"
	case KindBinOp:
		return "binop"
	case KindUnaryOp:
		return "unaryop"
	case KindSelect:
		return "select"
	case KindCast:
		return "cast"
	case KindCall:
		return "call"
	case KindInputParam:
		return "inputparam"
	case KindEnd:
		return "end"
	default:
		return "unknown"
	}
}

// Family classifies a Kind into its node family (D-15). FamilyLoop,
// FamilyTileOp, and FamilyCollective exist now — with no member kinds yet —
// so Phase 3-4 lowering kinds join via new constants without enum surgery
// (single-tier IR rule).
type Family int

const (
	FamilyAbstract Family = iota
	FamilyLoop
	FamilyTileOp
	FamilyCollective
	FamilyScalarExpr
)

// Family returns the family of the kind.
func (k Kind) Family() Family {
	switch k {
	case KindConst, KindParam, KindBinOp, KindUnaryOp, KindSelect, KindCast, KindCall:
		return FamilyScalarExpr
	default:
		// KindIter, KindIterCluster, KindMap, KindReduce, KindStore,
		// KindReturn, and the sentinel values all belong to the abstract
		// suite constructed in Phase 1.
		return FamilyAbstract
	}
}

// BinOpCode is the binary operator table, mirroring doc/syntax.md §7.6
// precedence levels from loosest to tightest binding.
type BinOpCode int

const (
	BinOpInvalid BinOpCode = iota
	BinOpLOr               // ||
	BinOpLAnd              // &&
	BinOpOr                // |
	BinOpXor               // ^
	BinOpAnd               // &
	BinOpEq                // ==
	BinOpNE                // !=
	BinOpLT                // <
	BinOpLE                // <=
	BinOpGT                // >
	BinOpGE                // >=
	BinOpShL               // <<
	BinOpShR               // >>
	BinOpAdd               // +
	BinOpSub               // -
	BinOpMul               // *
	BinOpDiv               // /
	BinOpMod               // %
)

// String returns the canonical source spelling of the operator.
func (op BinOpCode) String() string {
	switch op {
	case BinOpLOr:
		return "||"
	case BinOpLAnd:
		return "&&"
	case BinOpOr:
		return "|"
	case BinOpXor:
		return "^"
	case BinOpAnd:
		return "&"
	case BinOpEq:
		return "=="
	case BinOpNE:
		return "!="
	case BinOpLT:
		return "<"
	case BinOpLE:
		return "<="
	case BinOpGT:
		return ">"
	case BinOpGE:
		return ">="
	case BinOpShL:
		return "<<"
	case BinOpShR:
		return ">>"
	case BinOpAdd:
		return "+"
	case BinOpSub:
		return "-"
	case BinOpMul:
		return "*"
	case BinOpDiv:
		return "/"
	case BinOpMod:
		return "%"
	default:
		return "?"
	}
}

// UnaryOpCode is the unary operator table, mirroring doc/syntax.md §7.6.
type UnaryOpCode int

const (
	UnaryOpInvalid UnaryOpCode = iota
	UnaryOpPos                 // +
	UnaryOpNeg                 // -
	UnaryOpLNot                // !
	UnaryOpNot                 // ~
)

// String returns the canonical source spelling of the operator.
func (op UnaryOpCode) String() string {
	switch op {
	case UnaryOpPos:
		return "+"
	case UnaryOpNeg:
		return "-"
	case UnaryOpLNot:
		return "!"
	case UnaryOpNot:
		return "~"
	default:
		return "?"
	}
}
