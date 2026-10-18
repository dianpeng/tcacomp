package ir

// catalog.go is the dedicated listing file (R4e): one row per IR node kind
// recording its tier (HIR/LIR, D-3), family, operand arity, and a short doc.
// kind.go owns the enum itself; catalog.go owns every other per-kind table.
// The rows are ordered by Kind value so catalog[k-1] describes k (catalog[0]
// is KindMap — KindInvalid and KindEnd are sentinels and carry no row).
//
// Invariants tested in catalog_test.go:
//
//	len(AllKinds()) == int(KindEnd)-1           (every real kind has a row)
//	HIRKinds() are exactly the kinds < KindForLoop (tier boundary is the enum boundary)
//	every row's arity is -1 (variadic) or the fixed slot count its node allocates

// KindTier discriminates the HIR/LIR tiers of the node-kind taxonomy (D-3).
type KindTier int

const (
	TierHIR KindTier = iota // abstract kernel graph the builder produces
	TierLIR                 // lowered suite introduced by optimization passes
)

// kindEntry is one row of the node catalog.
type kindEntry struct {
	kind   Kind
	tier   KindTier
	family Family
	arity  int // fixed operand-slot count; -1 = variadic (slots allocated from operands)
	doc    string
}

// catalog lists every known IR node kind in enum order.
var catalog = []kindEntry{
	// --- HIR tier ---
	{KindMap, TierHIR, FamilyAbstract, -1, "lambda map stage: operands..., lambda, as-order"},
	{KindReduce, TierHIR, FamilyAbstract, 2, "lambda reduce stage: init, piped input, combiner lambda"},
	{KindMemLoad, TierHIR, FamilyMemory, -1, "GMEM load: tensor operand + axes; chains via ImpactBy"},
	{KindMemStore, TierHIR, FamilyMemory, 1, "GMEM store: (target access payload, value); HIR stage store"},
	{KindMemNew, TierHIR, FamilyMemory, 1, "GMEM allocation: (extent value)"},
	{KindIter, TierHIR, FamilyLoop, 2, "induction variable @m: (dim/bound payload), init, step"},
	{KindIterCluster, TierHIR, FamilyLoop, 1, "compound iter(@m, @k, @n) header: (dims payload), body"},
	{KindConst, TierHIR, FamilyScalarExpr, 0, "typed scalar constant"},
	{KindParam, TierHIR, FamilyScalarExpr, 0, "lambda parameter reference"},
	{KindBinOp, TierHIR, FamilyScalarExpr, 2, "binary scalar op (opcode table mirrors syntax.md §7.6)"},
	{KindUnaryOp, TierHIR, FamilyScalarExpr, 1, "unary scalar op"},
	{KindSelect, TierHIR, FamilyScalarExpr, 3, "typed conditional ?: (cond, then, else)"},
	{KindCast, TierHIR, FamilyScalarExpr, 1, "explicit cast to a named scalar type"},
	{KindCall, TierHIR, FamilyScalarExpr, -1, "registered pure builtin call: (builtin, args...)"},
	{KindInputParam, TierHIR, FamilyAbstract, 0, "graph-owned kernel/lambda input parameter (named)"},
	{KindReturn, TierHIR, FamilyAbstract, -1, "end node: kernel return (values...); graph's single end"},

	// --- LIR tier ---
	{KindForLoop, TierLIR, FamilyLoop, 3, "lowered loop nest: (iv dimvar, bound bdimvar, body)"},
	{KindDimVar, TierLIR, FamilyAbstract, 2, "lowered induction variable: (init, step) value object"},
	{KindBoundedDimVar, TierLIR, FamilyAbstract, 2, "dimvar bound to a max: ((iv tuple), max)"},
	{KindMemLoadSMem, TierLIR, FamilyMemory, -1, "SMEM load: (src smem buffer, bivs...)"},
	{KindMemStoreSMem, TierLIR, FamilyMemory, -1, "SMEM store: (value, dst smem buffer, bivs...)"},
	{KindMemNewSMem, TierLIR, FamilyMemory, 1, "SMEM allocation: (extent)"},
	{KindFragLoad, TierLIR, FamilyMemory, -1, "fragment load: (src frag buffer, bivs...)"},
	{KindFragStore, TierLIR, FamilyMemory, -1, "fragment store: (value, dst frag buffer, bivs...)"},
	{KindFragNew, TierLIR, FamilyMemory, 1, "fragment allocation: (extent)"},
}

// HIRKinds returns the HIR-tier kinds in enum order.
func HIRKinds() []Kind {
	out := make([]Kind, 0, len(catalog))
	for _, e := range catalog {
		if e.tier == TierHIR {
			out = append(out, e.kind)
		}
	}
	return out
}

// LIRKinds returns the LIR-tier kinds in enum order.
func LIRKinds() []Kind {
	out := make([]Kind, 0, len(catalog))
	for _, e := range catalog {
		if e.tier == TierLIR {
			out = append(out, e.kind)
		}
	}
	return out
}

// AllKinds returns every real node kind in enum order (KindInvalid and
// KindEnd excluded). Invariant: len(AllKinds()) == int(KindEnd)-1.
func AllKinds() []Kind {
	out := make([]Kind, 0, len(catalog))
	for _, e := range catalog {
		out = append(out, e.kind)
	}
	return out
}

// Tier returns the HIR/LIR tier of the kind. The sentinel values KindInvalid
// and KindEnd belong to no tier and report TierHIR by convention — callers
// that care must range-check with KindInvalid < k < KindEnd first.
func (k Kind) Tier() KindTier {
	if k > KindInvalid && k < KindEnd {
		return catalog[k-1].tier
	}
	return TierHIR
}

// fixedArity returns the fixed operand-slot count of the kind, or -1 when the
// kind is variadic (slots are allocated from the operand list at construction
// time). Unknown/out-of-range kinds report -1.
func fixedArity(k Kind) int {
	if k > KindInvalid && k < KindEnd {
		return catalog[k-1].arity
	}
	return -1
}
