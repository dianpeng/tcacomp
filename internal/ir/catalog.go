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
	{KindMap, TierHIR, FamilyAbstract, -1, "lambda map stage: value operands..., trailing As?, trailing lambda node"},
	{KindReduce, TierHIR, FamilyAbstract, -1, "lambda reduce stage: piped, KeyedAttribute(init), trailing As?, trailing lambda node"},
	{KindMemLoad, TierHIR, FamilyMemory, -1, "GMEM load: tensor operand + axes; chains via ImpactBy"},
	{KindMemStore, TierHIR, FamilyMemory, -1, "GMEM store: (value, tensor, axes...) — the HIR stage store"},
	{KindMemNew, TierHIR, FamilyMemory, 1, "GMEM allocation: (extent value)"},
	{KindIter, TierHIR, FamilyLoop, -1, "loop header: (biv, parent?) — the BoundedDimVar value object, plus the parental iter only when nested"},
	{KindIterCluster, TierHIR, FamilyLoop, -1, "compound iter(@m, @k, @n) header: (dims..., inputs...) — dims lead the input list"},
	{KindConst, TierHIR, FamilyScalarExpr, 0, "typed scalar constant"},
	{KindParam, TierHIR, FamilyScalarExpr, 1, "lambda/kernel parameter reference: (the InputParam)"},
	{KindBinOp, TierHIR, FamilyScalarExpr, 2, "binary scalar op (opcode table mirrors syntax.md §7.6)"},
	{KindUnaryOp, TierHIR, FamilyScalarExpr, 1, "unary scalar op"},
	{KindSelect, TierHIR, FamilyScalarExpr, 3, "typed conditional ?: (cond, then, else)"},
	{KindCast, TierHIR, FamilyScalarExpr, 1, "explicit cast to a named scalar type"},
	{KindCall, TierHIR, FamilyScalarExpr, -1, "registered pure builtin call: (builtin, args...)"},
	{KindInputParam, TierHIR, FamilyAbstract, 0, "graph-owned kernel/lambda input parameter (named)"},
	{KindReturn, TierHIR, FamilyAbstract, -1, "end node: kernel return (values...); graph's single end"},
	{KindKernelMeta, TierHIR, FamilyAbstract, 0, "graph-owned naming metadata: kernel/launcher symbol string payloads; anchored in the Prototype block (quick-261023-cgq)"},
	{KindIterClusterProject, TierHIR, FamilyAbstract, 1, "IterCluster input slot i bound to a Map's param i: (cluster), idx payload"},
	{KindKeyedAttribute, TierHIR, FamilyAbstract, 1, "named attribute riding an Input list: (value), key payload"},
	{KindAs, TierHIR, FamilyAbstract, -1, "as-order clause: dim params in order; trailing on Map/Reduce value inputs"},
	{KindLambda, TierHIR, FamilyAbstract, -1, "lambda expression: (param declarations..., body); map/reduce carry it as their trailing input"},

	// --- LIR tier ---
	{KindForLoop, TierLIR, FamilyLoop, -1, "lowered loop nest: (iv dimvar, bound bdimvar, body, KeyedAttribute annotations... (M1, quick-261021-jay))"},
	{KindDimVar, TierLIR, FamilyAbstract, 3, "lowered induction variable: (dim, init, step) value object"},
	{KindBoundedDimVar, TierLIR, FamilyAbstract, 2, "dimvar bound to a max: ((iv tuple), max)"},
	{KindMemLoadSMem, TierLIR, FamilyMemory, -1, "SMEM load: (src smem buffer, bivs...)"},
	{KindMemStoreSMem, TierLIR, FamilyMemory, -1, "SMEM store: (value, dst smem buffer, bivs...)"},
	{KindMemNewSMem, TierLIR, FamilyMemory, 1, "SMEM allocation: (extent)"},
	{KindFragLoad, TierLIR, FamilyMemory, -1, "fragment load: (src frag buffer, bivs...)"},
	{KindFragStore, TierLIR, FamilyMemory, -1, "fragment store: (value, dst frag buffer, bivs...)"},
	{KindFragNew, TierLIR, FamilyMemory, 1, "fragment allocation: (extent)"},
	{KindTensorDimOf, TierLIR, FamilyAbstract, 2, "shape upper bound: (tensor param, TU32 axis const)"},
	{KindInductionVariable, TierLIR, FamilyAbstract, 1, "u32 current-iteration value of an iter: (iter)"},
	{KindRuntimeGridQuery, TierLIR, FamilyAbstract, 1, "grid-map query: (DoSoC const u32); string field payload index|dimension; result u32. LIR, exists only between tile-lower and final-lower (renamed from IntrinsicGridMap, LR-6); Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeThreadQuery, TierLIR, FamilyAbstract, 1, "thread-map query: (DoSoC const u32); string field payload index|dimension; result u32. LIR, exists only between tile-lower and final-lower (renamed from IntrinsicThreadMap, LR-6); Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeKernelInfo, TierLIR, FamilyRuntime, 0, "kernel information object: no operands; source of total_cta/total_thread_per_cta attribute reads. final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeGridMapping, TierLIR, FamilyRuntime, -1, "per-LNT grid mapping: (kernel_info, TI32 ndim, TI32 portions...) with len(portions)==ndim, each >=1, -1 only last. final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeThreadMapping, TierLIR, FamilyRuntime, -1, "per-LNT thread mapping: (kernel_info, TI32 ndim, TI32 portions...) — same encoding as the grid mapping. final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeSequentialReducer, TierLIR, FamilyCollective, 2, "sequential reducer construction: (typed init value, KindLambda) — construction carries ALL state (LR-3). final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeWarpReducer, TierLIR, FamilyCollective, 2, "warp reducer construction: (typed init value, KindLambda) — construction carries ALL state (LR-3). final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeGlobalReducer, TierLIR, FamilyCollective, 2, "global (CTA-scope) reducer construction: (typed init value, KindLambda) — v0-unreachable forward scaffolding (tca.md §5.10). final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeCallMethod, TierLIR, FamilyRuntime, -1, "method call on a reducer object: (object, args...); string method payload Init|Reduce|Combine; ordered side effect (never CSE'd). final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
	{KindRuntimeGetAttribute, TierLIR, FamilyRuntime, -1, "attribute read on a runtime object: (object, args...); string attribute payload total_cta|total_thread_per_cta|index|dimension; pure read, GVN-sound. final-lower-onward: Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land (legality.go precedent)"},
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
