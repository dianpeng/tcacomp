package ir

// Kind identifies the operator of an IR node. The enumeration is versioned
// (IR-03): KindVersion 3 is the tinycudaopto re-organization (quick-261018-bl0)
// — the enum was renumbered ONCE, deliberately, here: HIR kinds group before
// LIR kinds (D-3), the generic Load/Store and the LoadTile/StoreTile kinds are
// gone, and the GMEM MemLoad/MemStore/MemNew family plus the lowered
// MemLoadSMem/MemStoreSMem/MemNewSMem and FragLoad/FragStore/FragNew families
// take their places. KindVersion 4 is the Input-list restructure
// (quick-261019-ktm): the HIR additions (IterClusterProject, KeyedAttribute,
// As, inserted after KindReturn) renumbered the LIR constants ONCE,
// deliberately — the tier-boundary rule (HIR kinds < KindForLoop) survives,
// and from v4 onward the rule is append-only again: new kinds introduced by
// later lowering phases insert before KindEnd within their tier and never
// renumber existing constants. The pass manager rejects any node
// with Kind >= KindEnd or Kind == KindInvalid.
//
// The authoritative per-kind metadata (tier, family, arity, doc) lives in
// catalog.go — the dedicated listing file (R4e); kind.go keeps the enum, the
// canonical spelling, and the family classification only.
type Kind uint32

// KindVersion is the IR node-kind schema version. v3 = the tinycudaopto
// re-organization (deliberate one-time renumbering, recorded above); v4 = the
// Input-list restructure (quick-261019-ktm): the HIR insertions renumbered
// the LIR constants once, deliberately.
const KindVersion = 4

const (
	KindInvalid Kind = iota // unused slot; verifier and legality reject it

	// --- HIR tier (D-3): the abstract kernel graph the builder produces ---
	// See catalog.go for the per-kind table; the grouping below mirrors it.

	KindMap         // lambda: map stage (value operands..., lambda, trailing As)
	KindReduce      // lambda: reduce stage (piped, KeyedAttribute init, trailing As?)
	KindMemLoad     // GMEM: load tensor[axes...] (tensor operand + axes, ImpactBy)
	KindMemStore    // GMEM: store(target access, value) — the HIR stage store
	KindMemNew      // GMEM: create a new global memory extent (extent value)
	KindIter        // loop: induction variable @m (dim/bound payload, init, step)
	KindIterCluster // loop: compound iter(@m, @k, @n) header (dims payload, inputs...)
	KindConst       // scalar-expr: typed constant
	KindParam       // scalar-expr: lambda parameter reference
	KindBinOp       // scalar-expr: binary op (opcode table mirrors syntax.md §7.6)
	KindUnaryOp     // scalar-expr: unary op
	KindSelect      // scalar-expr: typed conditional ?:
	KindCast        // scalar-expr: explicit cast to named type
	KindCall        // scalar-expr: registered pure builtin call
	KindInputParam  // var: graph-owned kernel/lambda input parameter
	KindReturn      // end: kernel return — the graph's single end node

	// --- HIR tier, Input-list restructure additions (quick-261019-ktm) ---
	// Every incoming reference of a Map/Reduce rides its Input (operand)
	// list: the as-order as a trailing As node, Reduce's init as a
	// KeyedAttribute at operand 1, and Map's lambda bindings as
	// IterClusterProject bridges into an IterCluster's input list.

	KindIterClusterProject // bridge: IterCluster input slot i → Map param i
	KindKeyedAttribute     // named attribute riding an Input list (Reduce's init)
	KindAs                 // as-order clause: dims..., trailing on Map/Reduce inputs

	// --- LIR tier (D-3): the lowered suite introduced by optimization passes ---

	KindForLoop           // lowered loop nest over a bounded dimvar (iv, bound, body)
	KindDimVar            // lowered induction variable, an (init, step) value object
	KindBoundedDimVar     // lowered dimvar bound to a max — ((iv tuple), Max)
	KindMemLoadSMem       // lowered SMEM load (src smem buffer, bivs...)
	KindMemStoreSMem      // lowered SMEM store (value, dst smem buffer, bivs...)
	KindMemNewSMem        // lowered SMEM allocation (extent)
	KindFragLoad          // lowered fragment load (src frag buffer, bivs...)
	KindFragStore         // lowered fragment store (value, dst frag buffer, bivs...)
	KindFragNew           // lowered fragment allocation (extent)
	KindTensorDimOf       // lowered shape upper bound: TensorDimOf(tensor, axis const)
	KindInductionVariable // lowered u32 iteration value: InductionVariable(iter)

	KindEnd // sentinel: all kinds < KindEnd
)

// String returns the lowercase canonical name of the kind (D-6: the canonical
// print renders this exact spelling — memstore(...), memloadsmem(...), ...).
func (k Kind) String() string {
	switch k {
	case KindInvalid:
		return "invalid"
	case KindMap:
		return "map"
	case KindReduce:
		return "reduce"
	case KindMemLoad:
		return "memload"
	case KindMemStore:
		return "memstore"
	case KindMemNew:
		return "memnew"
	case KindIter:
		return "iter"
	case KindIterCluster:
		return "itercluster"
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
	case KindReturn:
		return "return"
	case KindIterClusterProject:
		return "iterclusterproject"
	case KindKeyedAttribute:
		return "keyedattribute"
	case KindAs:
		return "as"
	case KindForLoop:
		return "forloop"
	case KindDimVar:
		return "dimvar"
	case KindBoundedDimVar:
		return "bdimvar"
	case KindMemLoadSMem:
		return "memloadsmem"
	case KindMemStoreSMem:
		return "memstoresmem"
	case KindMemNewSMem:
		return "memnewsmem"
	case KindFragLoad:
		return "fragload"
	case KindFragStore:
		return "fragstore"
	case KindFragNew:
		return "fragnew"
	case KindTensorDimOf:
		return "tensordimof"
	case KindInductionVariable:
		return "inductionvariable"
	case KindEnd:
		return "end"
	default:
		return "unknown"
	}
}

// Family classifies a Kind into its node family (D-15). FamilyMemory gained
// the whole HIR GMEM family (MemLoad/MemStore/MemNew) and the lowered SMEM/
// FRAG families with the v3 re-organization, replacing FamilyTileOp (the
// LoadTile/StoreTile kinds are gone — R4d). FamilyCollective remains reserved
// with no member kinds until reduce lowering lands. New lowering kinds join
// via new constants without enum surgery (single-tier IR rule).
type Family int

const (
	FamilyAbstract   Family = iota
	FamilyLoop              // iter/itercluster headers and the lowered for-loop
	FamilyMemory            // GMEM MemLoad/MemStore/MemNew + lowered SMEM/FRAG families
	FamilyCollective        // reserved: reducers/collectives (no members yet)
	FamilyScalarExpr        // const/param/binop/unaryop/select/cast/call
)

// Family returns the family of the kind.
func (k Kind) Family() Family {
	switch k {
	case KindConst, KindParam, KindBinOp, KindUnaryOp, KindSelect, KindCast, KindCall:
		return FamilyScalarExpr
	case KindMemLoad, KindMemStore, KindMemNew,
		KindMemLoadSMem, KindMemStoreSMem, KindMemNewSMem,
		KindFragLoad, KindFragStore, KindFragNew:
		return FamilyMemory
	case KindIter, KindIterCluster, KindForLoop:
		return FamilyLoop
	default:
		// KindMap, KindReduce, KindInputParam, KindReturn, KindDimVar,
		// KindBoundedDimVar, the Input-list restructure additions
		// (IterClusterProject, KeyedAttribute, As — reference-carrier nodes,
		// quick-261019-ktm), the lowered shape/iteration VALUE nodes
		// (TensorDimOf, InductionVariable — pure value nodes per the DimVar
		// precedent), and the sentinel values belong to the abstract suite
		// (the lowered dimvars are value-level IV objects, doc/optana/
		// induction_variable.md, not loop-nest structure).
		return FamilyAbstract
	}
}
