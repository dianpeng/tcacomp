package ir

// node_runtime.go holds the RUNTIME node family (quick-261022-l1o,
// doc/optana/final_lower.md as amended — the "Intrinsic" vocabulary of
// quick-261021-jay is renamed to Runtime throughout, LR-6): the two
// pre-final-lower launch-map QUERY kinds plus the eight final-lower kinds.
//
//   - Queries (renamed, enum values unchanged — a pure rename, no renumber):
//     RuntimeGridQuery/RuntimeThreadQuery — per-LNT scoped reads of the
//     GridConcurrencyMap / ThreadMap the runtime constructs for the enclosing
//     LNT nest. field ∈ {"index", "dimension"}; result u32; DoSoC is the
//     single const-u32 operand. FamilyAbstract (the pure-value precedent) so
//     EffectOf is free and GVN-CSE of repeated (DoSoC, field) queries is
//     sound. After final-lower's query rewrite these kinds vanish from every
//     graph (pre-final-lower material only, LR-6/LR-12).
//   - Constructions (LR-4/LR-5): RuntimeKernelInfo(), RuntimeGridMapping/
//     RuntimeThreadMapping(%kernel_info, ndim, portions...), and the three
//     reducer constructions RuntimeSequentialReducer/RuntimeWarpReducer/
//     RuntimeGlobalReducer(%init, %lambda) — construction carries ALL state
//     (LR-3): the init value is unwrapped from the Reduce's
//     KeyedAttribute("init") and the combiner rides as an SSAUse to the
//     Reduce's trailing lambda node; Init() is argument-free and the
//     (prev, cur) application is runtime-internal.
//   - The two generic nodes (LR-4): RuntimeCallMethod (ordered side effect —
//     never pure, never GVN-CSE'd; method payload ∈ {Init, Reduce, Combine};
//     applies to a reducer construction object only) and RuntimeGetAttribute
//     (pure read on immutable objects; attribute payload ∈ {total_cta,
//     total_thread_per_cta, index, dimension}). Names are STRING PAYLOADS
//     (the runtimeQueryNode.field precedent), never operands.
//
// Families (LR-5): FamilyRuntime (NEW) = KernelInfo/GridMapping/ThreadMapping
// + CallMethod + GetAttribute; FamilyCollective gets its first members — the
// three reducer constructions. All new kinds are LIR tier, appended before
// KindEnd (append-only rule, no KindVersion bump). RuntimeGlobalReducer is
// v0-unreachable forward scaffolding (tca.md §5.10: no CTA-parallel reduced
// dims in v0) — registered, never emitted by the Soc rule on reachable
// inputs.
//
// The six-companion checklist per new kind: node struct + catalog row
// (catalog.go), allocNode case (node.go), verify shape case (verify.go,
// P-3's closed cross-checks), typeinfer rule (internal/opt/typeinfer), the
// legality-window note (the catalog row doc — constructedWindow iterates
// AllKinds() so every new kind is Legal at StageConstructed automatically),
// and contentHash/contentEqual (value.go — string payloads via mixString).

// runtimeQueryNode: pre-final-lower launch-map query; ops = [DoSoC const
// u32]; field is the string payload selecting the query ("index" |
// "dimension"). Shared by both query kinds — the kind discriminates grid vs
// thread map.
type runtimeQueryNode struct {
	*Base
	operandList
	field string
}

// NewRuntimeGridQuery creates a GridConcurrencyMap query node: field selects
// the query ("index" | "dimension"), dosoc is the const-u32 map dimension
// binding. The verifier's shape check pins the operand count, the DoSoC
// constness, and the field vocabulary.
func (g *Graph) NewRuntimeGridQuery(origin string, field string, dosoc Node) Node {
	n := g.allocNode(KindRuntimeGridQuery, origin, 1).(*runtimeQueryNode)
	n.field = field
	wireOperands(n, dosoc)
	return n
}

// NewRuntimeThreadQuery creates a ThreadMap query node: field selects the
// query ("index" | "dimension"), dosoc is the const-u32 map dimension
// binding. Same shape contract as NewRuntimeGridQuery.
func (g *Graph) NewRuntimeThreadQuery(origin string, field string, dosoc Node) Node {
	n := g.allocNode(KindRuntimeThreadQuery, origin, 1).(*runtimeQueryNode)
	n.field = field
	wireOperands(n, dosoc)
	return n
}

// NewRuntimeGridQueryFlat creates the ZERO-OPERAND GridMapping query — the
// flat-index() ruling (2026-10-23): field is pinned "index" (the CTA's
// absolute flattened 1-D id, the blockIdx.x analog). The verifier's shape
// case admits the zero-operand form ONLY for field "index"; "dimension"
// still demands exactly one const-u32 DoSoC.
func (g *Graph) NewRuntimeGridQueryFlat(origin string) Node {
	n := g.allocNode(KindRuntimeGridQuery, origin, 0).(*runtimeQueryNode)
	n.field = "index"
	return n
}

// NewRuntimeThreadQueryFlat creates the ZERO-OPERAND ThreadMap query — the
// flat-index() ruling (2026-10-23): field is pinned "index" (the thread's
// absolute flattened 1-D id in the CTA, the threadIdx.x analog; lane/warp
// identity derives from it). Same shape contract as NewRuntimeGridQueryFlat.
func (g *Graph) NewRuntimeThreadQueryFlat(origin string) Node {
	n := g.allocNode(KindRuntimeThreadQuery, origin, 0).(*runtimeQueryNode)
	n.field = "index"
	return n
}

// NewRuntimeThreadQueryLane creates the ZERO-OPERAND lane query — the fix4
// §3.1 WARP lane-chunk walks (T4): field is pinned "lane_id" (the
// querying thread's lane within its warp — a pure function of the flat
// rank; the lane has NO map dimension, hence no DoSoC operand). The
// verifier's shape case admits the zero-operand form for field "lane_id"
// exactly as for "index"; final-lower rewrites it to the zero-arg
// lane_id attribute read on the LNT's ThreadMapping.
func (g *Graph) NewRuntimeThreadQueryLane(origin string) Node {
	n := g.allocNode(KindRuntimeThreadQuery, origin, 0).(*runtimeQueryNode)
	n.field = "lane_id"
	return n
}

// runtimeKernelInfoNode: the one-per-kernel runtime information object —
// zero operands, zero payload (LR-4: a dedicated construction node, no
// generic RuntimeConstruct). final-lower prepends it as the FIRST item of
// the kernel Body Group (LR-11).
type runtimeKernelInfoNode struct {
	*Base
	operandList
}

// NewRuntimeKernelInfo creates the kernel information object: the source of
// total_cta / total_thread_per_cta attribute reads. The verifier rejects any
// operand (arity 0).
func (g *Graph) NewRuntimeKernelInfo(origin string) Node {
	return g.allocNode(KindRuntimeKernelInfo, origin, 0)
}

// runtimeMappingNode: one per-LNT mapping object construction; ops =
// [%kernel_info, ndim const (TI32), shape consts (TI32) ...] with
// len(shapes) == ndim, each shape the dim's ABSOLUTE extent >= 2 (fix4
// §4.1: no portions, no −1 — the D3 remainder rule is retired; the T5
// verifier rule). Shared by RuntimeGridMapping and RuntimeThreadMapping —
// the kind discriminates.
type runtimeMappingNode struct {
	*Base
	operandList
}

// NewRuntimeGridMapping creates a GridMapping construction over the kernel
// info object with the given CTA portion vector: the constructor allocates
// the TI32 ndim and portion constants internally (ndim = len(portions), so
// the operand count always matches the pinned shape).
func (g *Graph) NewRuntimeGridMapping(origin string, ki Node, portions []int) Node {
	return g.newRuntimeMapping(KindRuntimeGridMapping, origin, ki, portions)
}

// NewRuntimeThreadMapping creates a ThreadMapping construction over the
// kernel info object with the given warp portion vector — same encoding as
// NewRuntimeGridMapping.
func (g *Graph) NewRuntimeThreadMapping(origin string, ki Node, portions []int) Node {
	return g.newRuntimeMapping(KindRuntimeThreadMapping, origin, ki, portions)
}

// newRuntimeMapping is the shared mapping constructor body.
func (g *Graph) newRuntimeMapping(k Kind, origin string, ki Node, portions []int) Node {
	ops := make([]Node, 0, len(portions)+2)
	ops = append(ops, ki)
	i32 := InternScalar(TI32)
	ops = append(ops, g.NewConst(origin, i32, float64(len(portions))))
	for _, p := range portions {
		ops = append(ops, g.NewConst(origin, i32, float64(p)))
	}
	return g.NewNode(k, origin, ops...)
}

// runtimeReducerNode: one specialized reducer construction; ops = [init,
// lambda] for the sequential specialization, [init, lambda, thread mapping,
// staging allocation] for warp/global (R1/R2 ruling 2026-10-23: the
// construction carries ALL state — combiner, mapping, and the compiler-pool
// staging row). Shared by the three specializations — the kind
// discriminates seq/warp/global.
type runtimeReducerNode struct {
	*Base
	operandList
}

// NewRuntimeSequentialReducer creates the sequential-scope reducer
// construction (X's Soc 0, LR-7): init is the Reduce's unwrapped
// KeyedAttribute("init") value, lambda is the Reduce's trailing lambda
// node. The verifier pins the operand pair (a typed init value, a
// KindLambda).
func (g *Graph) NewRuntimeSequentialReducer(origin string, init, lambda Node) Node {
	return g.NewNode(KindRuntimeSequentialReducer, origin, init, lambda)
}

// NewRuntimeWarpReducer creates the warp-scope reducer construction (X's
// Soc 2). Operand contract per the R1/R2 ruling (2026-10-23): (typed init,
// combiner lambda, the owning LNT's RuntimeThreadMapping, the staging
// SMEM-class MemNew — one #warps row in the compiler's dynamic-SMEM pool).
// Warp/lane/count identity is runtime-internal, derived from the mapping;
// the staging row is the ctor-provided SmemView source. The verifier pins
// all four operands and their kinds.
func (g *Graph) NewRuntimeWarpReducer(origin string, init, lambda, threadMap, staging Node) Node {
	return g.NewNode(KindRuntimeWarpReducer, origin, init, lambda, threadMap, staging)
}

// NewRuntimeGlobalReducer creates the global (CTA-scope) reducer
// construction (X's Soc 1). Same 4-operand contract as the warp reducer
// (R1/R2). v0-unreachable forward scaffolding — registered per the
// interface-uniformity ruling, never emitted on reachable inputs (tca.md
// §5.10: no CTA-parallel reduced dims in v0).
func (g *Graph) NewRuntimeGlobalReducer(origin string, init, lambda, threadMap, staging Node) Node {
	return g.NewNode(KindRuntimeGlobalReducer, origin, init, lambda, threadMap, staging)
}

// runtimeCallNode: a METHOD call on a runtime object — ops = [object,
// args...]; method is the string payload, closed vocabulary
// {Init, Reduce, Combine}. Ordered side effect (P-2): EffectOf is full, so
// the node is never GVN-CSE'd and never DCE-eligible through value
// numbering. Init/Reduce are statements (untyped); Combine produces the
// reduced value (typed T = the reducer construction's init operand type).
type runtimeCallNode struct {
	*Base
	operandList
	method string
}

// NewRuntimeCallMethod creates a method call node: method ∈ {Init, Reduce,
// Combine}, object is the reducer construction node, args are the call's
// arguments (the Reduce call carries between-loop IVs + the piped value,
// LR-13). The verifier pins the payload vocabulary and the object kind.
func (g *Graph) NewRuntimeCallMethod(origin string, method string, object Node, args ...Node) Node {
	n := g.allocNode(KindRuntimeCallMethod, origin, len(args)+1).(*runtimeCallNode)
	n.method = method
	ops := append([]Node{object}, args...)
	wireOperands(n, ops...)
	return n
}

// runtimeGetAttributeNode: an ATTRIBUTE read on a runtime object — ops =
// [object, args...]; attribute is the string payload, closed vocabulary
// {total_cta, total_thread_per_cta, index, dimension}. Pure read on
// immutable objects (KernelInfo, mappings) — GVN-sound (P-2); no pass runs
// after final-lower, so value-identical constructions can never be CSE-merged
// anyway.
type runtimeGetAttributeNode struct {
	*Base
	operandList
	attribute string
}

// NewRuntimeGetAttribute creates an attribute read node: attribute ∈
// {total_cta, total_thread_per_cta, index, dimension}; "total_*" applies to
// a KernelInfo object with NO extra args, "index"/"dimension" to a mapping
// object with EXACTLY ONE const-u32 arg (P-3 — the verifier pins every
// cross-check).
func (g *Graph) NewRuntimeGetAttribute(origin string, attribute string, object Node, args ...Node) Node {
	n := g.allocNode(KindRuntimeGetAttribute, origin, len(args)+1).(*runtimeGetAttributeNode)
	n.attribute = attribute
	ops := append([]Node{object}, args...)
	wireOperands(n, ops...)
	return n
}

// RuntimeMethodOf returns the method payload of a RuntimeCallMethod node
// ("Init", "Reduce", or "Combine"), or "" for any other node kind — the
// exported read face (the payload read-face precedent, node_expr.go), so
// passes outside package ir never assert the concrete node type.
func RuntimeMethodOf(n Node) string {
	if c, ok := n.(*runtimeCallNode); ok {
		return c.method
	}
	return ""
}

// RuntimeAttributeOf returns the attribute payload of a
// RuntimeGetAttribute node or the field payload of a launch-map query node
// (the renamed IntrinsicFieldOf face — queries read "index"/"dimension",
// the same vocabulary the rewritten attribute reads carry), or "" for any
// other node kind.
func RuntimeAttributeOf(n Node) string {
	switch t := n.(type) {
	case *runtimeGetAttributeNode:
		return t.attribute
	case *runtimeQueryNode:
		return t.field
	}
	return ""
}

// isRuntimeReducerKind reports whether k is one of the three reducer
// construction kinds (the CallMethod object domain, P-3).
func isRuntimeReducerKind(k Kind) bool {
	switch k {
	case KindRuntimeSequentialReducer, KindRuntimeWarpReducer, KindRuntimeGlobalReducer:
		return true
	}
	return false
}

// isRuntimeMappingKind reports whether k is one of the two mapping
// construction kinds (the getattr index/dimension object domain, P-3).
func isRuntimeMappingKind(k Kind) bool {
	return k == KindRuntimeGridMapping || k == KindRuntimeThreadMapping
}
