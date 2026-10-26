// Package tiling implements the DECIDE half of the loop-tiling pair plus
// the record interface of doc/optana/loop_tile_lower.md §1, under the
// FACTORIZATION model of doc/fix/fix4.md §1-§2 (which supersedes the
// portion-grammar model of loop_tiling.md wherever they disagree): the
// MemLoad overlap analysis, the feasible-area schedule search, the
// SMEM/FRAG cache-promotion decisions, and the decision-record side table
// the loop-tile-lower pass consumes.
//
// The fix4 constraint/objective model: per window dim an SoC ∈
// {SEQ, CTA, WARP} with an ABSOLUTE LoC (CTAs / warps) and a per-dim tile;
// composition is a factorization (gridDim = Π CTA LoC, blockDim = Π warp
// K × 32 — launch what you factorize); the constraints are MAPM/MALoC/
// CuMAP/MASRAM (§2.3) checked INCREMENTALLY over partial vectors; the
// objective is traffic avoided (K_d×T_d window bytes × lane-aware reuse
// with cross-warp sharing) with a partition-count tiebreak and the
// LoC-descending enumeration bias; budget overflow evicts by value
// density instead of failing.
//
// Ruling D2 (2026-10-20, overriding the stale "Cache Mark" section of
// loop_tiling.md): the pass NEVER mutates the IR graph — no MemLoad.ctaCache
// field, no node creation/erasure/rewiring, no field mutation. Every
// decision lives in the Record side table, published through the opt
// analysis cache as "tiling-record" and kept alive past the pass's own
// invalidation by the Preserved ["tiling-record"] whitelist (the
// loop_tile_lower.md §1 record-lifetime mechanism). The LOWER pass performs
// the node surgery when it consumes the record.
package tiling

import (
	"fmt"

	"tcacomp/internal/ir"
)

// RecordName is the analysis-cache name of the decision record — the key
// loop-tile-lower (and tiling-verify) fetch it under.
const RecordName = "tiling-record"

// SoCType is the space-of-concurrency assignment of one iter
// (doc/optana/loop_tiling.md "Parallisation Strategy"): CTA, WARP (v0's
// only thread granularity — always whole warps of 32), or sequential. The
// zero value is sequential, matching the doc's default for unassigned
// loops.
type SoCType int

const (
	// SoCSequential is a plain sequential loop; LoC is forced to 1.
	SoCSequential SoCType = iota
	// SoCTA is a CTA-mapped dimension; it implicitly couples a WARP
	// assignment for the tile-inside walk.
	SoCTA
	// SoCWARP is a warp-group-mapped dimension (a disjoint whole-warp shard
	// of the CTA's thread budget).
	SoCWARP
)

// String renders the SoC type for diagnostics.
func (s SoCType) String() string {
	switch s {
	case SoCSequential:
		return "seq"
	case SoCTA:
		return "cta"
	case SoCWARP:
		return "warp"
	default:
		return "?"
	}
}

// CacheClass is the storage class of a promoted cache site. The zero value
// (ClassNone) means unpromoted/unclassified — the record never invents a
// class for a site the judge declined.
type CacheClass int

const (
	// ClassNone is the zero value: the site is not promoted.
	ClassNone CacheClass = iota
	// ClassSMEM is a shared-memory cache (last positions parallel within a
	// CTA but not within a single thread).
	ClassSMEM
	// ClassFRAG is a register-fragment cache (last two positions fully
	// sequential, so the access is single-thread).
	ClassFRAG
)

// String renders the cache class for diagnostics.
func (c CacheClass) String() string {
	switch c {
	case ClassNone:
		return "none"
	case ClassSMEM:
		return "smem"
	case ClassFRAG:
		return "frag"
	default:
		return "?"
	}
}

// IterDecision is one per-Iter schedule entry (fix4 §2.7; the
// loop_tile_lower.md §1a interface): the space of concurrency, its level
// of concurrency, and the tile size. LoC is the ABSOLUTE count under the
// fix4 factorization model: CTA → the number of CTAs factorizing this dim
// (gridDim.x = Π CTA LoC over the LNT), WARP → the number of warps
// factorizing it (Π warp K ≤ MaxWarpsPerCTA, blockDim = Π K × 32), SEQ →
// 1 implicit.
type IterDecision struct {
	SoC      SoCType
	LoC      int
	TileSize int
}

// SiteDecision is one per-MemLoad cache entry (loop_tile_lower.md §1b):
// the promotion flag, the tie level, and the storage class. Tie == nil
// means the cache sits directly above the OUTERMOST loop of the site's
// LNT. DoSoC is deliberately absent — the lower pass derives it (§1c).
//
// The 2026-10-21 rulings (R2/R4/F4/F6) make the record self-describing —
// Reuse is the site's re-traversal count within one fill epoch (0 when
// not computed) under the 2026-10-26 one-currency amendment semantics
// (§2.4/§2.7: COLLECTIVE per window instance for SMEM sites, PER-THREAD
// for FRAG sites — the holders scaling that puts FRAG traffic in the same
// CTA-wide currency lives only in the objective, never in the record),
// Bytes the site's footprint under the window-bytes model (SMEM window
// bytes; per-thread bytes for FRAG), and Reason the decline explanation
// on a non-promoted entry ("" when promoted). Non-candidates (cross-LNT
// / never-bucketed loads, R4) are declared with Promoted=false entries —
// never silently dropped.
type SiteDecision struct {
	Promoted bool
	Tie      ir.Node
	Class    CacheClass
	// Reuse is the reuse count the judge computed under the §2.4
	// one-currency amendment: COLLECTIVE per window instance for SMEM
	// sites (>= 2 on every promoted site — collective trips on a reuse
	// dim are at least its tile), PER-THREAD for FRAG sites; the computed
	// value on a decline; else 0. The holders scaling to the CTA-wide
	// currency lives only in the objective, never here.
	Reuse int
	// Bytes is the site's window-model footprint (window bytes for SMEM,
	// per-thread bytes for FRAG); the computed value on a frag-byte or
	// reuse decline; else 0.
	Bytes int
	// Reason names the decline cause on a non-promoted entry ("" when
	// promoted) — one of the Reason constants below.
	Reason string
}

// The decline reasons (R2 tier rule, F4 drop-rule gate, F6 frag budget,
// the per-thread reuse gate, the R4 non-candidate declarations, and the
// fix4 §2.6 greedy-eviction demotion). Stable strings: the record,
// tiling-verify, and the tests pin them.
const (
	// ReasonCTACovered: a covered dim is CTA-assigned — no coherent fill
	// variant exists under the v0 mechanics (R2 tier rule, refusal).
	ReasonCTACovered = "covered dim CTA-assigned: no coherent fill variant (R2 tier rule)"
	// ReasonFragBytes: the per-thread FRAG footprint exceeds
	// MaxFragPerThread (F6).
	ReasonFragBytes = "FRAG bytes exceed MaxFragPerThread (F6)"
	// ReasonLowReuse: reuse < 2 in the site's currency (collective for
	// SMEM, per-thread for FRAG — the §2.4 one-currency gate): the cached
	// bytes are not re-traversed, so no CTA-wide traffic is avoided.
	ReasonLowReuse = "reuse < 2 in the site's currency: no traffic avoided"
	// ReasonEvicted: the site WAS promotion-eligible but the kernel-wide
	// MASRAM gates overflowed and the greedy value-density spill (fix4
	// §2.6) demoted it to GMEM — the loops stay tiled, only the cache is
	// lost. Class/Reuse/Bytes are retained on the entry so tiling-verify
	// can still cross-check the judge.
	ReasonEvicted = "kernel-wide budget overflow: evicted by value density (fix4 2.6)"
	// ReasonCrossLNT: the load's INS spans >= 2 LNTs — not a v0
	// candidate (R4 fail-closed; the union judge is future work).
	ReasonCrossLNT = "cross-LNT INS: not a v0 candidate (R4)"
	// ReasonEmptyINS: the load references no iter at all — never
	// bucketed, declared non-candidate (R4).
	ReasonEmptyINS = "empty INS: never bucketed (R4)"
	// ReasonNoOwner: the load's INS holds no forest member (e.g. only
	// erased iters) — declared non-candidate (R4).
	ReasonNoOwner = "INS holds no LNT member (R4)"
	// ReasonUnknownIter: a covered iter carries no schedule decision —
	// the judge fails closed.
	ReasonUnknownIter = "covered iter without a schedule decision"
	// ReasonEmptyCovered: the site covers no dimension (M2: degenerate
	// scalar cache, filtered at the decide pass).
	ReasonEmptyCovered = "empty covered set (M2)"
)

// Record is the tiling decision side table (loop_tile_lower.md §1):
// per-Iter entries keyed by BIV node identity — the Iter's operand 0,
// GVN-canonical so a later fusion re-run deleting Iter prefix nodes cannot
// orphan keys — and per-site entries keyed by the MemLoad node id
// (BasePtr().ID()).
//
// A MISSING Iters entry means the iter was not part of any forest tree the
// pass saw; a MISSING Sites entry means the load was not admitted to the
// candidate set — the lower pass treats the load as untouched.
//
// Verified is the tiling-verify stamp (F7, ruled 2026-10-21): tiling-verify
// sets it on a record that passed its consistency sweep, and tile-lower
// refuses an unstamped record fail-closed — the per-iter feasibility
// re-checks of the fix4 model (LoC preset membership, the per-LNT
// factorization product bounds, the CTA-dim presence, the tile floor, the
// kernel-wide budget sums, and the judge cross-check) live in
// tiling-verify, and a corrupted record must NOT reach the rewrite
// unvalidated. It is the one field tiling-verify mutates on the published
// record (261021-o61 T3; the decide-half surface is otherwise untouched).
type Record struct {
	Iters map[ir.Node]IterDecision
	Sites map[uint64]SiteDecision
	// Verified reports that tiling-verify ran to completion over THIS
	// record (the F7 gate tile-lower consumes).
	Verified bool
}

// Name satisfies analysis.Analysis structurally (the single-method Name()
// interface — no internal/analysis import, no registry entry): the record
// publishes itself through the opt analysis cache under RecordName.
func (Record) Name() string { return RecordName }

// IterDecisionOf resolves an Iter node to its decision: the record is keyed
// by the iter's BIV (DataDeps(iter)[0]), so the lookup resolves that operand
// first. Fail-closed: nil, erased, foreign, and operand-less nodes report
// false.
func (r Record) IterDecisionOf(iter ir.Node) (IterDecision, bool) {
	if iter == nil || iter.BasePtr() == nil || iter.BasePtr().Erased() {
		return IterDecision{}, false
	}
	deps := ir.DataDeps(iter)
	if len(deps) == 0 {
		return IterDecision{}, false
	}
	d, ok := r.Iters[deps[0]]
	return d, ok
}

// SiteDecisionOf resolves a MemLoad node to its site decision, keyed by the
// node id. Fail-closed: nil and erased nodes report false.
func (r Record) SiteDecisionOf(load ir.Node) (SiteDecision, bool) {
	if load == nil || load.BasePtr() == nil || load.BasePtr().Erased() {
		return SiteDecision{}, false
	}
	d, ok := r.Sites[load.BasePtr().ID()]
	return d, ok
}

// ---------------------------------------------------------------------------
// Tiling provenance record (quick-261022-l1o, doc/optana/final_lower.md
// "Tiling Provenance Record" as amended): the per-dim between/inside loop
// relationship + the per-LNT mapping-portion plans, published as DATA by
// tile-lower and consumed by final-lower. ir deliberately knows nothing
// about opt/tiling (the astcfg.go layering rule), so the relationship rides
// the analysis cache exactly like the decision record — but keyed by nodes
// that SURVIVE tile-lower (dim InputParams and the generated ForLoop roots),
// which is why it gets its own name and lifetime.
// ---------------------------------------------------------------------------

// ProvenanceName is the analysis-cache name of the provenance record — the
// key final-lower fetches it under. It must survive tile-lower's own
// invalidation (tile-lower's Preserved) and the schedule pass (schedule's
// Preserved).
const ProvenanceName = "tiling-provenance"

// DimProvenance is one rewritten dim's loop relationship (final_lower.md
// "per-dim provenance entry"): the tile-BETWEEN forloop, the tile-INSIDE
// forloop (every dim is tiled, T >= 2, so BOTH always exist — the 2026-10-24
// tile-disable ruling), the effective SoC (mirrors ASTLoopMeta — LR-7's
// specialization input), and the effective tile size. For SoCTA dims the
// entry additionally carries
// the COUPLED tile-inside walk's materialization facts (its ThreadMap slot
// and thread count) — final-lower's per-element ownership-gate walk over a
// reduce's output dims mirrors exactly this form (the 2026-10-23 reducer
// redirect).
type DimProvenance struct {
	Between ir.Node
	Inside  ir.Node // always populated (every dim is tiled, T >= 2)
	Soc     SoCType
	Tile    int
	// CoupledDoSoC is the CTA dim's coupled inside-walk ThreadMap slot
	// (-1 = the dim is not CTA-assigned).
	CoupledDoSoC int
	// CoupledThreads is the coupled inside-walk thread count (0 = not
	// CTA-assigned): under the fix4 model it is the LNT's blockDim =
	// (Π warp K) × 32 — the flat-init walk's step and the ownership-gate
	// walk's step.
	CoupledThreads int
}

// MappingPlan is one LNT's runtime mapping construction data (LR-11/P-3;
// fix4 §3.2/§4.1): the GridMapping and ThreadMapping SHAPE vectors — the
// field NAMES survive from the portion grammar, the semantics are the
// absolute factorization extents. GridPortions holds the per-CTA-dim
// absolute LoC_d (in DoSoC order — gridDim.x = Π LoC); ThreadPortions holds
// the per-WARP-dim warp counts K_d (in DoSoC order — warp coordinates in
// warp units, blockDim = Π K × 32). The CTA dim's coupled inside walk uses
// the NO-ARG flat thread rank and needs no ThreadMap slot (§1.3).
type MappingPlan struct {
	GridPortions   []int
	ThreadPortions []int
}

// Provenance is the tile-lower product final-lower consumes: per-dim entries
// keyed by the DIM node (a RoleDim InputParam — stable across every later
// pass, and exactly what final-lower's @r derivation naturally yields) and
// per-LNT plans keyed by the generated LNT root ForLoop node.
type Provenance struct {
	Dims map[ir.Node]DimProvenance
	LNTs map[ir.Node]MappingPlan
	// MaxThreadsPerCTA is the DERIVED launch blockDim under the fix4
	// factorization model: max over LNTs of (Π warp K × 32). The field name
	// survives (final-lower's reducer staging extent reads #warps =
	// MaxThreadsPerCTA/32 from here, ruling D8 2026-10-23) — it is NOT the
	// retired portion-grammar Params knob of the same name; with zero WARP
	// dims the product is the empty product 1 and the launch is one warp
	// (blockDim 32).
	MaxThreadsPerCTA int
}

// Name satisfies analysis.Analysis structurally (the single-method Name()
// interface — the Record precedent): the provenance publishes itself through
// the opt analysis cache under ProvenanceName.
func (Provenance) Name() string { return ProvenanceName }

// ---------------------------------------------------------------------------
// Params: the externally provided thresholds (doc: "the thresholds are
// EXTERNALLY PROVIDED to the pass") — the fix4 §2.1 factorization-model
// hyperparameter surface. The portion grammar (the named CTA/warp share
// constants, the per-LNT launch-total exhaustion knobs, and the tile-vs-
// coupled-walk gate) is RETIRED with fix4 §2.1: composition is a
// factorization now (absolute LoC/K per dim), the launch is derived from
// the factorization, and the flat-init coupled walk (init = the flat
// thread rank, step = blockDim) is single-owner exact for ANY tile_max —
// fix4 R-HI03 deletes all three gates the old grammar carried.
// ---------------------------------------------------------------------------

// Params carries the pass thresholds with the fix4 §2.1 defaults. Every
// field is validated by Validate before any computation — external input
// is never trusted (fail-fast at the boundary).
type Params struct {
	// WindowK is the innermost window depth (fix4 §2.1: 4, was 5).
	WindowK int
	// TileSizes is the per-dim tile preset set. Every entry must be >= 2:
	// the compiler never generates a non-tiling schedule (the 2026-10-24
	// tile-disable ruling), so tile 1 is not a legal schedule.
	TileSizes []int
	// CtaLoCPresets are the absolute CTA counts searched per CTA dim
	// (fix4 §1.1). Entries >= 2; subject to the extent cap at search time.
	CtaLoCPresets []int
	// WarpLoCPresets are the absolute warp counts searched per WARP dim.
	// Entries >= 2 — a WARP dim with K = 1 is SEQ and pruned from the
	// feasible area (dominated, fix4 §1.2).
	WarpLoCPresets []int
	// MaxPartitionCta is the MAPM cap on CTA dims per LNT (expanded
	// instances, fix4 §2.3).
	MaxPartitionCta int
	// MaxPartitionThr is the MAPM cap on expanded THR instances per LNT
	// (WARP dims + CTA dims).
	MaxPartitionThr int
	// MaxCTA is the gridDim policy cap (the MALoC CTA bound: per LNT,
	// Π CTA LoC <= MaxCTA).
	MaxCTA int
	// MaxWarpsPerCTA is the hardware warp bound (the MALoC warp bound:
	// per LNT, Π warp K <= MaxWarpsPerCTA <= 32).
	MaxWarpsPerCTA int
	// MaxCache is the SMEM budget in bytes (the project's 64 KB SMEM
	// budget; only promoted ClassSMEM sites count). The sum is
	// KERNEL-WIDE (R3, ruled 2026-10-21): the decide feasibility, the
	// tiling-verify gate, and the lower's budget resolution all gate the
	// same sum accumulated over ALL LNTs' promoted sites — never a
	// per-LNT one. Under fix4 §2.6 an overflow evicts by value density
	// (greedy spill); it is never a hard error.
	MaxCache int
	// MaxFragPerThread is the per-thread FRAG byte budget (F6, ruled
	// 2026-10-21; default 256 bytes/thread): the SUM of promoted FRAG
	// sites' per-thread bytes, kernel-wide, must stay within it — an
	// unbounded register footprint spills to local memory (doc/tca.md
	// §5.8), strictly worse than the GMEM load it replaced. It gates the
	// judge (a single site over the cap is declined) and the schedule
	// feasibility (the accumulated sum, spill-gated per fix4 §2.6), and
	// backs the lower's halve/drop fallback.
	MaxFragPerThread int
	// CacheDims is the dimension-level reuse count: only the last
	// CacheDims tensor dimensions may be promoted (hardcoded 2 in the doc).
	CacheDims int
	// MaxCS caps the candidate-set size ("do not consider too many MemLoad
	// to be replaced by cache").
	MaxCS int
	// ReducedDimWarp allows WARP-SoC on reduced (reducer-consuming) dims —
	// the split-k capability gate (fix4 §3.6, Wave 2). Wave 1 keeps it
	// false: the decide-side feasible area prunes WARP digits on reduced
	// dims, and tile-lower fails closed on any that slip through.
	ReducedDimWarp bool
	// NominalExtent is the documented stand-in extent for a reuse
	// dimension whose BIV max operand is NOT a constant. Dormant: reuse is
	// tile-derived under the fix4 model, so extents enter only the
	// R-LOCCAP preset cap (const extents), never the objective; the knob
	// stays on the surface for the future extent-aware ranking models.
	NominalExtent int
}

// DefaultParams returns the fix4 §2.1 defaults: window K=4, tiles
// {32,64,128,256}, CTA LoC presets {2,4,8,16,32,64}, warp K presets
// {2,4,8,16,32}, MAPM caps 3/3, MaxCTA 512 (gridDim policy cap),
// MaxWarpsPerCTA 32, 64 KB SMEM, 256 bytes/thread FRAG (F6), CacheDims 2,
// MaxCS 16, ReducedDimWarp false (Wave 1), nominal extent 4096.
func DefaultParams() Params {
	return Params{
		WindowK:          4,
		TileSizes:        []int{32, 64, 128, 256},
		CtaLoCPresets:    []int{2, 4, 8, 16, 32, 64},
		WarpLoCPresets:   []int{2, 4, 8, 16, 32},
		MaxPartitionCta:  3,
		MaxPartitionThr:  3,
		MaxCTA:           512,
		MaxWarpsPerCTA:   32,
		MaxCache:         65536,
		MaxFragPerThread: 256,
		CacheDims:        2,
		MaxCS:            16,
		ReducedDimWarp:   false,
		NominalExtent:    4096,
	}
}

// Validate fail-fasts on the fix4 §2.1 surface: the preset lists must be
// non-empty with entries >= 2 (a K=1 warp digit is SEQ, pruned; a LoC < 2
// dim is not a partition), the bounds positive, MaxPartitionCta/
// MaxPartitionThr >= 1 (a zero MAPM cap leaves no legal factorization),
// MaxWarpsPerCTA in [1, 32] (the hardware warp bound), and TileSizes
// non-empty with entries >= 2 (tile size 1 is not a legal schedule — the
// compiler never generates a non-tiling schedule, the 2026-10-24 ruling;
// a tile below 2 would poison the arithmetic).
func (p Params) Validate() error {
	if p.WindowK < 1 {
		return fmt.Errorf("tiling: WindowK %d must be >= 1", p.WindowK)
	}
	if len(p.TileSizes) == 0 {
		return fmt.Errorf("tiling: TileSizes must not be empty")
	}
	for _, v := range p.TileSizes {
		if v < 2 {
			return fmt.Errorf("tiling: TileSizes entry %d is below 2 — tile size 1 is not a legal schedule: the compiler never generates a non-tiling schedule (2026-10-24 ruling); every entry must be >= 2", v)
		}
	}
	if len(p.CtaLoCPresets) == 0 {
		return fmt.Errorf("tiling: CtaLoCPresets must not be empty")
	}
	for _, v := range p.CtaLoCPresets {
		if v < 2 {
			return fmt.Errorf("tiling: CtaLoCPresets entry %d is below 2 (an absolute CTA count of 1 is not a partition)", v)
		}
	}
	if len(p.WarpLoCPresets) == 0 {
		return fmt.Errorf("tiling: WarpLoCPresets must not be empty")
	}
	for _, v := range p.WarpLoCPresets {
		if v < 2 {
			return fmt.Errorf("tiling: WarpLoCPresets entry %d is below 2 — a WARP dim with K = 1 is SEQ and pruned from the feasible area (fix4 1.2)", v)
		}
	}
	if p.MaxPartitionCta < 1 {
		return fmt.Errorf("tiling: MaxPartitionCta %d must be >= 1 (a zero MAPM cap leaves no legal factorization)", p.MaxPartitionCta)
	}
	if p.MaxPartitionThr < 1 {
		return fmt.Errorf("tiling: MaxPartitionThr %d must be >= 1 (a zero MAPM cap leaves no legal factorization)", p.MaxPartitionThr)
	}
	if p.MaxCTA < 2 {
		return fmt.Errorf("tiling: MaxCTA %d must be >= 2 (the gridDim policy cap, the MALoC CTA bound)", p.MaxCTA)
	}
	if p.MaxWarpsPerCTA < 1 || p.MaxWarpsPerCTA > 32 {
		return fmt.Errorf("tiling: MaxWarpsPerCTA %d must be within [1, 32] (the hardware warp bound, the MALoC warp bound)", p.MaxWarpsPerCTA)
	}
	if p.MaxCache < 1 {
		return fmt.Errorf("tiling: MaxCache %d must be >= 1", p.MaxCache)
	}
	if p.MaxFragPerThread < 1 {
		return fmt.Errorf("tiling: MaxFragPerThread %d must be >= 1 (F6: the per-thread FRAG byte budget)", p.MaxFragPerThread)
	}
	if p.CacheDims < 1 {
		return fmt.Errorf("tiling: CacheDims %d must be >= 1", p.CacheDims)
	}
	if p.MaxCS < 1 {
		return fmt.Errorf("tiling: MaxCS %d must be >= 1", p.MaxCS)
	}
	if p.NominalExtent < 1 {
		return fmt.Errorf("tiling: NominalExtent %d must be >= 1", p.NominalExtent)
	}
	return nil
}
