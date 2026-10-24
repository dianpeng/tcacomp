// Package tiling implements the DECIDE half of the loop-tiling pair per
// doc/optana/loop_tiling.md (final model, as patched by 261021-gca) plus
// the record interface of doc/optana/loop_tile_lower.md §1: the MemLoad
// overlap analysis, the parallelization-schedule (PSched) search, the
// SMEM/FRAG cache-promotion decisions, and the decision-record side table
// the future loop-tile-lower pass consumes.
//
// The FINAL constraint/objective model (loop_tiling.md "LoC concurrency
// bounds, the FINAL MODEL" + "Objective Function: for a specific PSched,
// SINGLE-TERM"): the constraints are the PORTION GRAMMAR — the named
// PortionCTA/PortionWarp constants, per-LNT exhaustion of the launch
// totals, and the runtime assert-and-trap divisibility pre-conditions —
// while the objective is single-term traffic avoided: parallelism is
// CONSTANT under mandatory per-LNT exhaustion, so it carries no ranking
// signal. The searcher additionally obeys the v0 POLICY ("Composition
// Semantics, CTA x WARP"): at most one CTA dim and at most one independent
// WARP dim per LNT.
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

// IterDecision is one per-Iter schedule entry (loop_tile_lower.md §1a):
// the space of concurrency, its level of concurrency, and the tile size.
// Sequential forces LoC 1.
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
// The 2026-10-21 rulings (R2/R4/F4/F6) make the record self-describing:
// Reuse is the per-thread re-traversal count within one fill epoch (0
// when not computed), Bytes the site's footprint under the window-bytes
// model (SMEM window bytes; per-thread bytes for FRAG), and Reason the
// decline explanation on a non-promoted entry ("" when promoted).
// Non-candidates (cross-LNT / never-bucketed loads, R4) are declared
// with Promoted=false entries — never silently dropped.
type SiteDecision struct {
	Promoted bool
	Tie      ir.Node
	Class    CacheClass
	// Reuse is the per-thread reuse count the judge computed (>= 2 on
	// every promoted site; the computed value on a reuse decline; else 0).
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
// the per-thread reuse gate, and the R4 non-candidate declarations).
// Stable strings: the record, tiling-verify, and the tests pin them.
const (
	// ReasonCTACovered: a covered dim is CTA-assigned — no coherent fill
	// variant exists under the v0 mechanics (R2 tier rule, refusal).
	ReasonCTACovered = "covered dim CTA-assigned: no coherent fill variant (R2 tier rule)"
	// ReasonFragBytes: the per-thread FRAG footprint exceeds
	// MaxFragPerThread (F6).
	ReasonFragBytes = "FRAG bytes exceed MaxFragPerThread (F6)"
	// ReasonLowReuse: per-thread reuse < 2 — the cached bytes are not
	// re-traversed by the owning thread, so no traffic is avoided.
	ReasonLowReuse = "per-thread reuse < 2: no traffic avoided"
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
// re-checks (CTA LoC == MaxCTA, warp-portion multiples, the F2 floor, the
// v0 policy) live in tiling-verify, and a corrupted record must NOT reach
// the rewrite unvalidated. It is the one field tiling-verify mutates on the
// published record (261021-o61 T3; the decide-half surface is otherwise
// untouched).
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
	// CTA-assigned): the flat-init walk's step and the ownership-gate
	// walk's step.
	CoupledThreads int
}

// MappingPlan is one LNT's runtime mapping construction data (LR-11/P-3):
// the GridMapping and ThreadMapping portion vectors. v0 policy: grid is
// always [-1] (every rewritten LNT carries >= 1 CTA dim by the searcher's
// CTA-carrying fallback, and the whole PortionCTA=10 space goes to it);
// thread is [warpPortion, -1] (the independent warp dim's portion, the
// remainder left to the CTA dim's coupled tile-inside walk).
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
	// MaxThreadsPerCTA is the launch thread budget the pass rewrote against:
	// final-lower's reducer staging extent (#warps = MaxThreadsPerCTA/32,
	// ruling D8 2026-10-23) reads it from here so the staging allocation can
	// never disagree with the launch grammar the maps assert against.
	MaxThreadsPerCTA int
}

// Name satisfies analysis.Analysis structurally (the single-method Name()
// interface — the Record precedent): the provenance publishes itself through
// the opt analysis cache under ProvenanceName.
func (Provenance) Name() string { return ProvenanceName }

// ---------------------------------------------------------------------------
// Params: the externally provided thresholds (doc: "the thresholds are
// EXTERNALLY PROVIDED to the pass").
// ---------------------------------------------------------------------------

// The named portion constants of loop_tiling.md ("Named portion constants,
// future adaptation points"): the grammar the launch totals are divided
// into, and the compiler-side mirror of the runtime assert-and-trap
// pre-conditions.
const (
	// PortionCTA is PORTION_CTA: the CTA space is sharded into 10 portions,
	// and the runtime ASSERT-AND-TRAP pre-condition demands the provided CTA
	// count be a multiple of it.
	PortionCTA = 10
	// PortionWarp is PORTION_WARP: the warp space is sharded into 8
	// portions. It is NOT itself a warp count — at the minimum config each
	// portion is exactly ONE warp (the 8-warp / 256-thread minimum).
	PortionWarp = 8
	// AutoWarpPortion is the auto-selected WarpPortion when the LNT's
	// vector carries a CTA dim: the independent warp dim takes half the
	// warp portions (4 of 8) and the CTA dim's coupled tile-inside walk
	// implicitly consumes the remaining 4 — per-LNT exhaustion holds either
	// way (loop_tiling.md "Portion grammar").
	AutoWarpPortion = 4
)

// Params carries the pass thresholds with the final-design defaults
// (loop_tiling.md "LoC concurrency bounds, the FINAL MODEL"). Every field
// is validated by Validate before any computation — external input is
// never trusted (fail-fast at the boundary).
type Params struct {
	// MaxCTA is the total CTA allocation (gridDim.x), a multiple of
	// PortionCTA per the runtime assert-and-trap contract.
	MaxCTA int
	// MaxThreadsPerCTA is the per-CTA thread budget (blockDim.x); MUST be a
	// multiple of WARP (32) whose warp count is a multiple of PortionWarp
	// per the runtime assert-and-trap contract.
	MaxThreadsPerCTA int
	// WarpPortion is the independent warp dim's portion of the warp space;
	// 0 = auto: AutoWarpPortion when the LNT's vector carries a CTA dim
	// (the coupled tile-inside walk needs the remaining portions, >= 1),
	// else PortionWarp (full).
	WarpPortion int
	// MaxCache is the SMEM budget in bytes (the project's 64 KB SMEM
	// budget; only promoted ClassSMEM sites count). The sum is
	// KERNEL-WIDE (R3, ruled 2026-10-21): the decide feasibility, the
	// tiling-verify gate, and the lower's budget resolution all gate the
	// same sum accumulated over ALL LNTs' promoted sites — never a
	// per-LNT one.
	MaxCache int
	// MaxFragPerThread is the per-thread FRAG byte budget (F6, ruled
	// 2026-10-21; default 256 bytes/thread): the SUM of promoted FRAG
	// sites' per-thread bytes, kernel-wide, must stay within it — an
	// unbounded register footprint spills to local memory (doc/tca.md
	// §5.8), strictly worse than the GMEM load it replaced. It gates the
	// judge (a single site over the cap is declined) and the schedule
	// feasibility (the accumulated sum), and participates in the lower's
	// halve/drop fallback.
	MaxFragPerThread int
	// CacheDims is the dimension-level reuse count: only the last
	// CacheDims tensor dimensions may be promoted (hardcoded 2 in the doc).
	CacheDims int
	// MaxCS caps the candidate-set size ("do not consider too many MemLoad
	// to be replaced by cache").
	MaxCS int
	// WindowK is the innermost-K search window depth (K default 5).
	WindowK int
	// TileSizes is the common tile-size search set. Every entry must be
	// >= 2: the compiler never generates a non-tiling schedule (the
	// 2026-10-24 tile-disable ruling), so tile 1 is not a legal schedule.
	TileSizes []int
	// NominalExtent is the documented stand-in extent for a reuse
	// dimension whose BIV max operand is NOT a constant. NOTE (2026-10-21
	// amendment): reuse is PER-THREAD now — the product of the reuse dims'
	// inside-walk trip counts, tile-derived — so extents no longer enter
	// the objective and this knob is currently dormant; it stays on the
	// surface for the future extent-aware ranking models.
	NominalExtent int
}

// DefaultParams returns the final-design defaults (loop_tiling.md "LoC
// concurrency bounds, the FINAL MODEL"): 100 CTAs (Ampere's 108 SMs
// rounded DOWN to the PortionCTA multiple), 512 threads per CTA (= 16
// warps), 64 KB SMEM, 256 bytes/thread FRAG (F6), CacheDims 2, MaxCS 16,
// window K=5, tile sizes {32,64,128,256}, nominal extent 4096, WarpPortion
// auto. The legacy LoC presets and objective weights are gone — the
// portion grammar replaces them.
func DefaultParams() Params {
	return Params{
		MaxCTA:           100,
		MaxThreadsPerCTA: 512,
		MaxCache:         65536,
		MaxFragPerThread: 256,
		CacheDims:        2,
		MaxCS:            16,
		WindowK:          5,
		TileSizes:        []int{32, 64, 128, 256},
		WarpPortion:      0,
		NominalExtent:    4096,
	}
}

// Validate fail-fasts on malformed thresholds. The first two checks are the
// compiler-side mirror of the runtime ASSERT-AND-TRAP pre-conditions
// (loop_tiling.md "Runtime hard pre-condition, ASSERT-AND-TRAP"): MaxCTA a
// positive multiple of PortionCTA, and MaxThreadsPerCTA a multiple of WARP
// (32) whose warp count is a multiple of PortionWarp (8 warps, i.e. the
// 256-thread minimum). WarpPortion must be a legal portion ([0,
// PortionWarp], 0 = auto); the remaining bounds must be positive; the tile
// list must be non-empty with entries >= 2 (tile size 1 is not a legal
// schedule — the compiler never generates a non-tiling schedule, the
// 2026-10-24 ruling; a tile below 2 would
// poison the arithmetic) and, when the auto warp portion leaves a coupled
// inside walk, no entry may exceed that walk's thread count (HI-03 gate 1:
// tile > coupledThreads turns the single-owner flat-init walk into a
// double-writer race).
func (p Params) Validate() error {
	if p.MaxCTA <= 0 || p.MaxCTA%PortionCTA != 0 {
		return fmt.Errorf("tiling: MaxCTA %d must be a positive multiple of PortionCTA %d (the runtime assert-and-trap pre-condition)", p.MaxCTA, PortionCTA)
	}
	if p.MaxThreadsPerCTA <= 0 || p.MaxThreadsPerCTA%32 != 0 || (p.MaxThreadsPerCTA/32)%PortionWarp != 0 {
		return fmt.Errorf("tiling: MaxThreadsPerCTA %d must be a positive multiple of WARP (32) with a warp count that is a multiple of PortionWarp %d (the runtime assert-and-trap pre-condition, 256-thread minimum)", p.MaxThreadsPerCTA, PortionWarp)
	}
	if p.WarpPortion < 0 || p.WarpPortion > PortionWarp {
		return fmt.Errorf("tiling: WarpPortion %d must be within [0, PortionWarp %d] (0 = auto)", p.WarpPortion, PortionWarp)
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
	if p.WindowK < 1 {
		return fmt.Errorf("tiling: WindowK %d must be >= 1", p.WindowK)
	}
	if p.NominalExtent < 1 {
		return fmt.Errorf("tiling: NominalExtent %d must be >= 1", p.NominalExtent)
	}
	if len(p.TileSizes) == 0 {
		return fmt.Errorf("tiling: TileSizes must not be empty")
	}
	for _, v := range p.TileSizes {
		if v < 2 {
			return fmt.Errorf("tiling: TileSizes entry %d is below 2 — tile size 1 is not a legal schedule: the compiler never generates a non-tiling schedule (2026-10-24 ruling); every entry must be >= 2", v)
		}
	}
	// HI-03 gate 1 (2026-10-24 ruling): a CTA dim's tile must not exceed the
	// coupled inside walk's thread count — the flat-init coupled walk
	// (init = the flat thread rank, step = coupledThreads) is single-owner
	// exact only while tile <= coupled; any larger tile makes every element
	// at rank >= coupled walked by TWO threads (a write-write race, not a
	// perf bug). The resolution mirrors lower_rewrite.go coupledThreads():
	// warpPortion auto = AutoWarpPortion, the binding case per the searcher's
	// CTA-carrying fallback. When coupled == 0 (WarpPortion == PortionWarp)
	// the tile comparison is skipped — the F2 floor owns that configuration's
	// error with its clearer message. DefaultParams' max tile 256 == coupled
	// 256 stays valid.
	warpsPerPortion := (p.MaxThreadsPerCTA / 32) / PortionWarp
	warpPortion := p.WarpPortion
	if warpPortion == 0 {
		warpPortion = AutoWarpPortion
	}
	if coupled := (PortionWarp - warpPortion) * warpsPerPortion * 32; coupled > 0 {
		for _, v := range p.TileSizes {
			if v > coupled {
				return fmt.Errorf("tiling: TileSizes entry %d exceeds the CTA-coupled inside walk's thread count %d — the flat-init coupled walk is single-owner exact only while tile <= coupledThreads (the assert-and-trap family; tiling-verify and the tile-lower rewrite re-check this)",
					v, coupled)
			}
		}
	}
	return nil
}
