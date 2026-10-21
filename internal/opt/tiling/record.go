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
type SiteDecision struct {
	Promoted bool
	Tie      ir.Node
	Class    CacheClass
}

// Record is the tiling decision side table (loop_tile_lower.md §1):
// per-Iter entries keyed by BIV node identity — the Iter's operand 0,
// GVN-canonical so a later fusion re-run deleting Iter prefix nodes cannot
// orphan keys — and per-site entries keyed by the MemLoad node id
// (BasePtr().ID()).
//
// A MISSING Iters entry means the iter was not part of any forest tree the
// pass saw; a MISSING Sites entry means the load was not admitted to the
// candidate set — the lower pass treats the load as untouched.
type Record struct {
	Iters map[ir.Node]IterDecision
	Sites map[uint64]SiteDecision
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
	// budget; only promoted ClassSMEM sites count — FRAG sites are
	// register-budgeted by the lower pass).
	MaxCache int
	// CacheDims is the dimension-level reuse count: only the last
	// CacheDims tensor dimensions may be promoted (hardcoded 2 in the doc).
	CacheDims int
	// MaxCS caps the candidate-set size ("do not consider too many MemLoad
	// to be replaced by cache").
	MaxCS int
	// WindowK is the innermost-K search window depth (K default 5).
	WindowK int
	// TileSizes is the common tile-size search set; the uniform sweep also
	// includes t=1 (tiling disabled).
	TileSizes []int
	// NominalExtent is the documented stand-in extent for a reuse dimension
	// whose BIV max operand is NOT a constant: the doc demands static
	// arithmetic, and real tensors bound by TensorDimOf have symbolic
	// extents, so a symbolic dim contributes the nominal factor instead.
	// The knob keeps ranking meaningful when extents are unknown; a
	// constant max is always preferred when present.
	NominalExtent int
}

// DefaultParams returns the final-design defaults (loop_tiling.md "LoC
// concurrency bounds, the FINAL MODEL"): 100 CTAs (Ampere's 108 SMs
// rounded DOWN to the PortionCTA multiple), 512 threads per CTA (= 16
// warps), 64 KB SMEM, CacheDims 2, MaxCS 16, window K=5, tile sizes
// {32,64,128,256}, nominal extent 4096, WarpPortion auto. The legacy LoC
// presets and objective weights are gone — the portion grammar replaces
// them.
func DefaultParams() Params {
	return Params{
		MaxCTA:           100,
		MaxThreadsPerCTA: 512,
		MaxCache:         65536,
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
// list must be non-empty with positive entries (a non-positive tile would
// poison the arithmetic).
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
		if v < 1 {
			return fmt.Errorf("tiling: TileSizes contains non-positive entry %d", v)
		}
	}
	return nil
}
