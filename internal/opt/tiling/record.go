// Package tiling implements the DECIDE half of the loop-tiling pair per
// doc/optana/loop_tiling.md (as revised by rulings F1/F4/F5/F8/F9 and
// 261020-fd1) plus the record interface of doc/optana/loop_tile_lower.md §1:
// the MemLoad overlap analysis, the parallelization-schedule (PSched)
// search, the SMEM/FRAG cache-promotion decisions, and the decision-record
// side table the future loop-tile-lower pass consumes.
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

// Params carries the pass thresholds with the doc-mandated defaults
// (quick-261020-ve6). Every field is validated by Validate before any
// computation — external input is never trusted (fail-fast at the
// boundary).
type Params struct {
	// MaxCTA is the total CTA allocation bound (gridDim.x).
	MaxCTA int
	// MaxThreadsPerCTA is the per-CTA thread budget (blockDim.x); MUST be a
	// multiple of WARP (32) per the doc.
	MaxThreadsPerCTA int
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
	// CTALoCs / WarpLoCs are the LoC presets the greedy completion draws
	// from; every list must contain 1 (the always-feasible fallback).
	CTALoCs  []int
	WarpLoCs []int
	// TileSizes is the common tile-size search set; the uniform sweep also
	// includes t=1 (tiling disabled).
	TileSizes []int
	// ParallelWeight / ReuseWeight weight the objective
	// score = ParallelWeight*ΣLoC + ReuseWeight*trafficAvoided.
	ParallelWeight float64
	ReuseWeight    float64
	// NominalExtent is the documented stand-in extent for a reuse dimension
	// whose BIV max operand is NOT a constant: the doc demands static
	// arithmetic, and real tensors bound by TensorDimOf have symbolic
	// extents, so a symbolic dim contributes the nominal factor instead.
	// The knob keeps ranking meaningful when extents are unknown; a
	// constant max is always preferred when present.
	NominalExtent int
}

// DefaultParams returns the doc-pinned defaults: 65535 CTAs, 1024 threads
// per CTA, 64 KB SMEM, CacheDims 2, MaxCS 16, window K=5, CTA LoC presets
// {1,2,4,8,16}, warp LoC presets {1,2,4,8}, tile sizes {32,64,128,256},
// unit weights, nominal extent 4096.
func DefaultParams() Params {
	return Params{
		MaxCTA:           65535,
		MaxThreadsPerCTA: 1024,
		MaxCache:         65536,
		CacheDims:        2,
		MaxCS:            16,
		WindowK:          5,
		CTALoCs:          []int{1, 2, 4, 8, 16},
		WarpLoCs:         []int{1, 2, 4, 8},
		TileSizes:        []int{32, 64, 128, 256},
		ParallelWeight:   1,
		ReuseWeight:      1,
		NominalExtent:    4096,
	}
}

// Validate fail-fasts on malformed thresholds: MaxThreadsPerCTA not a
// positive multiple of WARP (32, the doc's "MUST be a multiple of WARP"),
// any non-positive bound (MaxCTA, MaxCache, CacheDims, MaxCS, WindowK,
// NominalExtent), non-positive weights, and preset/tile lists that are
// empty or contain a non-positive entry (the greedy completion's fallback
// of 1 must always be available, and a non-positive LoC or tile would
// poison the arithmetic).
func (p Params) Validate() error {
	if p.MaxThreadsPerCTA <= 0 || p.MaxThreadsPerCTA%32 != 0 {
		return fmt.Errorf("tiling: MaxThreadsPerCTA %d must be a positive multiple of WARP (32)", p.MaxThreadsPerCTA)
	}
	if p.MaxCTA < 1 {
		return fmt.Errorf("tiling: MaxCTA %d must be >= 1", p.MaxCTA)
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
	if p.ParallelWeight <= 0 {
		return fmt.Errorf("tiling: ParallelWeight %v must be positive", p.ParallelWeight)
	}
	if p.ReuseWeight <= 0 {
		return fmt.Errorf("tiling: ReuseWeight %v must be positive", p.ReuseWeight)
	}
	for name, list := range map[string][]int{"CTALoCs": p.CTALoCs, "WarpLoCs": p.WarpLoCs, "TileSizes": p.TileSizes} {
		if len(list) == 0 {
			return fmt.Errorf("tiling: %s must not be empty", name)
		}
		for _, v := range list {
			if v < 1 {
				return fmt.Errorf("tiling: %s contains non-positive entry %d", name, v)
			}
		}
	}
	return nil
}
