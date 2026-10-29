package tiling

import (
	"tcacomp/internal/ir"
)

// constraints.go is the SINGLE SOURCE of the fix4 §2.3 constraint
// arithmetic — MAPM (expanded-instance dim caps), MALoC (factorization
// product bounds), and CuMAP (the CTA-dim presence) — shared by the
// feasible-area search (incremental, over PARTIAL vectors: a digit is
// rejected the moment it violates a running bound) and tiling-verify
// (completed-vector re-derivation over the published record). Neither
// side duplicates a formula: the search prunes with the same pred that
// verify re-checks with.
//
// The counters are monotone in the digits placed so far (counts only
// grow, products only multiply by entries >= 2), so checking a partial
// prefix is sound: a prefix violation can never be repaired by later
// digits.

// partState is the running factorization state of one (partial or
// complete) schedule vector: the expanded-instance counts and the
// factorization products (fix4 §2.3, checked over expanded instances and
// factorizations).
type partState struct {
	ctaDims int // #CTA-assigned dims placed (1 CTA instance each)
	thrDims int // #expanded THR instances placed (WARP dims + CTA dims)
	ctaLoC  int // running Π CTA LoC (>= 1; the empty product is 1)
	warpK   int // running Π warp K (>= 1; the empty product is 1)
}

// initialPartState is the empty-vector state (all counters zero, both
// products the empty product 1).
func initialPartState() partState { return partState{ctaLoC: 1, warpK: 1} }

// place returns the state after placing one digit with the given SoC and
// absolute LoC (LoC is ignored for SEQ dims — they carry the implicit 1).
func (s partState) place(soc SoCType, loc int) partState {
	next := s
	switch soc {
	case SoCTA:
		next.ctaDims++
		next.thrDims++
		next.ctaLoC *= loc
	case SoCWARP:
		next.thrDims++
		next.warpK *= loc
	}
	return next
}

// mapmOK reports whether PLACING this digit keeps the partial vector
// inside the MAPM caps (fix4 §2.3 constraint 1: #CTA dims <=
// MaxPartitionCta; #THR instances — WARP dims + CTA dims — <=
// MaxPartitionThr). Checked at placement time: a violating digit rejects
// the whole subtree below it, never a completed vector.
func (s partState) mapmOK(soc SoCType, p Params) bool {
	switch soc {
	case SoCTA:
		return s.ctaDims+1 <= p.MaxPartitionCta && s.thrDims+1 <= p.MaxPartitionThr
	case SoCWARP:
		return s.thrDims+1 <= p.MaxPartitionThr
	default:
		return true // SEQ places no expanded instance
	}
}

// malocOK reports whether PLACING this digit keeps the partial vector
// inside the MALoC bounds (fix4 §2.3 constraint 2: per LNT, Π CTA LoC <=
// MaxCTA and Π warp K <= MaxWarpsPerCTA).
func (s partState) malocOK(soc SoCType, loc int, p Params) bool {
	switch soc {
	case SoCTA:
		return s.ctaLoC*loc <= p.MaxCTA
	case SoCWARP:
		return s.warpK*loc <= p.MaxWarpsPerCTA
	default:
		return true
	}
}

// cumapOK reports the CuMAP constraint (fix4 §2.3 constraint 3): at least
// one CTA-assigned dim. A COMPLETED vector without a CTA dim is skipped
// by the search; an LNT where no CTA-carrying vector exists at all is
// structural infeasibility — a hard error (fix4 §2.6).
func (s partState) cumapOK() bool { return s.ctaDims >= 1 }

// partitionCount is the fix4 §2.5 tiebreak metric: expanded instances per
// the §1.3 table — a CTA dim contributes 1 CTA instance + 1 THR instance,
// a WARP dim 1 THR instance, SEQ none (R-PARTITION: 2×#CTA + #WARP, <= 6
// under MAPM).
func (s partState) partitionCount() int { return 2*s.ctaDims + (s.thrDims - s.ctaDims) }

// locPresetOK reports whether an absolute LoC belongs to its SoC's preset
// list (fix4 §2.7's first re-check): CTA -> CtaLoCPresets, WARP ->
// WarpLoCPresets, SEQ -> the implicit 1.
func locPresetOK(soc SoCType, loc int, p Params) bool {
	switch soc {
	case SoCTA:
		return intIn(loc, p.CtaLoCPresets)
	case SoCWARP:
		return intIn(loc, p.WarpLoCPresets)
	default:
		return loc == 1
	}
}

// intIn reports membership in an int list.
func intIn(v int, xs []int) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

// occupancy is the SM-occupancy factor product of one completed schedule
// (quick-261029-n63, the locked design): the grid the factorization
// launches, the nominal-extent-active portion of it, the per-SM CTA
// residency, and the final multiplicative factor
//
//	factor = (1 − idleFrac) × (ε + (1−ε)·fill)
//
// applied to the post-eviction traffic sum as the search objective's
// score'. Every field is a pin: the unit tests in occupancy_test.go
// carry the derivations.
type occupancy struct {
	grid, activeGrid       int     // launched CTAs; the NominalExtent-active portion
	blockDim               int     // the launch blockDim (blockDimOf, Π warp K × 32)
	ctasPerSM, capacity    int     // resident CTAs per SM; SMs × ctasPerSM
	idleFrac, fill, factor float64 // wasted-grid fraction; active/capacity (clamped ≤ 1); the score' multiplier
}

// occupancyOf is THE single source of the occupancy arithmetic (the
// locked design point: the search and every test/verify re-derivation
// call this, never re-derive the formula). It extracts the CTA dims
// from a completed per-iter decision map, derives the blockDim via
// blockDimOf (judge.go — no second derivation), and delegates to
// occupancyCompute. Callers: the search applies the factor AFTER the
// greedy eviction loop (the eviction itself is occupancy-blind);
// tiling-verify never calls it — legality re-derivation is unchanged by
// construction (occupancy scales selection only).
//
// The CuMAP constraint guarantees ≥ 1 CTA dim on every evaluated
// schedule, so the extracted slice is never empty here; no empty-CTA
// path is built for it.
func occupancyOf(iters map[ir.Node]IterDecision, smemBytes int, p Params) occupancy {
	cta := make([]IterDecision, 0, len(iters))
	for _, d := range iters {
		if d.SoC == SoCTA {
			cta = append(cta, d)
		}
	}
	return occupancyCompute(cta, blockDimOf(iters), smemBytes, p)
}

// occupancyCompute is the formula core over a plain CTA-dim slice — the
// test-facing entry (no IR needed). Formulas, ALL over CTA dims only,
// per the locked design:
//
//	grid        = Π LoC_d
//	activeGrid  = Π_d min(LoC_d, ceil(NominalExtent/T_d))
//	smemPerCTA  = smemBytes + ReservedSmemPerCTA
//	ctasPerSM   = min(WarpsPerSM/(blockDim/32), ThreadsPerSM/blockDim,
//	                  (SmemPerSM−ReservedSmemPerCTA)/smemPerCTA)
//	              — INTEGER (floor) division throughout; the smem term
//	              clamps to 0 when smemPerCTA exceeds its numerator
//	capacity    = SMs × ctasPerSM  (capacity == 0 → fill = 0)
//	idleFrac    = 1 − activeGrid/grid  (per-dim min ⇒ activeGrid ≤ grid,
//	              so idleFrac ≥ 0 by construction)
//	fill        = min(1, activeGrid/capacity)  as float64
//	factor      = (1−idleFrac) × (ε + (1−ε)·fill)
//
// Every division is guarded: the clamps are TOTAL — capacity == 0 makes
// fill 0 and the factor degrades to (1−idleFrac)×ε, never a
// divide-by-zero panic (threat T-n63-02). NominalExtent models every
// CTA dim at the uniform stand-in extent: dynamic shapes carry no
// per-dim extent at compile time.
func occupancyCompute(cta []IterDecision, blockDim, smemBytes int, p Params) occupancy {
	if blockDim < 32 {
		blockDim = 32 // blockDimOf yields ≥ 32 (the empty warp product × 32); a defensive floor, unreachable from the search
	}
	grid, active := 1, 1
	for _, d := range cta {
		grid *= d.LoC
		perDim := ceilDiv(p.NominalExtent, d.TileSize)
		if d.LoC < perDim {
			perDim = d.LoC // min(LoC_d, ceil(N/T_d))
		}
		active *= perDim
	}
	smemPerCTA := smemBytes + p.ReservedSmemPerCTA
	smemTerm := 0 // smemPerCTA > the numerator: no CTA of this size fits — total clamp
	if num := p.SmemPerSM - p.ReservedSmemPerCTA; smemPerCTA <= num {
		smemTerm = num / smemPerCTA
	}
	warpsPerCTA := blockDim / 32
	ctasPerSM := min(p.WarpsPerSM/warpsPerCTA, p.ThreadsPerSM/blockDim, smemTerm)
	capacity := p.SMs * ctasPerSM
	fill := 0.0 // capacity == 0: the clamped, total fill — never a division
	if capacity > 0 {
		fill = float64(active) / float64(capacity)
		if fill > 1 {
			fill = 1
		}
	}
	idleFrac := 1.0 - float64(active)/float64(grid)
	return occupancy{
		grid:       grid,
		activeGrid: active,
		blockDim:   blockDim,
		ctasPerSM:  ctasPerSM,
		capacity:   capacity,
		idleFrac:   idleFrac,
		fill:       fill,
		factor:     (1.0 - idleFrac) * (p.OccEps + (1.0-p.OccEps)*fill),
	}
}
