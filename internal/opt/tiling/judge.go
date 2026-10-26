package tiling

// judge.go is the fix4 §2.4 COST MODEL — the site promotion judge, the
// ONE factor-model window-bytes function (shared by the search,
// tiling-verify's judge cross-check, and the lower's budget resolution —
// fail-closed agreement by construction), and the lane-aware reuse
// accounting. Split out of search.go so the judge reads as the spec's
// §2.4 checklist; the signatures carry over with the per-LNT blockDim
// context replacing the retired share-grammar context.
//
//   - TIERS (§2.4, in order): (1) any covered dim CTA-assigned -> REFUSED
//     (ReasonCTACovered); (2) C1b: any non-covered referenced iter (INS)
//     CTA- or WARP-assigned -> forced FRAG (a thread-varying source
//     coordinate at the fill point); (3) any covered dim WARP-assigned ->
//     SMEM window class; (4) all covered dims SEQ -> FRAG class.
//   - BYTES (§2.4 + R-C1B): SMEM window = Π over covered dims of
//     (WARP -> K_d × T_d, SEQ -> T_d) × sizeof — the K_d×T_d window is
//     32x smaller than the old whole-group shard window, so rhs-class
//     sites fit. FRAG per thread = Π(covered factors: SEQ -> T_d, WARP
//     under C1b -> ceil(T_d/32) — the lane's chunk) × sizeof, with NO
//     other lane division.
//   - REUSE (§2.4, 2026-10-26 one-currency amendment): counted at the
//     window's OWN granularity so that bytes × (reuse − 1) is CTA-wide
//     GMEM bytes avoided for EVERY site class. SMEM windows count
//     COLLECTIVE per-window-instance trips: CTA -> T_d (the
//     ceil(T/blockDim) per-thread trips × blockDim readers cancel), WARP
//     -> T_d (the lanes' consecutive chunks sum to the tile), SEQ ->
//     T_d × readers (readers = 32 for a warp-sliced window — always true
//     for SMEM under the current tiers — blockDim defensively for a
//     CTA-shared one), × the cross-warp sharing factor (Π K_d over
//     WARP-assigned dims NOT in the load's subscript set — warps
//     differing only on such coordinates read the SAME window instance).
//     FRAG sites keep the PER-THREAD trips (CTA -> ceil(T_d/blockDim),
//     WARP -> ceil(T_d/32), SEQ -> T_d; sharing 1 — registers are
//     private) and their traffic is scaled by HOLDERS (the LNT
//     blockDim — threads that fill a window per epoch) in the objective,
//     so both classes compare in one currency. The pre-amendment form
//     multiplied per-window bytes by per-thread trips, rewarding
//     warp-splits that shrink blockDim without moving wire bytes — that
//     phantom term is gone.
//   - GATES: traffic avoided (bytes × (reuse − 1) × holders, all sites in
//     the one CTA-wide currency — equivalently reads-served minus
//     elements-refilled per fill epoch) > 0 — this also fails closed a
//     zero-byte site; and a FRAG site's per-thread bytes <=
//     MaxFragPerThread (F6). Fill cost is therefore visible to the gate
//     and the eviction density both.

import (
	"tcacomp/internal/ir"
)

// blockDimCtx is the per-LNT launch context the judge accounts against:
// the LNT's blockDim = (Π warp K_d) × 32 under the fix4 factorization
// model (§1.2: the CTA dim's coupled inside walk strides over ALL
// blockDim threads; coupledThreads := blockDim). It replaces the retired
// share-grammar context, which carried only portion arithmetic.
type blockDimCtx struct {
	blockDim int
}

// blockDimOf derives the per-LNT blockDim from a decision map: the
// product of the WARP-assigned dims' warp counts × 32 (the empty product
// is 1 — an LNT with no WARP dim launches one warp).
func blockDimOf(decisions map[ir.Node]IterDecision) int {
	k := 1
	for _, d := range decisions {
		if d.SoC == SoCWARP {
			k *= d.LoC
		}
	}
	return k * 32
}

// siteJudgment is the promotion judge's product for one site.
type siteJudgment struct {
	promoted bool
	class    CacheClass
	reuse    int // re-traversal count within one fill epoch (§2.4: collective for SMEM, per-thread for FRAG)
	bytes    int // factor-model window footprint
	// holders is the currency scale (fix4 §2.4 amendment): the threads
	// that fill a window per epoch — 1 for a collectively-filled SMEM
	// window, the LNT blockDim for a per-thread FRAG window. Multiplied
	// into traffic (bytes × (reuse − 1) × holders) it puts every site
	// class in the ONE CTA-wide GMEM-byte currency.
	holders int
	reason  string
}

// judgeSite is the fix4 §2.4 promotion judge. The site is classified by
// the SoC of its COVERED dims (the cached trailing dimensions; rank >=
// CacheDims is guaranteed by the admission filter), priced by the shared
// siteWindowBytes, and gated on traffic avoided and the FRAG budget. An
// iter with NO decision, or a covered iter at tile < 2, fails closed
// (ReasonUnknownIter) — never scored.
func judgeSite(sf SiteFacts, iters map[ir.Node]IterDecision, ctx blockDimCtx, p Params) siteJudgment {
	if len(sf.Covered) == 0 {
		return siteJudgment{reason: ReasonEmptyCovered}
	}
	coveredSet := make(map[ir.Node]bool, len(sf.Covered))
	for _, c := range sf.Covered {
		coveredSet[c] = true
		d, ok := iters[c]
		if !ok {
			return siteJudgment{reason: ReasonUnknownIter} // unknown assignment: fail closed
		}
		if d.SoC == SoCTA {
			return siteJudgment{reason: ReasonCTACovered} // tier 1: covered-CTA refusal (R2)
		}
		if d.TileSize < 2 {
			return siteJudgment{reason: ReasonUnknownIter} // invalid tile: fail closed
		}
	}
	forcedFrag := false
	for _, it := range sf.INS {
		if coveredSet[it] {
			continue
		}
		d, ok := iters[it]
		if !ok {
			return siteJudgment{reason: ReasonUnknownIter} // unknown assignment: fail closed
		}
		if d.SoC == SoCTA || d.SoC == SoCWARP {
			forcedFrag = true // tier 2 (C1b): a thread-varying non-covered coordinate
			break
		}
	}
	class := ClassFRAG
	if !forcedFrag {
		for _, c := range sf.Covered {
			if iters[c].SoC == SoCWARP {
				class = ClassSMEM // tier 3: the SMEM window class
				break
			}
		}
	}
	// Holders (§2.4 amendment), set the moment the class exists so every
	// downstream return path carries a coherent judgment: an SMEM window
	// is filled collectively once (holders 1); a FRAG window is filled by
	// each of the LNT's blockDim threads privately.
	holders := 1
	if class == ClassFRAG {
		holders = ctx.blockDim
	}
	bytes := siteWindowBytes(sf, iters, class)
	if class == ClassFRAG && bytes > p.MaxFragPerThread {
		return siteJudgment{class: class, holders: holders, bytes: bytes, reason: ReasonFragBytes}
	}
	reuse := siteReuseCount(sf, iters, ctx, class)
	// One-currency promotion gate (§2.4): traffic avoided in the CTA-wide
	// currency must be positive. This subsumes the old reuse < 2 gate AND
	// fails closed a zero-byte site (an unavailable element type prices 0
	// bytes — no traffic claim). A fail-closed reuse 0 makes the product
	// negative, declining the site the same way.
	if bytes*(reuse-1)*holders <= 0 {
		return siteJudgment{class: class, holders: holders, reuse: reuse, bytes: bytes, reason: ReasonLowReuse}
	}
	return siteJudgment{promoted: true, class: class, holders: holders, reuse: reuse, bytes: bytes}
}

// siteWindowBytes is the site's factor-model footprint (fix4 §2.4, the
// ONE bytes function the search, tiling-verify's cross-check, and the
// lower's budget resolution share — decide and lower MUST agree, so
// neither side carries its own arithmetic): sizeof(dtype) × Π over the
// covered dims of the dim's factor,
//
//   - SMEM window class: WARP -> K_d × T_d (one window per warp
//     coordinate), SEQ -> T_d, CTA -> LoC_d × T_d (diagnostic only —
//     CTA-covered sites are refused by the tier rule; the value stays
//     well-defined),
//   - FRAG class: SEQ -> T_d (the thread walks the dim wholly), WARP
//     under C1b -> ceil(T_d/32) (the lane's consecutive chunk) — NO other
//     lane division.
//
// An unavailable element type reports 0 (fail closed: contributes no
// traffic).
func siteWindowBytes(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass) int {
	e, ok := elemTypeOf(sf.Load)
	if !ok {
		return 0
	}
	bytes := elemBytes(e)
	for _, c := range sf.Covered {
		d, ok := iters[c]
		if !ok || d.TileSize < 1 {
			return 0
		}
		switch {
		case d.SoC == SoCWARP && class == ClassSMEM:
			bytes *= d.LoC * d.TileSize // K_d windows of T_d each
		case d.SoC == SoCWARP:
			bytes *= ceilDiv(d.TileSize, 32) // C1b: the lane's chunk — NOT × T
		case d.SoC == SoCTA:
			bytes *= d.LoC * d.TileSize // diagnostic (refused sites)
		default: // SEQ: the thread walks the dim wholly
			bytes *= d.TileSize
		}
	}
	return bytes
}

// siteReuseCount is the fix4 §2.4 reuse factor under the 2026-10-26
// one-currency amendment: the product over the site's precomputed reuse
// dims ({tie} plus the LNT iters strictly inner than the tie, minus
// covered), counted at the window's OWN granularity per its class —
//
//   - ClassSMEM — COLLECTIVE, per window instance: CTA dim -> T_d (the
//     ceil(T/blockDim) per-thread trips × the blockDim readers cancel —
//     the tile is covered exactly once; the phantom warp-split reward of
//     the pre-amendment model dies here), WARP dim -> T_d (the lanes'
//     consecutive chunks sum to the tile), SEQ dim -> T_d × readers
//     (readers = 32 for a warp-sliced window — always true for SMEM under
//     the current tier rules, since tier 3 requires a WARP-covered dim —
//     and ctx.blockDim for a CTA-shared window; the defensive branch is
//     unreachable under current tiers but stays well-defined),
//   - ClassFRAG — PER-THREAD (the objective scales by holders instead):
//     SEQ -> T_d, CTA -> ceil(T_d / blockDim) (the coupled flat-strided
//     walk), WARP -> ceil(T_d / 32) (the owning lane's consecutive
//     chunk); sharing = 1 (the window lives in one thread's registers).
//
// The SMEM class then multiplies the cross-warp sharing factor: Π K_d
// over WARP-assigned dims NOT in the load's subscript set (warps
// differing only on such coordinates read the SAME window instance) —
// UNCHANGED by the amendment and EXACT under the factorization.
func siteReuseCount(sf SiteFacts, iters map[ir.Node]IterDecision, ctx blockDimCtx, class CacheClass) int {
	reuse := 1
	if class == ClassSMEM {
		// Readers per window instance: a warp-sliced window (any covered
		// dim WARP-assigned — always true for SMEM under the current tier
		// rules) is re-read by its owning warp's 32 lanes; the CTA-shared
		// fallback is defensive only.
		readers := 32
		warpSliced := false
		for _, c := range sf.Covered {
			if d, ok := iters[c]; ok && d.SoC == SoCWARP {
				warpSliced = true
				break
			}
		}
		if !warpSliced {
			readers = ctx.blockDim
		}
		for _, it := range sf.ReuseDims {
			d, ok := iters[it]
			if !ok || d.TileSize < 1 {
				return 0 // unknown dim: fail closed (no reuse claim)
			}
			switch d.SoC {
			case SoCSequential:
				reuse *= d.TileSize * readers
			default: // CTA, WARP: collective trips = the tile itself
				reuse *= d.TileSize
			}
		}
	} else {
		for _, it := range sf.ReuseDims {
			d, ok := iters[it]
			if !ok || d.TileSize < 1 {
				return 0 // unknown dim: fail closed (no reuse claim)
			}
			trips := d.TileSize
			switch d.SoC {
			case SoCTA:
				trips = ceilDiv(d.TileSize, ctx.blockDim)
			case SoCWARP:
				trips = ceilDiv(d.TileSize, 32)
			}
			reuse *= trips
		}
	}
	if class == ClassSMEM {
		// Cross-warp sharing: Π K_d over the LNT's WARP-assigned dims NOT
		// in the load's subscript set (warps differing only on such
		// coordinates read the SAME window instance). The decision map is
		// the FULL per-LNT map — an unreferenced WARP dim shares the
		// window too, so the INS slice alone would undercount. A product
		// over map entries is order-independent, so determinism holds.
		insSet := sf.INSSet()
		for it, d := range iters {
			if d.SoC == SoCWARP && !insSet[it] {
				reuse *= d.LoC
			}
		}
	}
	return reuse
}

// ceilDiv is the ceiling of a/b over positive integers.
func ceilDiv(a, b int) int {
	if b < 1 {
		return 1
	}
	if a < 1 {
		return 1
	}
	return (a + b - 1) / b
}
