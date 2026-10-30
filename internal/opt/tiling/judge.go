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
//   - FRAG PRICING (fix6 §3, quick-261030-v62 — the honest register-block
//     model): a FRAG site's RAW per-thread window (siteWindowBytesRaw —
//     Π T_d × sizeof over the covered dims) is priced as a STREAMED
//     REGISTER BLOCK for the budget: its budget footprint is
//
//        blockFootprint = min(rawWindow, MaxFragPerThread)
//
//     (the window's peak live per-thread bytes under budget-sized
//     streaming — one chunk of at most the whole budget lives in
//     registers at a time). The per-site F6 size DECLINE is REMOVED: an
//     over-budget window promotes when its traffic gate passes, exactly
//     like every other class. The TRAFFIC term keeps pricing the RAW
//     window: traffic = rawWindow × (reuse − 1) × holders, no cap — the
//     one-currency avoided-GMEM model is not weakened (the block pricing
//     must not reward shrinking the fill). The kernel-wide budget still
//     binds on the block footprints: Σ blockFootprint over the kernel's
//     promoted FRAG sites vs MaxFragPerThread spills by value density
//     (ReasonEvicted, fix4 §2.6) — never a per-site decline, never a
//     failure.
//   - GATES: traffic avoided (rawBytes × (reuse − 1) × holders, all sites
//     in the one CTA-wide currency — equivalently reads-served minus
//     elements-refilled per fill epoch) > 0 — this also fails closed a
//     zero-byte site. Fill cost is therefore visible to the gate and the
//     eviction density both.

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
	// bytes is the BUDGET footprint (fix6 §3): the SMEM window bytes, or
	// for FRAG the register-block footprint min(rawWindow,
	// MaxFragPerThread). It feeds the search's eviction sum, tiling-
	// verify's kernel-wide re-check, and the lower's budget resolution —
	// the ONE shared pricing source (siteWindowBytes).
	bytes int
	// rawBytes is the TRAFFIC footprint (fix6 §3): the RAW factor-model
	// window — sizeof × Π covered factors, no budget transform. The
	// objective's traffic term (rawBytes × (reuse − 1) × holders) prices
	// it uncapped; == bytes for SMEM sites and at-or-under-budget FRAG
	// sites.
	rawBytes int
	// holders is the currency scale (fix4 §2.4 amendment): the threads
	// that fill a window per epoch — 1 for a collectively-filled SMEM
	// window, the LNT blockDim for a per-thread FRAG window. Multiplied
	// into traffic (rawBytes × (reuse − 1) × holders) it puts every site
	// class in the ONE CTA-wide GMEM-byte currency.
	holders int
	reason  string
}

// judgeSite is the fix4 §2.4 promotion judge. The site is classified by
// the SoC of its COVERED dims (the cached trailing dimensions; rank >=
// CacheDims is guaranteed by the admission filter), priced by the shared
// siteWindowBytesRaw (traffic) / siteWindowBytes (budget), and gated on
// traffic avoided — there is NO per-site FRAG size decline (fix6 §3: an
// over-budget window prices as a streamed register block and promotes
// when its traffic gate passes, like every other class; the kernel-wide
// budget binds later, by eviction). An iter with NO decision, or a
// covered iter at tile < 2, fails closed (ReasonUnknownIter) — never
// scored.
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
	// fix6 §3 (quick-261030-v62): NO per-site FRAG size decline — the raw
	// window is priced, never declined on size. rawBytes is the traffic
	// footprint (uncapped); bytes is the budget footprint (the streamed
	// register block: min(raw, MaxFragPerThread) for FRAG, the window
	// itself for SMEM) — both from the ONE shared source.
	rawBytes := siteWindowBytesRaw(sf, iters, class)
	bytes := siteWindowBytes(sf, iters, class, p)
	reuse := siteReuseCount(sf, iters, ctx, class)
	// One-currency promotion gate (§2.4): traffic avoided in the CTA-wide
	// currency must be positive. This subsumes the old reuse < 2 gate AND
	// fails closed a zero-byte site (an unavailable element type prices 0
	// bytes — no traffic claim). A fail-closed reuse 0 makes the product
	// negative, declining the site the same way.
	if rawBytes*(reuse-1)*holders <= 0 {
		return siteJudgment{class: class, holders: holders, reuse: reuse, bytes: bytes, rawBytes: rawBytes, reason: ReasonLowReuse}
	}
	return siteJudgment{promoted: true, class: class, holders: holders, reuse: reuse, bytes: bytes, rawBytes: rawBytes}
}

// siteWindowBytesRaw is the site's RAW factor-model footprint (fix4 §2.4):
// sizeof(dtype) × Π over the covered dims of the dim's factor,
//
//   - SMEM window class: WARP -> K_d × T_d (one window per warp
//     coordinate), SEQ -> T_d, CTA -> LoC_d × T_d (diagnostic only —
//     CTA-covered sites are refused by the tier rule; the value stays
//     well-defined),
//   - FRAG class: SEQ -> T_d (the thread walks the dim wholly), WARP
//     under C1b -> ceil(T_d/32) (the lane's consecutive chunk) — NO other
//     lane division.
//
// This is the TRAFFIC footprint (the fill/buffer size the lower allocates
// — its per-dim buffer extents use exactly these factors) and the SMEM
// budget number. An unavailable element type reports 0 (fail closed:
// contributes no traffic).
func siteWindowBytesRaw(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass) int {
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

// siteWindowBytes is the site's BUDGET footprint — the ONE bytes function
// the search's eviction sum, tiling-verify's kernel-wide cross-check, and
// the lower's budget resolution share (decide and lower MUST agree, so
// neither side carries its own arithmetic). For SMEM sites it is the
// window itself (siteWindowBytesRaw). For FRAG sites it is the streamed
// register-block footprint under the fix6 §3 ruling (quick-261030-v62):
//
//	blockFootprint = min(siteWindowBytesRaw, MaxFragPerThread)
//
// the window's peak live per-thread bytes under budget-sized streaming.
// The over-budget remainder still fills (and its RAW bytes price the
// traffic term via siteJudgment.rawBytes — the one-currency avoided-GMEM
// model is not weakened); only the BUDGET accounting chunks it. The
// kernel-wide Σ blockFootprint vs MaxFragPerThread binds by eviction
// (fix4 §2.6), never by a per-site decline.
func siteWindowBytes(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass, p Params) int {
	raw := siteWindowBytesRaw(sf, iters, class)
	if class != ClassFRAG {
		return raw
	}
	return min(raw, p.MaxFragPerThread)
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

// reducedTierEff is the R4 tier-efficiency factor (fix5 §1 R4): for each
// of the site's COVERED dims that is reduced (reducer-consuming) AND
// WARP-assigned, multiply min(1, effExtent_k / (32 · K_k)) into the
// product; every other covered dim contributes exactly 1. A warp-sharded
// reduced dim whose effective extent cannot feed its lanes earns no full
// traffic credit for the fills it fragments — the minimal objective
// amendment that lets the search CHOOSE K=SEQ on small-K kernels (without
// it the collapse vectors keep losing to the warp-ceremony vectors).
// effExtent_k is the dim's const extent or the NominalExtent stand-in —
// the SAME effective-extent discipline as R1's adaptiveLoC.
//
// Selection-only (the n63 occupancy precedent): evalSchedule folds this
// into the float64 score' currency ONLY — the int traffic pin, the judge,
// the eviction, and tiling-verify stay R4-blind.
func reducedTierEff(reduced map[ir.Node]bool, iters map[ir.Node]IterDecision, covered []ir.Node, p Params) float64 {
	eff := 1.0
	for _, c := range covered {
		if !reduced[c] {
			continue // R4 guards reduced dims only
		}
		d, ok := iters[c]
		if !ok || d.SoC != SoCWARP {
			continue // SEQ/CTA-covered reduced dims fragment nothing
		}
		extent, ok := constExtentOf(c)
		if !ok {
			extent = p.NominalExtent // dynamic-extent stand-in (feeds every lane)
		}
		if e := float64(extent) / float64(32*d.LoC); e < 1.0 {
			eff *= e // min(1, ·): a fed lane army contributes exactly 1
		}
	}
	return eff
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
