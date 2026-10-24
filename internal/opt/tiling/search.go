// search.go is the parallelization-schedule (PSched) machinery of
// doc/optana/loop_tiling.md ("Assignment of Parallisation" + "Algorithm to
// solve the above formentioned optimization problem"): the innermost-K
// search window, the flat 3^K x tiles enumeration, the portion-grammar LoC
// completion, the feasibility gates, the promotion judge, and the
// single-term traffic-avoided objective.
//
// Documented interpretations (each cited where it fires):
//   - Spine window: "the INNERMOST K consecutive loops" of a TREE is the
//     last K levels of the root -> deepest-leaf spine (deepest by Depth,
//     ties by allocation order). Branched siblings stay sequential —
//     conservative and deterministic.
//   - Uniform tile sweep: the doc's "common searching size for tile_size"
//     is read as ONE uniform knob t applied to every window level, swept
//     over TileSizes only — every entry >= 2 (tile size 1 is not a legal
//     schedule: the compiler never generates a non-tiling schedule, the
//     2026-10-24 ruling). Per-cache-count heuristic tile sets are future
//     work. This keeps the doc's 3^K tractability claim literally true:
//     total <= 3^K * |TileSizes| <= 972 schedules per LNT at defaults (the
//     v0 policy filter only skips more of them).
//   - Portion completion replaces the legacy greedy LoC presets: under the
//     v0 policy there is at most one CTA dim (portion -1/full, LoC =
//     MaxCTA) and at most one independent warp dim (LoC = warp portion x
//     warpsPerPortion); the CTA dim's coupled tile-inside walk consumes
//     the REMAINING warp portions — never fewer than one (the F2 floor
//     below), its stride derived by the lower pass, not recorded.
//   - v0 POLICY (loop_tiling.md "Composition Semantics, CTA x WARP", the
//     "v0 default searcher POLICY" bullet): at most ONE CTA-assigned dim
//     and at most ONE independent WARP dim per LNT. Same-category
//     multi-dim composition (CTA+CTA, WARP+WARP) is FUTURE pending the
//     combined-slicing arithmetic, so vectors violating the policy are
//     skipped outright.
//   - COUPLED-PORTION FLOOR (F2, ruled 2026-10-21): whenever the LNT
//     carries a CTA dim, the effective independent WarpPortion must leave
//     >= 1 warp portion for the CTA dim's coupled inside walk — a
//     zero-portion coupled walk emits a zero-stride inside loop (an
//     infinite loop), so evalSchedule ERRORS on the violation instead of
//     ever selecting such a schedule ("the coupled walk implicitly
//     consumes the remaining >= 1" is a CONSTRAINT, never an assumption).
//   - WINDOW-BYTES feasibility (R2, ruled 2026-10-21 — rank-in-index =>
//     window-in-shape): a promoted site's footprint is element size x the
//     product over its covered dims of (alloc_d x tile_d) — WARP dims
//     contribute the whole group's round window (#threads_d x T), SEQ
//     dims contribute T, CTA dims the CTA group's window (nCTA x T;
//     CTA-covered sites are REFUSED by the tier rule regardless).
//   - PER-THREAD reuse (ruled 2026-10-21): reuse_count is re-traversal by
//     the OWNING thread within one fill epoch — the product over the
//     reuse dims (the tie level and the inner non-covered loop dims) of
//     the thread's own trip count of that dim's INSIDE walk: T_d for a
//     sequential or warp-owned chunk, ceil(T_d / coupledThreads) for a
//     CTA dim's coupled element walk. Re-traversal by OTHER threads never
//     counts; a site with reuse < 2 is NOT promoted.
//   - KERNEL-WIDE budgets (R3 + F6, ruled 2026-10-21): the SMEM sum and
//     the per-thread FRAG sum accumulate ACROSS LNTs — the caller threads
//     the running totals (smemFloor/fragFloor) through searchSchedule, so
//     an earlier LNT's promotions constrain later LNTs' feasibility; the
//     gates here, tiling-verify, and the lower's budget resolution all
//     gate the SAME kernel-wide sums.
//   - Single-term objective (loop_tiling.md "Objective Function ...,
//     SINGLE-TERM"): score = traffic avoided; the parallelism term is
//     deleted — under mandatory per-LNT exhaustion total parallelism is
//     CONSTANT, so it carries no ranking signal.
package tiling

import (
	"fmt"
	"sort"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// schedResult is one LNT's search product: per-iter decisions (keyed by the
// ITER node — the pass converts to BIV keys at record assembly), per-site
// decisions keyed by MemLoad node id, the chosen schedule's score, and the
// chosen schedule's kernel-wide budget consumption (what the caller adds to
// the running floors for the next LNT).
type schedResult struct {
	iters map[ir.Node]IterDecision
	sites map[uint64]SiteDecision
	score float64
	smem  int // the chosen schedule's promoted-SMEM window bytes
	frag  int // the chosen schedule's promoted-FRAG per-thread bytes
}

// schedCtx is the per-schedule portion context the judge, the window-bytes
// model, and the per-thread reuse accounting share.
type schedCtx struct {
	// warpsPerPortion is the warp quantum: total warps / PortionWarp.
	warpsPerPortion int
	// warpPortion is the EFFECTIVE independent warp portion (explicit
	// Params.WarpPortion, or the auto rule: AutoWarpPortion when the
	// vector carries a CTA dim, else PortionWarp).
	warpPortion int
	// coupledThreads is the CTA dim's coupled inside-walk thread count:
	// the remaining warp portions x quantum x 32 (0 when no CTA dim).
	coupledThreads int
}

// searchSchedule enumerates the PSched space of one LNT and returns the
// best schedule under the objective, gated by the KERNEL-WIDE budget floors
// already consumed by earlier LNTs (R3/F6: feasibility accumulates across
// LNTs — the caller threads the running totals through). Enumeration order
// is fixed — SoC odometer order (the outermost window level is the most
// significant digit), tiles ascending within each vector — and selection
// keeps the FIRST strict maximum, so the result is deterministic (INV).
func searchSchedule(lf lntFacts, p Params, smemFloor, fragFloor int) (schedResult, error) {
	window := windowOf(lf, p)
	soc := make([]SoCType, len(window))
	var best schedResult
	found := false
	for {
		// Feasibility gates: >= 1 CTA dim ("an assignment is feasible when
		// it at least has one CTA assignment") AND the v0 policy (at most
		// one CTA dim, at most one independent warp dim — loop_tiling.md
		// "Composition Semantics": same-category multi-dim is FUTURE
		// pending the combined-slicing arithmetic). The odometer keeps
		// running over all 3^K and skips violators — simpler than pruning
		// the enumeration.
		if hasCTALevel(soc) && v0PolicyClean(soc) {
			for _, tile := range tileCandidates(p) {
				res, ok, err := evalSchedule(lf, window, soc, tile, p, smemFloor, fragFloor)
				if err != nil {
					return schedResult{}, err // e.g. the F2 coupled-portion floor
				}
				if !ok {
					continue // infeasible (kernel-wide budget): skipped
				}
				if !found || res.score > best.score {
					best, found = res, true
				}
			}
		}
		// Odometer increment: the innermost level's digit varies fastest.
		i := len(soc) - 1
		for i >= 0 {
			soc[i]++
			if soc[i] <= SoCWARP {
				break
			}
			soc[i] = SoCSequential
			i--
		}
		if i < 0 {
			break
		}
	}
	if !found {
		// The 2026-10-24 tile-disable ruling deleted the t=1 fallback: an
		// LNT with no feasible tile >= 2 schedule under the kernel-wide
		// budgets is a configuration/schedule ERROR, never an un-tiled
		// schedule. (The former t=1 fallback point — a CTA-carrying vector
		// at tile 1, budget-feasible because nothing promotes — is gone:
		// tiling is never disabled.)
		return schedResult{}, fmt.Errorf(
			"tiling: LNT rooted at iter node %d has no feasible schedule at any tile >= 2 under the kernel-wide budgets (SMEM floor %d + candidate / FRAG floor %d + candidate vs MaxCache %d / MaxFragPerThread %d) — tiling is never disabled (2026-10-24 ruling); widen the budgets or the TileSizes set",
			lf.Tree.Iter.BasePtr().ID(), smemFloor, fragFloor, p.MaxCache, p.MaxFragPerThread)
	}
	return best, nil
}

// evalSchedule scores one candidate (window SoC vector + uniform tile):
// portion-grammar LoC completion, full per-iter decision map (window
// assignments plus the sequential defaults for every non-window iter), the
// promotion judge + traffic over the candidate sites, and the KERNEL-WIDE
// budget feasibility gates (the floors are what earlier LNTs already
// consume; R3/F6). The portion grammar holds by construction: the (at most
// one) CTA dim exhausts the CTA space (portion -1/full) and the (at most
// one) independent warp dim takes its portion of the warp space.
func evalSchedule(lf lntFacts, window []ir.Node, soc []SoCType, tile int, p Params, smemFloor, fragFloor int) (schedResult, bool, error) {
	treeIters := treeItersOf(lf.Tree)
	iters := make(map[ir.Node]IterDecision, len(treeIters))
	for _, it := range treeIters {
		// R1 (2026-10-24 tile-disable ruling): the non-window sequential
		// default carries the SWEPT UNIFORM TILE — every LNT dim is tiled
		// (tile >= 2 everywhere; there is no untiled default). A sequential
		// dim tiled at T is still exhaustive: between rounds step T, the
		// inside walk is bounded by the saturating tileMax.
		iters[it] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: tile}
	}
	// PORTION COMPLETION (the final model, replacing the legacy greedy
	// preset loops; loop_tiling.md "Portion grammar"). With the v0 policy
	// there is at most one CTA dim and at most one independent warp dim, so
	// no running product/sum accounting is needed — each dim's LoC follows
	// from its portion directly, and the walk order is irrelevant:
	//
	//   - SoCTA: portion -1/full — the single CTA dim exhausts the CTA
	//     space, so LoC = MaxCTA.
	//   - SoCWARP: LoC = warpPortion x warpsPerPortion warps, where
	//     warpsPerPortion = (MaxThreadsPerCTA/32)/PortionWarp. The CTA
	//     dim's coupled tile-inside walk consumes the REMAINING warp
	//     portions; its stride is derived by the lower pass, not
	//     recorded. warpPortion is p.WarpPortion, or auto (0):
	//     AutoWarpPortion when the vector carries a CTA dim (the coupled
	//     inside needs the remaining portions, >= 1), else PortionWarp
	//     (full).
	//   - Sequential: LoC 1.
	warpsPerPortion := (p.MaxThreadsPerCTA / 32) / PortionWarp
	warpPortion := p.WarpPortion
	if warpPortion == 0 {
		if hasCTALevel(soc) {
			warpPortion = AutoWarpPortion
		} else {
			warpPortion = PortionWarp
		}
	}
	// COUPLED-PORTION FLOOR (F2, ruled 2026-10-21): whenever the LNT
	// carries a CTA dim, the independent WarpPortion must leave >= 1 warp
	// portion for the coupled inside walk. A violation is a configuration
	// error, not a skip: every feasible vector for a CTA-carrying LNT
	// would emit a zero-stride inside loop, so the evaluation fails loud
	// (defensive error at schedule evaluation — tiling-verify re-checks
	// the same floor on published records).
	if hasCTALevel(soc) && warpPortion > PortionWarp-1 {
		return schedResult{}, false, fmt.Errorf(
			"tiling: WarpPortion %d leaves %d warp portion(s) for the CTA dim's coupled inside walk — the coupled-portion floor (F2) demands >= 1, i.e. WarpPortion <= PortionWarp-1 = %d whenever the LNT carries a CTA dim (a zero-portion coupled walk is a zero-stride inside loop)",
			warpPortion, PortionWarp-warpPortion, PortionWarp-1)
	}
	ctx := schedCtx{
		warpsPerPortion: warpsPerPortion,
		warpPortion:     warpPortion,
		coupledThreads:  (PortionWarp - warpPortion) * warpsPerPortion * 32,
	}
	for i := range window {
		d := IterDecision{SoC: soc[i], LoC: 1, TileSize: tile}
		switch soc[i] {
		case SoCTA:
			d.LoC = p.MaxCTA
		case SoCWARP:
			d.LoC = warpPortion * warpsPerPortion
		}
		iters[window[i]] = d
	}

	res := schedResult{iters: iters, sites: make(map[uint64]SiteDecision, len(lf.CS))}
	traffic := 0
	for _, sf := range lf.CS { // slice order: deterministic float summation
		j := judgeSite(sf, iters, ctx, p)
		if j.promoted {
			traffic += j.bytes * (j.reuse - 1)
			switch j.class {
			case ClassSMEM:
				res.smem += j.bytes
			case ClassFRAG:
				res.frag += j.bytes
			}
		}
		res.sites[sf.Load.BasePtr().ID()] = SiteDecision{
			Promoted: j.promoted, Tie: sf.Tie, Class: j.class,
			Reuse: j.reuse, Bytes: j.bytes, Reason: j.reason,
		}
	}
	// KERNEL-WIDE budget gates (R3 SMEM scope + F6 FRAG budget): the
	// floors are what earlier LNTs' chosen schedules already consume; the
	// same sums are what tiling-verify and the lower's resolveBudget gate.
	if smemFloor+res.smem > p.MaxCache {
		return schedResult{}, false, nil // kernel-wide SMEM budget gate
	}
	if fragFloor+res.frag > p.MaxFragPerThread {
		return schedResult{}, false, nil // kernel-wide FRAG budget gate
	}

	// SINGLE-TERM OBJECTIVE (loop_tiling.md "Objective Function ...,
	// SINGLE-TERM"): score is the GMEM traffic avoided alone — parallelism
	// is CONSTANT under mandatory per-LNT exhaustion, so the deleted
	// weighted parallelism term carried no ranking signal.
	res.score = float64(traffic)
	return res, true, nil
}

// siteJudgment is the promotion judge's product for one site.
type siteJudgment struct {
	promoted bool
	class    CacheClass
	reuse    int // per-thread re-traversal within one fill epoch
	bytes    int // window-model footprint
	reason   string
}

// judgeSite is the doc's promotion judge ("Judge Criterial"), as amended
// 2026-10-21 (R2 tier rule + F4 drop-rule gate + F6 frag budget + the
// per-thread reuse gate) and 2026-10-23 (C1b tier precedence). The site is
// classified by the SoC of its COVERED dims (the cached trailing dimensions;
// rank >= CacheDims is guaranteed by the admission filter):
//
//   - GATE: any covered dim CTA-assigned -> the site is REFUSED (a
//     CTA-covered tile has no coherent fill variant under the v0
//     mechanics). An iter with NO decision fails closed the same way.
//   - TIER PRECEDENCE (C1b, 2026-10-23): a site with a THREAD-VARYING
//     non-covered source coordinate — any referenced iter (INS) NOT in the
//     covered set, assigned CTA or WARP — takes ClassFRAG regardless of
//     the covered dims' SoC. A cooperative SMEM fill is CTA-uniform (R1),
//     and a fill whose own source coordinates move with the thread (the
//     worked example's lhs: the row coordinate is the coupled per-thread
//     walk) has no coherent shared-window variant.
//   - TIER: otherwise any covered dim WARP-assigned -> ClassSMEM (the
//     window class: cooperative LoadTileCta, window-sized buffer); all
//     covered dims sequential -> ClassFRAG (per-thread LoadTileThread).
//   - GATE: a FRAG site's per-thread bytes must fit MaxFragPerThread (F6).
//   - GATE: per-thread reuse < 2 -> not promoted (bytes loaded once and
//     read once avoid no traffic on the owning thread's behalf).
func judgeSite(sf SiteFacts, iters map[ir.Node]IterDecision, ctx schedCtx, p Params) siteJudgment {
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
			return siteJudgment{reason: ReasonCTACovered} // R2 tier refusal
		}
		if d.TileSize < 2 {
			return siteJudgment{reason: ReasonUnknownIter} // invalid tile (tile 1 is not a legal schedule): fail closed
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
			forcedFrag = true // C1b: a thread-varying non-covered coordinate
			break
		}
	}
	class := ClassFRAG
	if !forcedFrag {
		for _, c := range sf.Covered {
			if iters[c].SoC == SoCWARP {
				class = ClassSMEM
				break
			}
		}
	}
	bytes := siteWindowBytes(sf, iters, class)
	if class == ClassFRAG && bytes > p.MaxFragPerThread {
		return siteJudgment{class: class, bytes: bytes, reason: ReasonFragBytes}
	}
	reuse := siteReuseCount(sf, iters, ctx)
	if reuse < 2 {
		return siteJudgment{class: class, reuse: reuse, bytes: bytes, reason: ReasonLowReuse}
	}
	return siteJudgment{promoted: true, class: class, reuse: reuse, bytes: bytes}
}

// siteWindowBytes is the site's post-demotion footprint under the
// WINDOW-BYTES model (R2, ruled 2026-10-21 — rank-in-index =>
// window-in-shape), made class-aware by the C1b precedence (2026-10-23).
// Element size x the product over the covered dims of (alloc_d x
// tile_size_d), where alloc_d is the number of execution slots partitioning
// that dim's round window in the schedule:
//
//   - SEQ dim: 1 (the tile alone — one thread's sequential chunk),
//   - WARP dim: ClassSMEM — LoC_d x 32 threads (the whole group's round
//     window, NOT the single-thread chunk); ClassFRAG — 1 (C1b: the
//     per-thread window: a forced-FRAG site's WARP-covered dim is each
//     thread's own T_d-chunk, priced and sized per thread),
//   - CTA dim: LoC_d CTAs (the CTA group's round window; such sites are
//     refused by the covered-CTA gate regardless — the value stays
//     well-defined for diagnostics).
func siteWindowBytes(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass) int {
	e, ok := elemTypeOf(sf.Load)
	if !ok {
		return 0 // unavailable element type: contributes no traffic (fail closed)
	}
	bytes := elemBytes(e)
	for _, c := range sf.Covered {
		d, ok := iters[c]
		if !ok || d.TileSize < 1 {
			return 0
		}
		alloc := 1
		switch d.SoC {
		case SoCWARP:
			if class == ClassSMEM {
				alloc = d.LoC * 32
			}
		case SoCTA:
			alloc = d.LoC
		}
		bytes *= alloc * d.TileSize
	}
	return bytes
}

// siteReuseCount is the site's PER-THREAD reuse factor (ruled 2026-10-21):
// re-traversal by the OWNING thread within one fill epoch — the product
// over the site's precomputed reuse dims ({tie} plus the LNT iters strictly
// inner than the tie, minus covered) of the thread's own trip count of
// that dim's INSIDE walk, the only loops that run inside the fill position:
//
//   - SEQ dim: T_d (the sequential tile-interior walk),
//   - WARP dim: T_d (the owned chunk's sequential walk),
//   - CTA dim: ceil(T_d / coupledThreads) (the coupled element-stride
//     walk — the thread owns every coupledThreads-th element).
//
// Re-traversal by OTHER threads never counts toward the site's reuse.
func siteReuseCount(sf SiteFacts, iters map[ir.Node]IterDecision, ctx schedCtx) int {
	reuse := 1
	for _, it := range sf.ReuseDims {
		d, ok := iters[it]
		if !ok || d.TileSize < 1 {
			return 0 // unknown dim: fail closed (no reuse claim)
		}
		trips := d.TileSize
		if d.SoC == SoCTA {
			trips = ceilDiv(d.TileSize, ctx.coupledThreads)
		}
		reuse *= trips
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

// windowOf returns the LNT's search window: the last WindowK levels of the
// root -> deepest-leaf spine, outermost-of-window first. Every iter
// outside the window and every non-spine iter defaults sequential.
func windowOf(lf lntFacts, p Params) []ir.Node {
	spine := spineOf(lf.Tree)
	w := p.WindowK
	if w > len(spine) {
		w = len(spine)
	}
	out := make([]ir.Node, 0, w)
	for _, t := range spine[len(spine)-w:] {
		out = append(out, t.Iter)
	}
	return out
}

// spineOf returns the root -> deepest-leaf chain (deepest by Depth; ties
// broken by allocation order — the DFS visits children in allocation
// order, so the FIRST maximizer wins). Root first, deepest leaf last.
func spineOf(root *loopforest.Tree) []*loopforest.Tree {
	if root == nil {
		return nil
	}
	best := root
	var visit func(t *loopforest.Tree)
	visit = func(t *loopforest.Tree) {
		if t.Depth > best.Depth {
			best = t
		}
		for _, c := range t.Children {
			visit(c)
		}
	}
	visit(root)
	chain := []*loopforest.Tree{}
	for t := best; t != nil; t = t.Parent {
		chain = append(chain, t)
	}
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}
	return chain
}

// tileCandidates returns the uniform tile sweep — Params.TileSizes only,
// ascending and deduped (every entry >= 2 by Params.Validate; tile 1 is not
// a legal schedule, so nothing injects it).
func tileCandidates(p Params) []int {
	seen := make(map[int]bool, len(p.TileSizes))
	out := make([]int, 0, len(p.TileSizes))
	for _, t := range p.TileSizes {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	sort.Ints(out)
	return out
}

// hasCTALevel reports whether the SoC vector carries at least one CTA
// assignment ("feasible when it at least has one CTA assignment").
func hasCTALevel(soc []SoCType) bool {
	for _, s := range soc {
		if s == SoCTA {
			return true
		}
	}
	return false
}

// v0PolicyClean reports whether the SoC vector obeys the v0 searcher
// policy (loop_tiling.md "Composition Semantics, CTA x WARP", the "v0
// default searcher POLICY" bullet): at most ONE CTA-assigned dim and at
// most ONE independent WARP dim per LNT. Same-category multi-dim
// composition (CTA+CTA or independent WARP+WARP) is FUTURE pending the
// combined-slicing arithmetic — product factors of the level's space and
// portions as the selection grammar — so the searcher never emits such
// schedules in v0.
func v0PolicyClean(soc []SoCType) bool {
	cta, warp := 0, 0
	for _, s := range soc {
		switch s {
		case SoCTA:
			cta++
		case SoCWARP:
			warp++
		}
	}
	return cta <= 1 && warp <= 1
}
