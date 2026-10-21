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
//     over {1} ∪ TileSizes (t=1 = tiling disabled). Per-cache-count
//     heuristic tile sets are future work. This keeps the doc's 3^K
//     tractability claim literally true: total <= 3^K * (|TileSizes|+1)
//     <= 1215 schedules per LNT at defaults (the v0 policy filter only
//     skips more of them).
//   - Portion completion replaces the legacy greedy LoC presets: under the
//     v0 policy there is at most one CTA dim (portion -1/full, LoC =
//     MaxCTA) and at most one independent warp dim (LoC = warp portion x
//     warpsPerPortion); the CTA dim's coupled tile-inside walk implicitly
//     consumes the remaining warp portions, its stride derived by the
//     lower pass, not recorded.
//   - v0 POLICY (loop_tiling.md "Composition Semantics, CTA x WARP", the
//     "v0 default searcher POLICY" bullet): at most ONE CTA-assigned dim
//     and at most ONE independent WARP dim per LNT. Same-category
//     multi-dim composition (CTA+CTA, WARP+WARP) is FUTURE pending the
//     combined-slicing arithmetic, so vectors violating the policy are
//     skipped outright.
//   - Single-term objective (loop_tiling.md "Objective Function ...,
//     SINGLE-TERM"): score = traffic avoided; the parallelism term is
//     deleted — under mandatory per-LNT exhaustion total parallelism is
//     CONSTANT, so it carries no ranking signal.
//   - SMEM-only budget: the doc's Max.cache gate counts SMEM caches; FRAG
//     sites are register-budgeted by the LOWER pass per its spec's ruling.
package tiling

import (
	"sort"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// schedResult is one LNT's search product: per-iter decisions (keyed by the
// ITER node — the pass converts to BIV keys at record assembly), per-site
// decisions keyed by MemLoad node id, and the chosen schedule's score.
type schedResult struct {
	iters map[ir.Node]IterDecision
	sites map[uint64]SiteDecision
	score float64
}

// searchSchedule enumerates the PSched space of one LNT and returns the
// best schedule under the objective. Enumeration order is fixed — SoC
// odometer order (the outermost window level is the most significant
// digit), tiles ascending within each vector — and selection keeps the
// FIRST strict maximum, so the result is deterministic (INV).
func searchSchedule(f loopforest.Forest, lf lntFacts, p Params) schedResult {
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
				res, ok := evalSchedule(f, lf, window, soc, tile, p)
				if !ok {
					continue // infeasible (SMEM budget): skipped
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
		// Defensive: the t=1 point of any CTA-carrying vector is always
		// feasible (nothing promotes at tile 1, so the SMEM sum is 0), so
		// this is unreachable for validated Params — but a search that
		// found nothing still owes every LNT a schedule. The legacy
		// all-sequential fallback is INVALID under the >= 1-CTA ruling (an
		// LNT without a CTA dim cannot execute under the CUDA thread
		// model), so the fallback CARRIES a CTA dim: the outermost window
		// level is SoCTA, the rest sequential, t=1.
		fb := make([]SoCType, len(window))
		if len(fb) > 0 {
			fb[0] = SoCTA
		}
		res, _ := evalSchedule(f, lf, window, fb, 1, p)
		return res
	}
	return best
}

// evalSchedule scores one candidate (window SoC vector + uniform tile):
// portion-grammar LoC completion, full per-iter decision map (window
// assignments plus the sequential defaults for every non-window iter), the
// promotion judge + traffic over the candidate sites, and the SMEM
// feasibility gate. The portion grammar holds by construction: the (at
// most one) CTA dim exhausts the CTA space (portion -1/full) and the (at
// most one) independent warp dim takes its portion of the warp space.
func evalSchedule(f loopforest.Forest, lf lntFacts, window []ir.Node, soc []SoCType, tile int, p Params) (schedResult, bool) {
	treeIters := treeItersOf(lf.Tree)
	iters := make(map[ir.Node]IterDecision, len(treeIters))
	for _, it := range treeIters {
		iters[it] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: 1}
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
	//     dim's coupled tile-inside walk implicitly consumes the REMAINING
	//     warp portions; its stride is derived by the lower pass, not
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
	traffic, smem := 0, 0
	for _, sf := range lf.CS { // slice order: deterministic float summation
		prom, class := judgeSite(sf, iters, p)
		bytes := siteBytesPerTile(sf, iters)
		if prom {
			traffic += bytes * (siteReuseCount(f, sf, iters, p) - 1)
			if class == ClassSMEM {
				smem += bytes
			}
		}
		res.sites[sf.Load.BasePtr().ID()] = SiteDecision{Promoted: prom, Tie: sf.Tie, Class: class}
	}
	if smem > p.MaxCache {
		return schedResult{}, false // SMEM budget gate
	}

	// SINGLE-TERM OBJECTIVE (loop_tiling.md "Objective Function ...,
	// SINGLE-TERM"): score is the GMEM traffic avoided alone — parallelism
	// is CONSTANT under mandatory per-LNT exhaustion, so the deleted
	// weighted parallelism term carried no ranking signal.
	res.score = float64(traffic)
	return res, true
}

// judgeSite is the doc's promotion judge ("Judge Criterial"), over the
// load's subscript positions (rank >= CacheDims is guaranteed by the
// Task-1 admission filter):
//
//   - GATE: any CTA-assigned iter referenced at the LAST subscript position
//     blocks promotion (the last dim crosses CTA boundaries — no coherent
//     cache line). An iter with NO decision fails closed the same way.
//   - GATE: any covered iter with tile_size 1 blocks promotion (the doc's
//     "if tile_size == 1 for a specific iteration, cache is NOT promoted").
//   - CLASS: the last two positions (xn-1, xn) both fully sequential ->
//     ClassFRAG (the access lives in a single thread); otherwise ->
//     ClassSMEM (within a CTA but not within a single thread). A position
//     referencing an iter with no decision counts as non-sequential.
func judgeSite(sf SiteFacts, iters map[ir.Node]IterDecision, p Params) (bool, CacheClass) {
	if len(sf.AxesIters) == 0 {
		return false, ClassNone
	}
	last := sf.AxesIters[len(sf.AxesIters)-1]
	for _, it := range last {
		d, ok := iters[it]
		if !ok {
			return false, ClassNone // unknown assignment: fail closed
		}
		if d.SoC == SoCTA {
			return false, ClassNone // CTA at the last position
		}
	}
	for _, c := range sf.Covered {
		d, ok := iters[c]
		if !ok || d.TileSize < 1 {
			return false, ClassNone // unknown/invalid tile: fail closed
		}
		if d.TileSize == 1 {
			return false, ClassNone // tiling disabled on a covered dim
		}
	}
	if len(sf.AxesIters) >= 2 &&
		positionSequential(sf.AxesIters[len(sf.AxesIters)-2], iters) &&
		positionSequential(last, iters) {
		return true, ClassFRAG
	}
	return true, ClassSMEM
}

// positionSequential reports whether every iter referenced at one
// subscript position holds a sequential assignment (unknown iters count as
// non-sequential — the caller's fail-closed stance).
func positionSequential(axis []ir.Node, iters map[ir.Node]IterDecision) bool {
	for _, it := range axis {
		d, ok := iters[it]
		if !ok || d.SoC != SoCSequential {
			return false
		}
	}
	return true
}

// siteBytesPerTile is the site's post-demotion footprint: element size
// times the product of the covered iters' tile sizes (RULE 3).
func siteBytesPerTile(sf SiteFacts, iters map[ir.Node]IterDecision) int {
	e, ok := elemTypeOf(sf.Load)
	if !ok {
		return 0 // unavailable element type: contributes no traffic (fail closed)
	}
	bytes := elemBytes(e)
	for _, c := range sf.Covered {
		if d, ok := iters[c]; ok && d.TileSize > 1 {
			bytes *= d.TileSize
		}
	}
	return bytes
}

// siteReuseCount is the site's reuse factor: the product over the reuse
// dims — {tie} plus the LNT iters strictly inner than the tie, minus
// covered — of ceil(extent_d / tile_d). Extents come from the BIV max
// operand via ir.ConstValueOf; a non-const (symbolic) extent contributes
// the NominalExtent factor.
func siteReuseCount(f loopforest.Forest, sf SiteFacts, iters map[ir.Node]IterDecision, p Params) int {
	covered := make(map[ir.Node]bool, len(sf.Covered))
	for _, c := range sf.Covered {
		covered[c] = true
	}
	reuse := 1
	consider := func(it ir.Node) {
		if covered[it] {
			return
		}
		tile := 1
		if d, ok := iters[it]; ok && d.TileSize > 0 {
			tile = d.TileSize
		}
		reuse *= ceilDiv(extentOf(it, p), tile)
	}
	if sf.Tie != nil {
		consider(sf.Tie)
	}
	for _, it := range sf.Levels { // the LNT's iters, inner -> outer
		if sf.Tie != nil && it == sf.Tie {
			continue
		}
		if sf.Tie == nil || depthOfIter(f, it) > depthOfIter(f, sf.Tie) {
			consider(it)
		}
	}
	return reuse
}

// depthOfIter returns the iter's forest depth (-1 for non-members).
func depthOfIter(f loopforest.Forest, it ir.Node) int {
	node, ok := f.NodeOf(it)
	if !ok {
		return -1
	}
	return node.Depth
}

// extentOf reads an iter's static extent from its BIV's max operand; a
// non-const or malformed bound yields the NominalExtent stand-in, and a
// non-positive constant is clamped to 1 (a malformed extent must not
// produce a zero reuse factor).
func extentOf(iter ir.Node, p Params) int {
	deps := ir.DataDeps(iter)
	if len(deps) == 0 {
		return p.NominalExtent
	}
	ops := ir.DataDeps(deps[0])
	if len(ops) < 2 {
		return p.NominalExtent
	}
	if v, ok := ir.ConstValueOf(ops[1]); ok && v >= 1 {
		return int(v)
	}
	return p.NominalExtent
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

// tileCandidates returns the uniform tile sweep {1} ∪ TileSizes, ascending
// and deduped.
func tileCandidates(p Params) []int {
	seen := make(map[int]bool, len(p.TileSizes)+1)
	out := make([]int, 0, len(p.TileSizes)+1)
	for _, t := range append([]int{1}, p.TileSizes...) {
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
