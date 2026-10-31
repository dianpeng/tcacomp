// search.go is the fix4 §2.2-§2.6 feasible-area machinery: the
// innermost-window digit lists (SoC × LoC preset × tile per dim, ordered
// SEQ/CTA/WARP, LoC DESCENDING, tile ASCENDING), the mixed-radix
// enumeration with INCREMENTAL pruning (a partial vector is rejected the
// moment the digit being placed violates MAPM/MALoC — never
// enumerate-then-filter), the completed-vector CuMAP skip with the
// structural-infeasibility hard error, the site judge + objective, and
// the greedy value-density eviction against the kernel-wide MASRAM
// budgets. The judge and the bytes/reuse arithmetic live in judge.go; the
// constraint predicates live in constraints.go.
//
// Determinism contract (fix4 §2.2, INV): the enumeration order is fixed
// (outermost window dim = most significant digit), selection is
// strictly-greater on the lexicographic key (score', partitionCount)
// with exact ties keeping the FIRST-enumerated candidate, and the
// LoC-descending digit order resolves full ties toward LARGER LoC
// vectors ("use the partitioned space fully" — traffic is LoC-neutral,
// so the tie-only preference never overrides traffic).
//
// score' (quick-261029-n63) is the occupancy-aware objective:
// score' = traffic × (1 − idleFrac) × (ε + (1−ε)·fill), where traffic is
// the post-eviction one-currency sum and the occupancy factor comes from
// occupancyOf (constraints.go — the single source). The eviction loop,
// the constraint predicates, and the enumeration order are untouched:
// occupancy scales ONLY the final selection.
package tiling

import (
	"context"
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
	"tcacomp/internal/log"
)

// schedResult is one LNT's search product: per-iter decisions (keyed by
// the ITER node — the pass converts to BIV keys at record assembly),
// per-site decisions keyed by MemLoad node id, the chosen schedule's
// occupancy-aware score (score' = traffic × the occupancy factor, applied
// AFTER the greedy eviction), the raw post-eviction traffic sum it was
// scaled from (diagnostics + the fixed-schedule traffic pins), the
// partition count (the tiebreak), and its kernel-wide budget consumption
// (what the caller adds to the running floors for the next LNT).
type schedResult struct {
	iters   map[ir.Node]IterDecision
	sites   map[uint64]SiteDecision
	score   float64 // score' — the occupancy-scaled objective
	traffic int     // the raw post-eviction one-currency traffic sum
	partism int     // partition count: 2×#CTA dims + #WARP dims (R-PARTITION)
	smem    int     // the chosen schedule's promoted-SMEM window bytes
	frag    int     // the chosen schedule's promoted-FRAG register-block bytes (Σ min(raw, MaxFragPerThread), fix6 §3)
	// promotedCount is the post-eviction promoted-site count (the
	// zero-cache warning's predicate — read by the caller, never the
	// record).
	promotedCount int
}

// schedDigit is one per-dim digit of the feasible area: the SoC, the
// ABSOLUTE LoC (CTA dim -> #CTAs, WARP dim -> #warps, SEQ -> 1), and the
// per-dim tile.
type schedDigit struct {
	soc  SoCType
	loc  int
	tile int
}

// searchStats counts the enumeration's work — the incremental-pruning
// probe (a completed count far below the raw digit-vector product proves
// partial rejection, and pruned > 0 proves digits were turned away at
// placement).
type searchStats struct {
	placements int // digits accepted into a partial vector
	pruned     int // digits rejected at placement (MAPM/MALoC)
	completed  int // vectors fully assembled (pre-CuMAP-skip)
	evaluated  int // CTA-carrying completed vectors scored
}

// searchSchedule enumerates the feasible area of one LNT and returns the
// best schedule under the objective, gated by the KERNEL-WIDE budget
// floors already consumed by earlier LNTs (fix4 §2.3 MASRAM: feasibility
// accumulates across LNTs — the caller threads the running totals
// through; overflow EVICTS by value density, never errors).
func searchSchedule(lf lntFacts, p Params, smemFloor, fragFloor int) (schedResult, error) {
	res, _, err := searchScheduleStats(lf, p, smemFloor, fragFloor)
	return res, err
}

// searchScheduleStats is searchSchedule with the enumeration counters
// (the pruning probe's read face).
func searchScheduleStats(lf lntFacts, p Params, smemFloor, fragFloor int) (schedResult, searchStats, error) {
	var stats searchStats
	window := windowOf(lf, p)
	// R3 (fix5 §1): the WARP digit lands on exactly ONE dim per LNT — the
	// extent-aware winner, computed once here and passed into every
	// digitListOf call (feasible-area shaping, decide-side only: the
	// enumeration order, the judge, the eviction, and tiling-verify are
	// untouched, exactly like R-REDUCED).
	warpDim := warpDimOf(window, lf.CS, lf.eliminated, len(lf.reduced) > 0, p)
	digitLists := make([][]schedDigit, len(window))
	anyCTADigit := false
	for i, it := range window {
		digitLists[i] = digitListOf(it, p, lf.reduced, warpDim)
		for _, d := range digitLists[i] {
			if d.soc == SoCTA {
				anyCTADigit = true
				break
			}
		}
	}
	// Structural infeasibility (fix4 §2.6): no dim can carry a CTA
	// assignment — no legal factorization exists. This consciously
	// extends the 2026-10-24 "tiling never disabled" ruling: loops always
	// stay tiled >= 2; CACHES may be zero, parallelism may not.
	if !anyCTADigit {
		return schedResult{}, stats, fmt.Errorf(
			"tiling: LNT rooted at iter node %d admits no CTA-carrying schedule (no CTA digit survives the preset caps) — structural infeasibility (fix4 2.6); widen CtaLoCPresets or the extent caps",
			lf.Tree.Iter.BasePtr().ID())
	}

	defaultTile := maxTileOf(p)
	digits := make([]schedDigit, len(window))
	var best schedResult
	found := false

	better := func(res schedResult) bool {
		if !found {
			return true
		}
		if res.score != best.score {
			return res.score > best.score
		}
		return res.partism > best.partism
	}

	var enumerate func(i int, st partState)
	enumerate = func(i int, st partState) {
		if i == len(window) {
			stats.completed++
			if !st.cumapOK() {
				return // completed-vector CuMAP skip (>= 1 CTA dim)
			}
			stats.evaluated++
			res := evalSchedule(lf, window, digits, st, defaultTile, p, smemFloor, fragFloor)
			if better(res) {
				best, found = res, true
			}
			return
		}
		for _, dg := range digitLists[i] {
			// INCREMENTAL PRUNING (fix4 §2.2): reject the partial vector
			// the moment the placed digit violates MAPM/MALoC — the whole
			// subtree below it dies here, never at completion.
			if !st.mapmOK(dg.soc, p) || !st.malocOK(dg.soc, dg.loc, p) {
				stats.pruned++
				continue
			}
			stats.placements++
			digits[i] = dg
			enumerate(i+1, st.place(dg.soc, dg.loc))
		}
	}
	enumerate(0, initialPartState())
	if !found {
		// Defensive: anyCTADigit guarantees a CTA-carrying vector exists
		// (every digit list is non-empty — SEQ digits always exist), so
		// this arm is unreachable barring enumerator corruption.
		return schedResult{}, stats, fmt.Errorf(
			"tiling: LNT rooted at iter node %d produced no evaluable schedule (enumerator corruption?)",
			lf.Tree.Iter.BasePtr().ID())
	}
	if len(lf.CS) > 0 && best.promotedCount == 0 {
		// Zero-cache outcome (fix4 §2.6): every candidate site of the
		// CHOSEN schedule evicted or declined — allowed, warned (once per
		// LNT, never per candidate), never an error.
		log.Warn(context.Background(), "tiling: zero-cache schedule (all candidate sites evicted or declined)",
			"lnt_iter", lf.Tree.Iter.BasePtr().ID(), "sites", len(lf.CS))
	}
	return best, stats, nil
}

// evalSchedule scores one completed CTA-carrying vector: the per-iter
// decision map (window digits plus the sequential defaults for every
// non-window iter, R-DEFTILE), the per-LNT blockDim context, the site
// judge, the greedy value-density eviction against the kernel-wide
// floors, and the (score', partitionCount) key. Budget overflow never
// fails here — it spills (fix4 §2.6). score' (quick-261029-n63) is the
// post-eviction traffic sum scaled by the occupancy factor
// (occupancyOf — constraints.go): the factor uses the POST-EVICTION smem
// (evictability itself is occupancy-blind, unchanged).
func evalSchedule(lf lntFacts, window []ir.Node, digits []schedDigit, st partState, defaultTile int, p Params, smemFloor, fragFloor int) schedResult {
	treeIters := treeItersOf(lf.Tree)
	iters := make(map[ir.Node]IterDecision, len(treeIters))
	for _, it := range treeIters {
		// R-DEFTILE: non-window (and non-spine) iters default to SEQ at
		// the max tile preset — per-dim tiles are searched digits now, so
		// there is no swept uniform tile; every dim stays tiled >= 2
		// (the 2026-10-24 ruling preserved).
		iters[it] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: defaultTile}
	}
	for i := range window {
		iters[window[i]] = IterDecision{SoC: digits[i].soc, LoC: digits[i].loc, TileSize: digits[i].tile}
	}
	ctx := blockDimCtx{blockDim: blockDimOf(iters)}

	// Judge every candidate site (CS slice order — deterministic).
	type judged struct {
		sf SiteFacts
		j  siteJudgment
	}
	sites := make([]judged, 0, len(lf.CS))
	res := schedResult{iters: iters, sites: make(map[uint64]SiteDecision, len(lf.CS)), partism: st.partitionCount()}
	smem, frag := 0, 0
	for _, sf := range lf.CS {
		j := judgeSite(sf, iters, ctx, p)
		sites = append(sites, judged{sf: sf, j: j})
		if j.promoted {
			switch j.class {
			case ClassSMEM:
				smem += j.bytes
			case ClassFRAG:
				frag += j.bytes
			}
		}
	}

	// GREEDY VALUE-DENSITY EVICTION (fix4 §2.6, R-EVICT): while either
	// kernel-wide MASRAM gate overflows against the running floors,
	// repeatedly drop the promoted site with the least density
	// (density = reuse − 1; ties: smaller bytes first; further ties: CS
	// order). Evicted sites demote to GMEM with ReasonEvicted, retaining
	// their class/reuse/bytes for the verify cross-check; the loops stay
	// tiled. An over-budget state with nothing left to evict (floors
	// alone over budget) is the zero-cache outcome — allowed.
	for smemFloor+smem > p.MaxCache || fragFloor+frag > p.MaxFragPerThread {
		pick := -1
		for i := range sites {
			if !sites[i].j.promoted {
				continue
			}
			if pick < 0 {
				pick = i
				continue
			}
			di, dp := sites[i].j.reuse-1, sites[pick].j.reuse-1
			if di != dp {
				if di < dp {
					pick = i
				}
				continue
			}
			if sites[i].j.bytes < sites[pick].j.bytes {
				pick = i
			}
		}
		if pick < 0 {
			break // nothing evictable left: floors alone overflow — zero cache
		}
		switch sites[pick].j.class {
		case ClassSMEM:
			smem -= sites[pick].j.bytes
		case ClassFRAG:
			frag -= sites[pick].j.bytes
		}
		sites[pick].j.promoted = false
		sites[pick].j.reason = ReasonEvicted
	}

	promoted := 0
	scaledTraffic := 0.0 // R4 (fix5 §1): the score' currency — traffic × per-site tierEff
	for _, s := range sites {
		if s.j.promoted {
			promoted++
			// One CTA-wide currency (fix4 §2.4 amendment / §2.5): SMEM
			// windows are filled collectively (holders 1), FRAG windows
			// by each of the blockDim holder threads — RAW window bytes
			// × (reuse − 1) × holders is CTA-wide GMEM bytes avoided for
			// every class (fix6 §3: the traffic term prices the RAW
			// window, uncapped — the budget side prices the register
			// block via j.bytes, the ONE shared siteWindowBytes). The
			// int res.traffic stays the RAW pin (fixed-schedule tests
			// and tiling-verify consume it); R4's tierEff scales ONLY
			// the float64 score currency.
			res.traffic += s.j.rawBytes * (s.j.reuse - 1) * s.j.holders
			scaledTraffic += float64(s.j.rawBytes*(s.j.reuse-1)) * float64(s.j.holders) *
				reducedTierEff(lf.reduced, iters, s.sf.Covered, p)
		}
		res.sites[s.sf.Load.BasePtr().ID()] = SiteDecision{
			Promoted: s.j.promoted, Tie: s.sf.Tie, Class: s.j.class,
			Reuse: s.j.reuse, Bytes: s.j.bytes, Reason: s.j.reason,
		}
	}
	res.smem = smem
	res.frag = frag
	res.promotedCount = promoted
	// The occupancy factor (quick-261029-n63) applies AFTER the eviction
	// loop, to the post-eviction promoted bytes: score' = R4-scaled traffic
	// × (1 − idleFrac) × (ε + (1−ε)·fill). Selection-only — the eviction,
	// the constraints, and the enumeration are occupancy- and tierEff-blind;
	// a site covering no reduced warp-sharded dim carries tierEff exactly 1
	// (bit-identical to the pre-R4 score).
	res.score = scaledTraffic * occupancyOf(iters, smem, p).factor
	return res
}

// digitListOf builds one window dim's digit list in the fix4 §2.2
// enumeration order: [SEQ × tiles ascending] ++ [CTA × LoC descending ×
// tiles ascending] ++ [WARP × K descending × tiles ascending]. The
// R-LOCCAP preset cap applies per (dim, tile): a preset is feasible for
// a CONST extent only when preset <= max(2, ceil(extent/T)) — non-const
// extents enumerate uncapped. R-REDUCED: when ReducedDimWarp is false (the
// Wave-1 policy, now the explicit opt-out), a reducer-consuming dim carries
// no WARP digits at all.
//
// R1 (fix5 §1): after each SoC section's preset digits, the section gains
// its ONE extent-derived adaptive top digit per tile —
// clamp(ceil(effExtent/T), 2, MaxCTA) for CTA / (..., MaxWarpsPerCTA) for
// WARP, effExtent the const extent or the NominalExtent stand-in. Dedup:
// a value already present (a preset, or an adaptive value from an earlier
// tile) appears exactly once — presets win the duplicate. Placement is
// deterministic: tiles ascending, appended after the section's presets.
//
// R3 (fix5 §1, quick-261029-v9k): the WARP preset section AND this
// adaptiveSection(SoCWARP) call emit ONLY when iter is the LNT's
// warpDimOf winner — the per-LNT WARP digit is enumerated on exactly one
// (the extent-aware-chosen) dim. The R-REDUCED prune below stands on top
// (a reduced winner under ReducedDimWarp=false carries no WARP section at
// all — narrower than the pre-R3 flag=false semantics, where non-reduced
// dims kept their WARP digits; fix5 §1 R3 consequence, recorded in the
// task SUMMARY).
func digitListOf(iter ir.Node, p Params, reduced map[ir.Node]bool, warpDim ir.Node) []schedDigit {
	tiles := tileCandidates(p) // deduped, ascending (entries >= 2 by Validate)
	out := make([]schedDigit, 0, len(tiles)*(2+len(p.CtaLoCPresets)+len(p.WarpLoCPresets)))
	for _, t := range tiles {
		out = append(out, schedDigit{soc: SoCSequential, loc: 1, tile: t})
	}
	extent, constExtent := constExtentOf(iter)
	effExtent := extent // the R1/R4 effective-extent discipline (const wins)
	if !constExtent {
		effExtent = p.NominalExtent
	}
	capOf := func(tile int) int {
		if !constExtent {
			return -1 // uncapped
		}
		c := ceilDiv(extent, tile)
		if c < 2 {
			c = 2
		}
		return c
	}
	// adaptiveSection appends the R1 adaptive digit per tile (tiles
	// ascending) after the section's preset digits, skipping values already
	// present in the digit list (presets win; repeated adaptive values
	// collapse to the first-lowest tile). The adaptive value derives from
	// the effective extent, so it needs no R-LOCCAP cap of its own.
	adaptiveSection := func(soc SoCType) {
		seen := make(map[int]bool, len(out))
		for _, d := range out {
			if d.soc == soc {
				seen[d.loc] = true
			}
		}
		for _, t := range tiles {
			loc := adaptiveLoC(soc, t, effExtent, p)
			if seen[loc] {
				continue
			}
			seen[loc] = true
			out = append(out, schedDigit{soc: soc, loc: loc, tile: t})
		}
	}
	for _, loc := range presetDesc(p.CtaLoCPresets) {
		for _, t := range tiles {
			if c := capOf(t); c >= 0 && loc > c {
				continue // R-LOCCAP: preset exceeds the extent cap
			}
			out = append(out, schedDigit{soc: SoCTA, loc: loc, tile: t})
		}
	}
	adaptiveSection(SoCTA)
	if reduced[iter] && !p.ReducedDimWarp {
		// R-REDUCED: WARP on a reducer-consuming dim is pruned from the
		// feasible area while the split-k capability gate is closed
		// (explicit ReducedDimWarp=false — the Wave-1 policy).
		return out
	}
	if iter != warpDim {
		// R3: this dim lost the per-LNT WARP ranking — no WARP section at
		// all (the winner dim's list below carries presets + adaptive).
		return out
	}
	for _, k := range presetDesc(p.WarpLoCPresets) {
		for _, t := range tiles {
			if c := capOf(t); c >= 0 && k > c {
				continue // R-LOCCAP (a warp count is a per-dim preset too)
			}
			out = append(out, schedDigit{soc: SoCWARP, loc: k, tile: t})
		}
	}
	adaptiveSection(SoCWARP)
	return out
}

// presetDesc returns a preset list deduped and sorted DESCENDING (the
// LoC-descending digit order; a fresh slice — the Params field is never
// mutated, per the immutability discipline).
func presetDesc(xs []int) []int {
	seen := make(map[int]bool, len(xs))
	out := make([]int, 0, len(xs))
	for _, v := range xs {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] > out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// maxTileOf returns the max tile preset (the R-DEFTILE sequential
// default; entries >= 2 by Validate).
func maxTileOf(p Params) int {
	m := p.TileSizes[0]
	for _, v := range p.TileSizes {
		if v > m {
			m = v
		}
	}
	return m
}

// reducedIters derives the reducer-consuming iter set (R-REDUCED): the
// union over live KindReduce nodes of the iters referenced by the
// reduce's piped value input (operand 0), chased with the same Input-list
// walk the overlap analysis uses (findAxisIters). A WARP-SoC digit on
// any of these dims is a split-k schedule; under explicit
// Params.ReducedDimWarp=false such digits are pruned from the feasible
// area (the Wave-1 policy; fix4 §3.5, Wave 2 landed with the default true).
func reducedIters(g *ir.Graph) map[ir.Node]bool {
	out := make(map[ir.Node]bool)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindReduce {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) == 0 || ops[0] == nil {
			continue
		}
		for _, it := range findAxisIters(ops[0]) {
			out[it] = true
		}
	}
	return out
}

// reducedAxesOf returns the kernel's ELIMINATED iter set: the iters that
// feed a Reduce's value but are NOT a subscript axis of any MemStore —
// the axes the reduction DROPS. Distinct from reducedIters (the
// reducer-CONSUMING set, which additionally contains the SURVIVING axes:
// in scale_copy [R,C,K] -> [R,C] every IV feeds the reduce value, but only
// K is eliminated). The §3.1 prune (warpDimOf) and its tile-lower mirror
// (rejectWarpSurvivingDims) need the ELIMINATED semantic: a WARP digit on
// a SURVIVING axis is exactly the GPU-broken cross-product class
// (row_sum/softmax, canary 1/64). Store subscripts pin the surviving
// axes (MemStore operands are [value, tensor, axes...]).
func reducedAxesOf(g *ir.Graph) map[ir.Node]bool {
	stored := make(map[ir.Node]bool)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		ops := ir.OperandsOf(n)
		for _, ax := range ops[min(2, len(ops)):] { // [value, tensor, axes...]
			for _, it := range findAxisIters(ax) {
				stored[it] = true
			}
		}
	}
	out := make(map[ir.Node]bool)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindReduce {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) == 0 || ops[0] == nil {
			continue
		}
		for _, it := range findAxisIters(ops[0]) {
			if !stored[it] {
				out[it] = true
			}
		}
	}
	return out
}

// windowOf returns the LNT's search window: the last WindowK levels of
// the root -> deepest-leaf spine, outermost-of-window first. Every iter
// outside the window and every non-spine iter defaults sequential
// (R-DEFTILE).
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

// warpDimOf is the fix5 §1 R3 extent-aware WARP-dim ranking (quick-
// 261029-v9k): computed ONCE per LNT in searchScheduleStats, its winner
// is the only dim whose digit list carries a WARP section. The comparator
// is TOTAL — it always resolves, never panics, never leaves a tie:
//
//	L1: effective extent DESC — constExtentOf's const value, else
//	    Params.NominalExtent (the SAME discipline as R1's adaptiveLoC and
//	    R4's reducedTierEff, so the ranking, the enumeration, and the
//	    verify re-check can never disagree on an extent).
//	L2: stride-1 set membership first — stride1Set is the union over cs of
//	    the iters at each site's INNERMOST subscript position (the last
//	    non-empty AxesIters entry — the coalescing axis of every admitted
//	    candidate load); in-set dims rank above out-of-set dims, and
//	    WITHIN the set the innermost WINDOW position wins.
//	L3: non-reduced before reduced (the reduced axis as last-resort
//	    fallback — fires only on residuals L1/L2 left tied; VACUOUS under
//	    the §3.1 prune below, kept for comparator totality).
//	L4: innermost window position (final fallback — window positions are
//	    unique, so this always resolves).
//
// §3.1 PRUNE (author ruling, 2026-10-30, iq6 candidate-1 — supersedes the
// fix5 §1 R3 "any dim" reading): the candidate set is RESTRICTED TO
// REDUCED dims. A WARP digit on a non-reduced (surviving) dim lowers
// through the CTA-coupled-inside × lane-chunk composition whose cross
// product is 63/64 UNCOVERED — GPU-proven on the R3 re-choice kernels
// (row_sum/softmax, canary exactly 1/64 of the output space; the iq6
// §3.1 class "every (m,n) pair carries 16/1024 k values"). fix5 §3's
// premise ("the layout already flows — only the CHOICE of dim is new")
// is GPU-disproven. The prune restores the fix4 reduced-only WARP
// feasible area until the tile-lower cross-product lowering is designed;
// the R3 ranking stays as the tie-break AMONG REDUCED candidates. An LNT
// with NO reduced dim in the window gets NO WARP digit at all. tile-lower
// carries the fail-closed MIRROR (rejectWarpSurvivingDims).
//
// Consequences (fix5 §1 R3, recorded in the task SUMMARY): at most one
// WARP dim per LNT (multi-warp-dim vectors leave the feasible area;
// MAPM/MALoC accounting unchanged); an LNT whose winner is reduced under
// explicit ReducedDimWarp=false carries no WARP dim at all. The ranking
// reads only verified IR facts (constExtentOf, SiteFacts.AxesIters, the
// reduced map) and never indexes an empty window (windowOf is non-empty
// for every searched LNT; a nil/empty window yields nil, and digitListOf
// then emits no WARP section for any dim — fail-safe, T-v9k-01).
func warpDimOf(window []ir.Node, cs []SiteFacts, eliminated map[ir.Node]bool, hasReduce bool, p Params) ir.Node {
	// §3.1 prune (ruling above): in a REDUCE-CARRYING kernel (every real
	// v0 kernel — the chain shape is map|reduce[|map]|store), surviving
	// (non-eliminated) candidates are excluded from the ranking
	// entirely; no eliminated dim in the window means NO WARP digit
	// anywhere (nil winner — digitListOf then emits no WARP section for
	// any dim; fail-safe, T-v9k-01). The set is the ELIMINATED axes
	// (reducedAxesOf) — NOT lf.reduced, whose reducer-consuming semantic
	// also contains the SURVIVING axes. A crafted NO-REDUCE graph (the
	// template-corpus fixtures — unreachable from the v0 pipeline) keeps
	// the full-window R3 ranking: nothing real to protect, and the
	// template pins stay intact for the future §3.1 lowering's machinery
	// tests. The lower-side mirror (rejectWarpSurvivingDims) shares this
	// scope.
	if hasReduce {
		var windowEliminated []ir.Node
		for _, it := range window {
			if eliminated[it] {
				windowEliminated = append(windowEliminated, it)
			}
		}
		if len(windowEliminated) == 0 {
			return nil
		}
		window = windowEliminated
	}
	if len(window) == 0 {
		return nil
	}
	// stride1Set: per site, the iters at the innermost NON-EMPTY subscript
	// position (the load's coalescing axis).
	stride1 := make(map[ir.Node]bool)
	for _, sf := range cs {
		for i := len(sf.AxesIters) - 1; i >= 0; i-- {
			if len(sf.AxesIters[i]) == 0 {
				continue
			}
			for _, it := range sf.AxesIters[i] {
				stride1[it] = true
			}
			break
		}
	}
	pos := make(map[ir.Node]int, len(window))
	for i, it := range window {
		pos[it] = i
	}
	effExtent := func(it ir.Node) int {
		if e, ok := constExtentOf(it); ok {
			return e
		}
		return p.NominalExtent
	}
	// better reports whether a ranks strictly above b (window order:
	// index 0 = outermost, len-1 = innermost).
	better := func(a, b ir.Node) bool {
		if ea, eb := effExtent(a), effExtent(b); ea != eb {
			return ea > eb // L1
		}
		sa, sb := stride1[a], stride1[b]
		if sa != sb {
			return sa // L2 membership: in-set first
		}
		if sa && pos[a] != pos[b] {
			return pos[a] > pos[b] // L2 within-set: innermost window first
		}
		if ra, rb := eliminated[a], eliminated[b]; ra != rb {
			return !ra // L3: non-reduced before reduced
		}
		return pos[a] > pos[b] // L4: innermost window (total — positions unique)
	}
	best := window[0]
	for _, it := range window[1:] {
		if better(it, best) {
			best = it
		}
	}
	return best
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

// tileCandidates returns the deduped ascending tile sweep —
// Params.TileSizes only (every entry >= 2 by Params.Validate; tile 1 is
// not a legal schedule, so nothing injects it).
func tileCandidates(p Params) []int {
	seen := make(map[int]bool, len(p.TileSizes))
	out := make([]int, 0, len(p.TileSizes))
	for _, t := range p.TileSizes {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
