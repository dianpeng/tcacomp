package tiling

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/memopt"
)

// tiling_test.go pins the DECIDE pass (priority 114) per
// doc/optana/loop_tiling.md: the PSched search (innermost-K window, flat
// 3^K x tiles enumeration, greedy LoC completion, feasibility gates), the
// promotion judge, the parallelism + traffic-avoided objective, and the
// pass plumbing (registry, record publication, determinism, no mutation).
//
// The package-internal test position is deliberate: the judge and the
// search's shared helpers are exercised directly. The opt/all composition
// root cannot be imported here (import cycle through the test binary), so
// the registry-order pin imports the memopt sub-module directly — the same
// registrations opt/all accumulates.

// runPass runs the decide pass over g and returns the published record.
func runPass(t *testing.T, g *ir.Graph, p Params) *Record {
	t.Helper()
	cache := opt.NewCache()
	if err := (Pass{Params: p}).Run(&opt.Context{Pass: "tiling", Cache: cache}, g); err != nil {
		t.Fatalf("tiling.Run: %v", err)
	}
	got := cache.Get(RecordName)
	if got == nil {
		t.Fatal("tiling.Run must publish the record under \"tiling-record\"")
	}
	rec, ok := got.(*Record)
	if !ok {
		t.Fatalf("cache entry under %q is %T, want *Record", RecordName, got)
	}
	return rec
}

// decisionOf resolves iter's decision from the record.
func decisionOf(t *testing.T, rec *Record, iter ir.Node) IterDecision {
	t.Helper()
	d, ok := rec.IterDecisionOf(iter)
	if !ok {
		t.Fatalf("iter node %d has no record entry", iter.BasePtr().ID())
	}
	return d
}

// feasibilityOf summarizes a record's per-LNT feasibility.
type feas struct {
	ctaCount   int
	ctaProduct int
	warpSum    int
	locSum     int
}

func feasibilityOf(rec *Record) feas {
	f := feas{ctaProduct: 1}
	for _, d := range rec.Iters {
		f.locSum += d.LoC
		switch d.SoC {
		case SoCTA:
			f.ctaCount++
			f.ctaProduct *= d.LoC
		case SoCWARP:
			f.warpSum += d.LoC
		}
	}
	return f
}

// renderRecord renders a record deterministically (BIV id / site key
// ascending) for equality comparison.
func renderRecord(rec *Record) string {
	var b strings.Builder
	keys := make([]uint64, 0, len(rec.Iters))
	byID := make(map[uint64]IterDecision, len(rec.Iters))
	for biv, d := range rec.Iters {
		keys = append(keys, biv.BasePtr().ID())
		byID[biv.BasePtr().ID()] = d
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, k := range keys {
		d := byID[k]
		fmt.Fprintf(&b, "%d:%s/%d/%d;", k, d.SoC, d.LoC, d.TileSize)
	}
	skeys := make([]int, 0, len(rec.Sites))
	sbyID := make(map[int]SiteDecision, len(rec.Sites))
	for id, d := range rec.Sites {
		skeys = append(skeys, int(id))
		sbyID[int(id)] = d
	}
	sort.Ints(skeys)
	for _, k := range skeys {
		d := sbyID[k]
		tie := "nil"
		if d.Tie != nil {
			tie = fmt.Sprintf("%d", d.Tie.BasePtr().ID())
		}
		fmt.Fprintf(&b, "%d:%v/%s/%s;", k, d.Promoted, tie, d.Class)
	}
	return b.String()
}

// iterSocMap rebuilds the judge's iter-keyed view from a record.
func iterSocMap(t *testing.T, tree *loopforest.Tree, rec *Record) map[ir.Node]IterDecision {
	t.Helper()
	out := make(map[ir.Node]IterDecision)
	for _, it := range treeItersOf(tree) {
		out[it] = decisionOf(t, rec, it)
	}
	return out
}

// TestMatmulEndToEnd pins the doc's worked example through the whole pass:
// all three BIV keys recorded, a feasible schedule (at least one CTA,
// bounds holding), the Task-1 ties carried into the site entries, and the
// promotion classes matching the judge recomputed over the chosen schedule.
func TestMatmulEndToEnd(t *testing.T) {
	fx, lhs, rhs := matmul()
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	if len(rec.Iters) != 3 {
		t.Fatalf("all three BIV keys must be present, got %d entries", len(rec.Iters))
	}
	f := feasibilityOf(rec)
	if f.ctaCount < 1 {
		t.Fatal("the schedule must carry at least one CTA assignment")
	}
	if f.ctaProduct > p.MaxCTA {
		t.Fatalf("CTA LoC product %d exceeds MaxCTA %d", f.ctaProduct, p.MaxCTA)
	}
	if f.warpSum*32 > p.MaxThreadsPerCTA {
		t.Fatalf("warp LoC sum %d warps exceeds the thread budget", f.warpSum)
	}

	sl, ok := rec.SiteDecisionOf(lhs)
	if !ok {
		t.Fatal("lhs must carry a site entry")
	}
	if sl.Tie != fx.iters[1] {
		t.Fatalf("lhs site tie must be iterN, got %s", tieLabel(sl.Tie))
	}
	sr, ok := rec.SiteDecisionOf(rhs)
	if !ok {
		t.Fatal("rhs must carry a site entry")
	}
	if sr.Tie != nil {
		t.Fatalf("rhs site tie must be nil (above the outermost loop), got %s", tieLabel(sr.Tie))
	}

	se := sideeffect.Analyze(fx.g)
	forest := loopforest.Analyze(fx.g)
	allFacts := analyzeOverlap(fx.g, forest, se, p)
	facts := allFacts[0]
	socMap := iterSocMap(t, facts.Tree, rec)
	for _, sf := range facts.CS {
		want, class := judgeSite(sf, socMap, p)
		got, ok := rec.SiteDecisionOf(sf.Load)
		if !ok {
			t.Fatalf("CS load node %d missing its site entry", sf.Load.BasePtr().ID())
		}
		if got.Promoted != want || got.Class != class {
			t.Fatalf("site node %d: record says (%v, %s), judge over the chosen schedule says (%v, %s)",
				sf.Load.BasePtr().ID(), got.Promoted, got.Class, want, class)
		}
	}
}

// TestFeasibilityBounds pins the greedy LoC completion and the accounting
// gates: a tight MaxCTA caps the CTA product, a tight thread budget caps
// the warp sum, and a zero-CTA schedule is never selected.
func TestFeasibilityBounds(t *testing.T) {
	fx, _, _ := matmul()

	t.Run("tight cta budget", func(t *testing.T) {
		p := DefaultParams()
		p.MaxCTA = 4
		rec := runPass(t, fx.g, p)
		f := feasibilityOf(rec)
		if f.ctaCount < 1 {
			t.Fatal("at least one CTA assignment is required even under a tight budget")
		}
		if f.ctaProduct > 4 {
			t.Fatalf("CTA LoC product %d exceeds the tight MaxCTA 4", f.ctaProduct)
		}
	})

	t.Run("tight warp budget", func(t *testing.T) {
		p := DefaultParams()
		p.MaxThreadsPerCTA = 64 // two warps
		rec := runPass(t, fx.g, p)
		f := feasibilityOf(rec)
		if f.warpSum*32 > 64 {
			t.Fatalf("warp LoC sum %d x 32 exceeds the 64-thread budget", f.warpSum)
		}
	})

	t.Run("zero-cta never selected", func(t *testing.T) {
		rec := runPass(t, fx.g, DefaultParams())
		if feasibilityOf(rec).ctaCount == 0 {
			t.Fatal("a schedule with zero CTA levels must never be selected")
		}
	})
}

// TestSearchWindow pins the innermost-K window: a 7-deep chain under
// WindowK 5 records the 2 outermost iters as SoCSequential / LoC 1 /
// TileSize 1.
func TestSearchWindow(t *testing.T) {
	maxes := []float64{64, 64, 64, 64, 64, 64, 64}
	fx := newChain("window7", maxes...)
	scaffoldStore(fx, fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	rec := runPass(t, fx.g, DefaultParams())
	for i, it := range fx.iters[:2] {
		d := decisionOf(t, rec, it)
		if d.SoC != SoCSequential || d.LoC != 1 || d.TileSize != 1 {
			t.Fatalf("outside-window iter %d (index %d) must be sequential/1/1, got %s/%d/%d",
				it.BasePtr().ID(), i, d.SoC, d.LoC, d.TileSize)
		}
	}
	if feasibilityOf(rec).ctaCount < 1 {
		t.Fatal("the window search must still assign at least one CTA")
	}
}

// TestPromotionJudge pins the doc's judge verbatim: CTA at the last
// subscript position blocks promotion; the last two positions fully
// sequential yield FRAG; warp anywhere relevant yields SMEM; a covered
// iter with tile 1 blocks promotion.
func TestPromotionJudge(t *testing.T) {
	fx, lhs, rhs := matmul()
	facts := overlapOf(fx, DefaultParams())
	sl := siteOf(t, facts, lhs) // axes [M, K]
	sr := siteOf(t, facts, rhs) // axes [N, K]
	p := DefaultParams()

	soc := func(dM, dN, dK IterDecision) map[ir.Node]IterDecision {
		return map[ir.Node]IterDecision{fx.iters[0]: dM, fx.iters[1]: dN, fx.iters[2]: dK}
	}

	// CTA at the last position (K): not promoted.
	prom, class := judgeSite(sl, soc(
		IterDecision{SoCTA, 16, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCTA, 4, 32}), p)
	if prom || class != ClassNone {
		t.Fatalf("CTA at the last position must block promotion, got (%v, %s)", prom, class)
	}

	// Last two positions fully sequential: FRAG.
	prom, class = judgeSite(sl, soc(
		IterDecision{SoCSequential, 1, 32}, IterDecision{SoCTA, 16, 32}, IterDecision{SoCSequential, 1, 32}), p)
	if !prom || class != ClassFRAG {
		t.Fatalf("last two positions sequential must promote to FRAG, got (%v, %s)", prom, class)
	}

	// Warp at the last position: SMEM.
	prom, class = judgeSite(sl, soc(
		IterDecision{SoCTA, 16, 32}, IterDecision{SoCSequential, 1, 32}, IterDecision{SoCWARP, 8, 32}), p)
	if !prom || class != ClassSMEM {
		t.Fatalf("warp at the last position must promote to SMEM, got (%v, %s)", prom, class)
	}

	// Warp at the second-to-last position (rhs axis N): SMEM.
	prom, class = judgeSite(sr, soc(
		IterDecision{SoCTA, 16, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCSequential, 1, 32}), p)
	if !prom || class != ClassSMEM {
		t.Fatalf("warp at the second-to-last position must promote to SMEM, got (%v, %s)", prom, class)
	}

	// Any covered iter with tile 1 (lhs covers K): not promoted.
	prom, class = judgeSite(sl, soc(
		IterDecision{SoCTA, 16, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCSequential, 1, 1}), p)
	if prom || class != ClassNone {
		t.Fatalf("a covered iter with tile 1 must block promotion, got (%v, %s)", prom, class)
	}
}

// TestTrafficArithmeticAndOptimality pins the objective's arithmetic on the
// matmul fixture (const extents) and cross-checks the chosen schedule's
// score against a test-local enumeration of the whole schedule space — the
// pass must pick a maximizer.
func TestTrafficArithmeticAndOptimality(t *testing.T) {
	fx, lhs, rhs := matmul()
	const M, N, K = 1024, 2048, 512
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	dM := decisionOf(t, rec, fx.iters[0])
	dN := decisionOf(t, rec, fx.iters[1])
	dK := decisionOf(t, rec, fx.iters[2])

	ceil := func(a, b int) int { return (a + b - 1) / b }
	// lhs: covered {K} -> bytes 4*tile(K); reuse {N} -> ceil(N/tile(N)).
	lhsBytes := 4 * dK.TileSize
	lhsReuse := ceil(N, dN.TileSize)
	// rhs: covered {N,K} -> bytes 4*tile(N)*tile(K); reuse {M} -> ceil(M/tile(M)).
	rhsBytes := 4 * dN.TileSize * dK.TileSize
	rhsReuse := ceil(M, dM.TileSize)

	sl, _ := rec.SiteDecisionOf(lhs)
	sr, _ := rec.SiteDecisionOf(rhs)
	traffic := 0
	smem := 0
	if sl.Promoted {
		traffic += lhsBytes * (lhsReuse - 1)
		if sl.Class == ClassSMEM {
			smem += lhsBytes
		}
	}
	if sr.Promoted {
		traffic += rhsBytes * (rhsReuse - 1)
		if sr.Class == ClassSMEM {
			smem += rhsBytes
		}
	}
	if smem > p.MaxCache {
		t.Fatalf("chosen schedule's SMEM footprint %d exceeds MaxCache %d", smem, p.MaxCache)
	}
	score := float64(feasibilityOf(rec).locSum) + float64(traffic)

	best := enumerateBest(t, fx, p)
	if score != best {
		t.Fatalf("chosen schedule scores %v, but the enumerated maximum is %v", score, best)
	}
}

// enumerateBest mirrors the pass's search over the matmul fixture and
// returns the maximum score — the optimality cross-check.
func enumerateBest(t *testing.T, fx fixture, p Params) float64 {
	t.Helper()
	se := sideeffect.Analyze(fx.g)
	forest := loopforest.Analyze(fx.g)
	facts := analyzeOverlap(fx.g, forest, se, p)[0]
	treeIters := treeItersOf(facts.Tree)
	depths := make(map[ir.Node]int, len(treeIters))
	for _, it := range treeIters {
		node, _ := forest.NodeOf(it)
		depths[it] = node.Depth
	}
	window := windowOf(facts, p)

	tiles := []int{1}
	for _, ts := range p.TileSizes {
		tiles = append(tiles, ts)
	}
	sort.Ints(tiles)

	extents := make(map[ir.Node]int, len(treeIters))
	for _, it := range treeIters {
		extents[it] = extentOf(it, p)
	}
	ceil := func(a, b int) int { return (a + b - 1) / b }

	soc := make([]SoCType, len(window))
	best := -1.0
	for {
		hasCTA := false
		for _, s := range soc {
			if s == SoCTA {
				hasCTA = true
			}
		}
		if hasCTA {
			for _, tile := range tiles {
				iters := make(map[ir.Node]IterDecision, len(treeIters))
				for _, it := range treeIters {
					iters[it] = IterDecision{SoCSequential, 1, 1}
				}
				prod, warpSum := 1, 0
				feasible := true
				for i := len(window) - 1; i >= 0; i-- { // greedy inner -> outer
					d := IterDecision{SoC: soc[i], LoC: 1, TileSize: tile}
					switch soc[i] {
					case SoCTA:
						for _, c := range p.CTALoCs {
							if prod*c <= p.MaxCTA && c > d.LoC {
								d.LoC = c
							}
						}
						prod *= d.LoC
					case SoCWARP:
						for _, w := range p.WarpLoCs {
							if warpSum+w <= p.MaxThreadsPerCTA/32 && w > d.LoC {
								d.LoC = w
							}
						}
						warpSum += d.LoC
					}
					iters[window[i]] = d
				}
				traffic, smem := 0, 0
				for _, sf := range facts.CS {
					prom, class := judgeSite(sf, iters, p)
					if !prom {
						continue
					}
					bytes := 4
					for _, c := range sf.Covered {
						bytes *= iters[c].TileSize
					}
					reuseDims := reuseDimsOf(sf, depths, treeIters)
					reuse := 1
					for _, rd := range reuseDims {
						reuse *= ceil(extents[rd], iters[rd].TileSize)
					}
					traffic += bytes * (reuse - 1)
					if class == ClassSMEM {
						smem += bytes
					}
				}
				if smem > p.MaxCache {
					feasible = false
				}
				locSum := 0
				for _, d := range iters {
					locSum += d.LoC
				}
				score := p.ParallelWeight*float64(locSum) + p.ReuseWeight*float64(traffic)
				if feasible && score > best {
					best = score
				}
			}
		}
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
	return best
}

// reuseDimsOf mirrors the objective's reuse-dim derivation: {tie} plus the
// LNT iters strictly inner than the tie, minus covered.
func reuseDimsOf(sf SiteFacts, depths map[ir.Node]int, treeIters []ir.Node) []ir.Node {
	covered := make(map[ir.Node]bool, len(sf.Covered))
	for _, c := range sf.Covered {
		covered[c] = true
	}
	var out []ir.Node
	if sf.Tie != nil {
		if !covered[sf.Tie] {
			out = append(out, sf.Tie)
		}
	}
	for _, it := range treeIters {
		if covered[it] || it == sf.Tie {
			continue
		}
		if sf.Tie == nil || depths[it] > depths[sf.Tie] {
			out = append(out, it)
		}
	}
	return out
}

// TestDeterminismIdempotent pins INV determinism: a second Run over the
// same graph produces an equal record.
func TestDeterminismIdempotent(t *testing.T) {
	fx, _, _ := matmul()
	cache := opt.NewCache()
	run := func() *Record {
		if err := (Pass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tiling", Cache: cache}, fx.g); err != nil {
			t.Fatalf("tiling.Run: %v", err)
		}
		return cache.Get(RecordName).(*Record)
	}
	first := renderRecord(run())
	second := renderRecord(run())
	if first != second {
		t.Fatalf("a second Run must produce an equal record:\n%s\n%s", first, second)
	}
}

// TestRunMutatesNothing pins ruling D2: the pass creates no MutContext and
// the graph is byte-identical before and after Run (node set, ids, and
// LastTouchedBy stamps).
func TestRunMutatesNothing(t *testing.T) {
	fx, _, _ := matmul()
	type stamp struct {
		id   uint64
		ltb  string
		kind ir.Kind
	}
	snapshot := func() []stamp {
		var out []stamp
		for _, n := range fx.g.AllocatedNodes() {
			out = append(out, stamp{n.BasePtr().ID(), n.BasePtr().LastTouchedBy(), n.KindOf()})
		}
		return out
	}
	before := snapshot()
	runPass(t, fx.g, DefaultParams())
	after := snapshot()
	if len(before) != len(after) {
		t.Fatalf("node count changed: %d before, %d after", len(before), len(after))
	}
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("node %d changed: %+v before, %+v after", before[i].id, before[i], after[i])
		}
	}
}

// TestRegistryOrder pins the default pipeline position: tiling (114) lands
// immediately after memopt-verify (113). The memopt sub-module is imported
// directly — the same registration opt/all accumulates (opt/all itself
// cannot be imported from this package-internal test: import cycle).
func TestRegistryOrder(t *testing.T) {
	_ = memopt.Pass{} // side import: fires memopt + memopt-verify registration
	pipe := opt.BuildPipeline()
	mv, til := -1, -1
	for i, p := range pipe {
		switch p.ID() {
		case "memopt-verify":
			mv = i
		case "tiling":
			til = i
		}
	}
	if mv < 0 || til < 0 {
		t.Fatalf("default pipeline missing memopt-verify (%d) or tiling (%d)", mv, til)
	}
	if til != mv+1 {
		t.Fatalf("tiling (index %d) must land immediately after memopt-verify (index %d)", til, mv)
	}
}
