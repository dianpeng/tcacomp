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
// doc/optana/loop_tiling.md's final model: the PSched search (innermost-K
// window, flat 3^K x tiles enumeration, portion-grammar LoC completion,
// feasibility gates, v0 policy), the promotion judge, the single-term
// traffic-avoided objective, and the pass plumbing (registry, record
// publication, determinism, no mutation).
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
	warpCount  int
	warpSum    int
}

func feasibilityOf(rec *Record) feas {
	f := feas{ctaProduct: 1}
	for _, d := range rec.Iters {
		switch d.SoC {
		case SoCTA:
			f.ctaCount++
			f.ctaProduct *= d.LoC
		case SoCWARP:
			f.warpCount++
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
	facts := allFacts.Trees[0]
	socMap := iterSocMap(t, facts.Tree, rec)
	ctx := judgeCtxOf(t, facts.Tree, socMap, p)
	for _, sf := range facts.CS {
		want := judgeSite(sf, socMap, ctx, p)
		got, ok := rec.SiteDecisionOf(sf.Load)
		if !ok {
			t.Fatalf("CS load node %d missing its site entry", sf.Load.BasePtr().ID())
		}
		if got.Promoted != want.promoted || got.Class != want.class ||
			got.Reuse != want.reuse || got.Bytes != want.bytes {
			t.Fatalf("site node %d: record says (%v, %s, reuse %d, bytes %d), judge over the chosen schedule says (%v, %s, reuse %d, bytes %d)",
				sf.Load.BasePtr().ID(), got.Promoted, got.Class, got.Reuse, got.Bytes,
				want.promoted, want.class, want.reuse, want.bytes)
		}
	}
}

// judgeCtxOf rebuilds the judge's portion context over an iter-keyed
// decision map, mirroring evalSchedule's resolution: explicit
// Params.WarpPortion, else auto (AutoWarpPortion when the LNT carries a
// CTA dim, PortionWarp otherwise).
func judgeCtxOf(t *testing.T, tree *loopforest.Tree, iters map[ir.Node]IterDecision, p Params) schedCtx {
	t.Helper()
	warpsPerPortion := (p.MaxThreadsPerCTA / 32) / PortionWarp
	warpPortion := p.WarpPortion
	hasCTA := false
	for _, d := range iters {
		if d.SoC == SoCTA {
			hasCTA = true
		}
	}
	if warpPortion == 0 {
		if hasCTA {
			warpPortion = AutoWarpPortion
		} else {
			warpPortion = PortionWarp
		}
	}
	return schedCtx{
		warpsPerPortion: warpsPerPortion,
		warpPortion:     warpPortion,
		coupledThreads:  (PortionWarp - warpPortion) * warpsPerPortion * 32,
	}
}

// TestFeasibilityBounds pins the portion completion and the accounting
// gates: a minimal MaxCTA is exhausted exactly, a minimal (PortionWarp-warp)
// thread budget caps the warp sum, and a zero-CTA schedule is never
// selected.
func TestFeasibilityBounds(t *testing.T) {
	fx, _, _ := matmul()

	t.Run("minimal cta budget", func(t *testing.T) {
		p := DefaultParams()
		p.MaxCTA = PortionCTA // one CTA portion
		rec := runPass(t, fx.g, p)
		f := feasibilityOf(rec)
		if f.ctaCount < 1 {
			t.Fatal("at least one CTA assignment is required even under a tight budget")
		}
		if f.ctaProduct != PortionCTA {
			t.Fatalf("the single CTA dim must exhaust the minimal MaxCTA %d, got LoC product %d", PortionCTA, f.ctaProduct)
		}
	})

	t.Run("minimal warp budget", func(t *testing.T) {
		p := DefaultParams()
		p.MaxThreadsPerCTA = PortionWarp * 32 // eight warps: the assert-and-trap minimum
		rec := runPass(t, fx.g, p)
		f := feasibilityOf(rec)
		if f.warpSum*32 > PortionWarp*32 {
			t.Fatalf("warp LoC sum %d x 32 exceeds the %d-thread budget", f.warpSum, PortionWarp*32)
		}
	})

	t.Run("zero-cta never selected", func(t *testing.T) {
		rec := runPass(t, fx.g, DefaultParams())
		if feasibilityOf(rec).ctaCount == 0 {
			t.Fatal("a schedule with zero CTA levels must never be selected")
		}
	})
}

// TestPortionCompletion pins the portion-grammar LoC derivation directly on
// evalSchedule: the single CTA dim exhausts MaxCTA (portion -1/full), the
// independent warp dim takes warpPortion x quantum warps (auto =
// AutoWarpPortion when a CTA dim is present; explicit WarpPortion
// overrides), and sequential dims stay at LoC 1. The coupled tile-inside
// warp half of the CTA dim is deliberately NOT recorded (the lower pass
// derives its stride).
func TestPortionCompletion(t *testing.T) {
	fx, _, _ := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p) // outermost first: [iterM, iterN, iterK]
	quantum := (p.MaxThreadsPerCTA / 32) / PortionWarp

	soc := []SoCType{SoCTA, SoCWARP, SoCSequential}
	res, ok, err := evalSchedule(lf, window, soc, 32, p, 0, 0)
	if err != nil {
		t.Fatalf("[CTA, warp, seq] at t=32 must not error: %v", err)
	}
	if !ok {
		t.Fatal("[CTA, warp, seq] at t=32 must be budget-feasible on the matmul fixture (rhs declines by reuse, lhs is a 128 B FRAG)")
	}
	if d := res.iters[window[0]]; d.SoC != SoCTA || d.LoC != p.MaxCTA {
		t.Fatalf("CTA dim must exhaust MaxCTA %d (portion -1/full), got %s/%d", p.MaxCTA, d.SoC, d.LoC)
	}
	if d := res.iters[window[1]]; d.SoC != SoCWARP || d.LoC != AutoWarpPortion*quantum {
		t.Fatalf("auto warp dim must carry %d x %d warps, got %s/%d", AutoWarpPortion, quantum, d.SoC, d.LoC)
	}
	if d := res.iters[window[2]]; d.SoC != SoCSequential || d.LoC != 1 {
		t.Fatalf("sequential dim must stay at LoC 1, got %s/%d", d.SoC, d.LoC)
	}

	// An explicit WarpPortion overrides the auto rule.
	p.WarpPortion = 2
	res, ok, err = evalSchedule(lf, window, soc, 32, p, 0, 0)
	if err != nil {
		t.Fatalf("explicit WarpPortion must not error: %v", err)
	}
	if !ok {
		t.Fatal("explicit WarpPortion must not change feasibility here")
	}
	if d := res.iters[window[1]]; d.LoC != 2*quantum {
		t.Fatalf("explicit warp portion 2 must give LoC %d, got %d", 2*quantum, d.LoC)
	}
}

// TestV0PolicyFilter pins the v0 searcher policy (loop_tiling.md
// "Composition Semantics, CTA x WARP"): the emitted record never carries
// more than one CTA dim or more than one independent WARP dim per LNT —
// 2-CTA and 2-warp SoC vectors exist in the raw 3^K enumeration but are
// skipped, on both a deep window (many raw violators) and the matmul.
func TestV0PolicyFilter(t *testing.T) {
	for _, tc := range []struct {
		name string
		g    func() (fixture, ir.Node, ir.Node)
	}{
		{"window7", func() (fixture, ir.Node, ir.Node) {
			maxes := []float64{64, 64, 64, 64, 64, 64, 64}
			fx := newChain("window7", maxes...)
			scaffoldStore(fx, fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
			return fx, nil, nil
		}},
		{"matmul", func() (fixture, ir.Node, ir.Node) { return matmul() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fx, _, _ := tc.g()
			rec := runPass(t, fx.g, DefaultParams())
			f := feasibilityOf(rec)
			if f.ctaCount != 1 {
				t.Fatalf("v0 policy: exactly one CTA dim expected, got %d", f.ctaCount)
			}
			if f.warpCount > 1 {
				t.Fatalf("v0 policy: at most one independent WARP dim expected, got %d", f.warpCount)
			}
		})
	}
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

// TestPromotionJudge pins the amended judge (R2 tier rule + F4 drop-rule
// gate + F6 frag budget + the per-thread reuse gate): classification is by
// the SoC of the site's COVERED dims — ANY CTA-covered dim refuses the
// site; ANY WARP-covered dim yields the SMEM window class; all-SEQ covered
// dims yield FRAG (subject to the per-thread byte and reuse gates); ANY
// covered dim at tile 1 blocks promotion (the judge-side F4 gate, matching
// the lower's drop rule).
func TestPromotionJudge(t *testing.T) {
	fx, lhs, rhs := matmul()
	facts := overlapOf(fx, DefaultParams())
	sl := siteOf(t, facts, lhs) // covered {K}
	sr := siteOf(t, facts, rhs) // covered {N, K}
	p := DefaultParams()

	soc := func(dM, dN, dK IterDecision) (map[ir.Node]IterDecision, schedCtx) {
		m := map[ir.Node]IterDecision{fx.iters[0]: dM, fx.iters[1]: dN, fx.iters[2]: dK}
		quantum := (p.MaxThreadsPerCTA / 32) / PortionWarp
		warpPortion := AutoWarpPortion // the auto rule for CTA-carrying vectors
		return m, schedCtx{warpsPerPortion: quantum, warpPortion: warpPortion,
			coupledThreads: (PortionWarp - warpPortion) * quantum * 32}
	}

	// A CTA-COVERED dim refuses the site (lhs covers K): R2 tier rule.
	// (The old judge's "CTA at the last subscript position" gate is
	// superseded — the covered set, not the axes, classifies.)
	iters, ctx := soc(
		IterDecision{SoCTA, 100, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCTA, 4, 32})
	j := judgeSite(sl, iters, ctx, p)
	if j.promoted || j.class != ClassNone || j.reason != ReasonCTACovered {
		t.Fatalf("a CTA-covered dim must refuse the site, got (%v, %s, %q)", j.promoted, j.class, j.reason)
	}

	// All covered dims sequential, per-thread reuse >= 2: FRAG.
	iters, ctx = soc(
		IterDecision{SoCTA, 100, 32}, IterDecision{SoCSequential, 1, 32}, IterDecision{SoCSequential, 1, 32})
	j = judgeSite(sl, iters, ctx, p)
	if !j.promoted || j.class != ClassFRAG || j.bytes != 4*32 || j.reuse != 32 {
		t.Fatalf("all-SEQ covered dims must promote to FRAG (128 B, reuse 32), got (%v, %s, %d B, reuse %d)",
			j.promoted, j.class, j.bytes, j.reuse)
	}

	// A WARP-covered dim (lhs covers K): SMEM window class — bytes are the
	// whole group's round window (#threads_d x T), not the bare tile.
	iters, ctx = soc(
		IterDecision{SoCTA, 100, 32}, IterDecision{SoCSequential, 1, 32}, IterDecision{SoCWARP, 8, 32})
	j = judgeSite(sl, iters, ctx, p)
	wantBytes := 4 * (8 * 32) * 32 // 8 warps = 256 threads, T 32
	if !j.promoted || j.class != ClassSMEM || j.bytes != wantBytes {
		t.Fatalf("a WARP-covered dim must promote to the SMEM window class with window bytes %d, got (%v, %s, %d B)",
			wantBytes, j.promoted, j.class, j.bytes)
	}

	// Any covered dim at tile 1 (lhs covers K): not promoted (F4 gate —
	// ANY covered iter, not only all of them).
	iters, ctx = soc(
		IterDecision{SoCTA, 100, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCSequential, 1, 1})
	j = judgeSite(sl, iters, ctx, p)
	if j.promoted || j.reason != ReasonTileOne {
		t.Fatalf("a covered iter at tile 1 must block promotion, got (%v, %q)", j.promoted, j.reason)
	}

	// Per-thread reuse < 2 declines the site: rhs's only reuse dim is M;
	// with M the CTA dim the owning thread walks ceil(32/256) = 1
	// m-element per epoch — no re-traversal, no traffic avoided.
	iters, ctx = soc(
		IterDecision{SoCTA, 100, 32}, IterDecision{SoCWARP, 8, 32}, IterDecision{SoCSequential, 1, 32})
	j = judgeSite(sr, iters, ctx, p)
	if j.promoted || j.reason != ReasonLowReuse || j.reuse != 1 {
		t.Fatalf("rhs at default-like shapes must decline with per-thread reuse 1, got (%v, %q, reuse %d)",
			j.promoted, j.reason, j.reuse)
	}
}

// TestWorkedExampleJudge pins the doc's worked-example JUDGE output
// verbatim (loop_tiling.md "Promotion OUTCOME at default shapes" +
// loop_tile_lower.md §6bis "Worked example, ACCEPTED OUTCOME"): over the
// pinned schedule M=CTA/100/T32, N=WARP/8 warps=256 threads/T32,
// K=SEQ/T32 —
//
//   - lhs (tie iterN, covered {K} all-SEQ) is PROMOTED as a 128 B per-thread
//     FRAG with per-thread reuse 32 (its 32-step n walk per (m-element,
//     k-round) epoch);
//   - rhs (tie nil, covered {N WARP, K SEQ}) is DECLINED — its reuse dim is
//     M and the owning thread's per-epoch m-count is ceil(32/256) = 1 < 2 —
//     consistent with its 1 MB SMEM window exceeding the 64 KB budget.
func TestWorkedExampleJudge(t *testing.T) {
	fx, lhs, rhs := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p) // [iterM, iterN, iterK]

	soc := []SoCType{SoCTA, SoCWARP, SoCSequential}
	res, ok, err := evalSchedule(lf, window, soc, 32, p, 0, 0)
	if err != nil || !ok {
		t.Fatalf("the worked-example schedule must evaluate feasible: ok=%v err=%v", ok, err)
	}

	sl, ok := res.sites[lhs.BasePtr().ID()]
	if !ok {
		t.Fatal("lhs must carry a site entry")
	}
	if !sl.Promoted || sl.Class != ClassFRAG || sl.Reuse != 32 || sl.Bytes != 128 || sl.Reason != "" {
		t.Fatalf("worked-example lhs = (promoted %v, %s, reuse %d, bytes %d, %q), want (true, frag, reuse 32, bytes 128, \"\")",
			sl.Promoted, sl.Class, sl.Reuse, sl.Bytes, sl.Reason)
	}
	if sl.Tie != fx.iters[1] {
		t.Fatalf("worked-example lhs tie must be iterN, got %s", tieLabel(sl.Tie))
	}

	sr, ok := res.sites[rhs.BasePtr().ID()]
	if !ok {
		t.Fatal("rhs must carry a site entry")
	}
	if sr.Promoted || sr.Reason != ReasonLowReuse || sr.Reuse != 1 {
		t.Fatalf("worked-example rhs = (promoted %v, reuse %d, %q), want declined with per-thread reuse 1",
			sr.Promoted, sr.Reuse, sr.Reason)
	}
	// The declined rhs's recorded bytes are its would-be window: 256
	// threads x 32 x 32 x 4 B = 1 MB against the 64 KB kernel-wide budget.
	if sr.Bytes != 4*(8*32)*32*32 {
		t.Fatalf("worked-example rhs window bytes = %d, want %d", sr.Bytes, 4*(8*32)*32*32)
	}

	// The full pass at defaults must publish rhs declined with a recorded
	// reason (whatever schedule wins, rhs never promotes at these shapes:
	// its per-thread m-count is 1 under every CTA placement).
	rec := runPass(t, fx.g, p)
	got, ok := rec.SiteDecisionOf(rhs)
	if !ok {
		t.Fatal("rhs must carry a record entry")
	}
	if got.Promoted || got.Reason == "" {
		t.Fatalf("the published record must decline rhs with a recorded reason, got (%v, %q)",
			got.Promoted, got.Reason)
	}
}

// TestCoupledPortionFloor pins F2 (HI-04): whenever the LNT carries a CTA
// dim, the independent WarpPortion must leave >= 1 warp portion for the
// coupled inside walk — evalSchedule ERRORS (a zero-portion coupled walk
// is a zero-stride inside loop), the pass fails loud, and tiling-verify
// rejects a published record whose warp portion starves the coupled walk.
func TestCoupledPortionFloor(t *testing.T) {
	fx, _, _ := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p)

	t.Run("eval schedule rejects warpportion 8 with cta", func(t *testing.T) {
		q := p
		q.WarpPortion = PortionWarp // legal as a PORTION, illegal with a CTA dim
		_, _, err := evalSchedule(lf, window, []SoCType{SoCTA, SoCWARP, SoCSequential}, 32, q, 0, 0)
		if err == nil {
			t.Fatal("WarpPortion == PortionWarp with a CTA dim must error at schedule evaluation")
		}
		for _, want := range []string{"WarpPortion", "F2", "coupled"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("the floor error must name %q, got %q", want, err.Error())
			}
		}
		// The no-warp-dim CTA vector errors the same way: the coupled walk
		// starves regardless of whether an independent dim takes the
		// portions.
		_, _, err = evalSchedule(lf, window, []SoCType{SoCTA, SoCSequential, SoCSequential}, 32, q, 0, 0)
		if err == nil {
			t.Fatal("WarpPortion == PortionWarp must starve the coupled walk even without an independent warp dim")
		}
	})

	t.Run("pass fails loud", func(t *testing.T) {
		q := p
		q.WarpPortion = PortionWarp
		cache := opt.NewCache()
		err := (Pass{Params: q}).Run(&opt.Context{Pass: "tiling", Cache: cache}, fx.g)
		if err == nil {
			t.Fatal("the decide pass must fail loud on a floor-violating WarpPortion")
		}
		if !strings.Contains(err.Error(), "F2") {
			t.Fatalf("the pass error must name the F2 floor, got %q", err.Error())
		}
	})

	t.Run("verify rejects starved record", func(t *testing.T) {
		q := p
		q.WarpPortion = 7 // floor-legal: 8-7 = 1 coupled portion
		cache, rec := runDecide(t, fx.g, q)
		// Corrupt the record's warp LoC to the full 8 portions: the
		// coupled walk is starved, verify must flag it (the F2 record
		// check runs on published records).
		quantum := (q.MaxThreadsPerCTA / 32) / PortionWarp
		for biv, d := range rec.Iters {
			if d.SoC == SoCWARP {
				d.LoC = PortionWarp * quantum
				rec.Iters[biv] = d
				break
			}
		}
		err := verifyWith(fx.g, cache, q)
		if err == nil {
			t.Fatal("tiling-verify must reject a record whose warp portion starves the coupled walk")
		}
		if !strings.Contains(err.Error(), "F2") {
			t.Fatalf("the record floor error must name F2, got %q", err.Error())
		}
	})
}

// TestTrafficArithmeticAndOptimality pins the objective's amended
// arithmetic on the matmul fixture — window-bytes footprints, per-thread
// reuse (tile-derived inside-walk trips), the F6/CTA gates — and
// cross-checks the chosen schedule's score against a test-local
// enumeration of the whole schedule space: the pass must pick a maximizer.
func TestTrafficArithmeticAndOptimality(t *testing.T) {
	fx, lhs, rhs := matmul()
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	dM := decisionOf(t, rec, fx.iters[0])
	dN := decisionOf(t, rec, fx.iters[1])
	dK := decisionOf(t, rec, fx.iters[2])

	// Independent window-bytes + per-thread reuse arithmetic over the
	// record's own schedule. alloc_d = the execution slots partitioning
	// dim d's round window; the owning thread's inside-walk trips are T_d,
	// except a CTA dim's coupled walk: ceil(T / coupledThreads).
	alloc := func(d IterDecision) int {
		switch d.SoC {
		case SoCWARP:
			return d.LoC * 32
		case SoCTA:
			return d.LoC
		default:
			return 1
		}
	}
	quantum := (p.MaxThreadsPerCTA / 32) / PortionWarp
	coupled := (PortionWarp - AutoWarpPortion) * quantum * 32 // the chosen schedule carries a CTA dim
	trips := func(d IterDecision) int {
		if d.SoC == SoCTA {
			return ceilDiv(d.TileSize, coupled)
		}
		return d.TileSize
	}
	// lhs: covered {K} -> bytes 4·alloc(K)·T_k; reuse dims {N} -> trips(N).
	lhsBytes := 4 * alloc(dK) * dK.TileSize
	lhsReuse := trips(dN)
	// rhs: covered {N,K} -> bytes 4·alloc(N)·T_n·alloc(K)·T_k; reuse dims {M}.
	rhsBytes := 4 * alloc(dN) * dN.TileSize * alloc(dK) * dK.TileSize
	rhsReuse := trips(dM)

	sl, _ := rec.SiteDecisionOf(lhs)
	sr, _ := rec.SiteDecisionOf(rhs)
	if sl.Bytes != lhsBytes || sl.Reuse != lhsReuse {
		t.Fatalf("lhs record bytes/reuse = %d/%d, independent arithmetic says %d/%d",
			sl.Bytes, sl.Reuse, lhsBytes, lhsReuse)
	}
	if sr.Bytes != rhsBytes || sr.Reuse != rhsReuse {
		t.Fatalf("rhs record bytes/reuse = %d/%d, independent arithmetic says %d/%d",
			sr.Bytes, sr.Reuse, rhsBytes, rhsReuse)
	}

	traffic := 0
	smem, frag := 0, 0
	if sl.Promoted {
		traffic += lhsBytes * (lhsReuse - 1)
		if sl.Class == ClassSMEM {
			smem += lhsBytes
		} else {
			frag += lhsBytes
		}
	}
	if sr.Promoted {
		traffic += rhsBytes * (rhsReuse - 1)
		if sr.Class == ClassSMEM {
			smem += rhsBytes
		} else {
			frag += rhsBytes
		}
	}
	if smem > p.MaxCache {
		t.Fatalf("chosen schedule's kernel-wide SMEM footprint %d exceeds MaxCache %d", smem, p.MaxCache)
	}
	if frag > p.MaxFragPerThread {
		t.Fatalf("chosen schedule's FRAG footprint %d exceeds MaxFragPerThread %d", frag, p.MaxFragPerThread)
	}
	// SINGLE-TERM objective: score is the traffic avoided alone.
	score := float64(traffic)

	best := enumerateBest(t, fx, p)
	if score != best {
		t.Fatalf("chosen schedule scores %v, but the enumerated maximum is %v", score, best)
	}
}

// enumerateBest mirrors the pass's search over the matmul fixture and
// returns the maximum score — the optimality cross-check. The judge's
// gates and arithmetic are re-derived here INDEPENDENTLY (no judgeSite
// call): window bytes, per-thread inside-walk trips, the tier rule, the
// F4 tile-1 gate, the F6 per-site frag gate, the reuse gate, and the
// kernel-wide budget feasibility.
func enumerateBest(t *testing.T, fx fixture, p Params) float64 {
	t.Helper()
	facts := overlapOf(fx, p)
	window := windowOf(facts, p)

	tiles := []int{1}
	for _, ts := range p.TileSizes {
		tiles = append(tiles, ts)
	}
	sort.Ints(tiles)
	quantum := (p.MaxThreadsPerCTA / 32) / PortionWarp

	soc := make([]SoCType, len(window))
	best := -1.0
	for {
		hasCTA := false
		for _, s := range soc {
			if s == SoCTA {
				hasCTA = true
			}
		}
		if hasCTA && v0PolicyClean(soc) { // the search's feasibility + v0 policy gates
			warpPortion := p.WarpPortion
			if warpPortion == 0 {
				warpPortion = AutoWarpPortion // hasCTA holds here
			}
			if warpPortion > PortionWarp-1 {
				t.Fatalf("test params violate the F2 floor (WarpPortion %d with a CTA dim)", warpPortion)
			}
			coupled := (PortionWarp - warpPortion) * quantum * 32
			for _, tile := range tiles {
				iters := make(map[ir.Node]IterDecision, len(facts.Levels))
				for _, it := range treeItersOf(facts.Tree) {
					iters[it] = IterDecision{SoCSequential, 1, 1}
				}
				for i := range window { // portion completion
					d := IterDecision{SoC: soc[i], LoC: 1, TileSize: tile}
					switch soc[i] {
					case SoCTA:
						d.LoC = p.MaxCTA
					case SoCWARP:
						d.LoC = warpPortion * quantum
					}
					iters[window[i]] = d
				}
				traffic, smem, frag := 0, 0, 0
				for _, sf := range facts.CS {
					// The mirror judge: tier rule + gates, independent formulas.
					promoted, ctaCovered, tileOne := true, false, false
					class := ClassFRAG
					for _, c := range sf.Covered {
						d := iters[c]
						if d.SoC == SoCTA {
							ctaCovered = true
						}
						if d.TileSize == 1 {
							tileOne = true
						}
						if d.SoC == SoCWARP {
							class = ClassSMEM
						}
					}
					if ctaCovered || tileOne || len(sf.Covered) == 0 {
						promoted = false
					}
					bytes := 0
					if promoted {
						e, ok := elemTypeOf(sf.Load)
						bytes = elemBytes(e)
						for _, c := range sf.Covered {
							d := iters[c]
							perDim := d.TileSize
							if d.SoC == SoCWARP {
								perDim *= d.LoC * 32
							}
							bytes *= perDim
						}
						if !ok {
							promoted = false
						}
						if class == ClassFRAG && bytes > p.MaxFragPerThread {
							promoted = false
						}
					}
					reuse := 0
					if promoted {
						reuse = 1
						for _, rd := range sf.ReuseDims {
							d := iters[rd]
							tr := d.TileSize
							if d.SoC == SoCTA {
								tr = ceilDiv(d.TileSize, coupled)
							}
							reuse *= tr
						}
						if reuse < 2 {
							promoted = false
						}
					}
					if promoted {
						traffic += bytes * (reuse - 1)
						if class == ClassSMEM {
							smem += bytes
						} else {
							frag += bytes
						}
					}
				}
				feasible := smem <= p.MaxCache && frag <= p.MaxFragPerThread
				score := float64(traffic) // single-term objective
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

// addWarpWindowLNT appends one 3-deep chain (root->mid->leaf, extents 64)
// with two co-traversed rank-2 loads referencing {root, leaf}: SET @ leaf,
// tie mid, covered {leaf}. When the leaf is WARP-assigned the sites are
// SMEM windows of 4 x (#threads x T) bytes each; with WarpPortion 1 the
// group is 64 threads, so each site costs 256·T bytes.
func addWarpWindowLNT(g *ir.Graph, tag string) (root, mid, leaf, lp, lq ir.Node) {
	f32 := ir.InternScalar(ir.TF32)
	dim := func(name string) ir.Node { return g.NewInputParam("ctor", ir.RoleDim, name, nil) }
	var parent ir.Node
	for i := 0; i < 3; i++ {
		dv := g.NewDimVar("ctor", dim(fmt.Sprintf("%s_d%d", tag, i)), g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", f32, 64))
		parent = g.NewIter("ctor", biv, parent)
		switch i {
		case 0:
			root = parent
		case 1:
			mid = parent
		case 2:
			leaf = parent
		}
	}
	lp = g.NewMemLoad("ctor", tensorParam(g, tag+"_p", 2), []ir.Node{iv(g, root), iv(g, leaf)})
	lq = g.NewMemLoad("ctor", tensorParam(g, tag+"_q", 2), []ir.Node{iv(g, root), iv(g, leaf)})
	return root, mid, leaf, lp, lq
}

// scaffoldConstStore wires a store+return observation over a const (the
// decide-side fixtures need no data flow through their loads).
func scaffoldConstStore(g *ir.Graph, name string) {
	out := g.NewInputParam("ctor", ir.RoleTensor, name, ir.InternTensor(ir.TF32, []ir.DimRef{{Name: name + "_d0"}}))
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, nil, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
}

// TestKernelWideSmemBudget pins the R3 ruling (HI-06): the SMEM budget is
// KERNEL-WIDE — the running total threads through the searcher, so two
// LNTs whose promotions are EACH individually legal under MaxCache cannot
// both promote when their SUM exceeds it: the first LNT (Roots() order)
// consumes the budget and the second LNT's sites are declined in every
// feasible schedule. tiling-verify stays green over the result — decide
// and verify gate the SAME kernel-wide sum.
func TestKernelWideSmemBudget(t *testing.T) {
	p := DefaultParams()
	p.WarpPortion = 1        // 1 portion x 2 warps = 64 threads per WARP dim
	p.MaxFragPerThread = 128 // both FRAG sites (2 x 4·T >= 256 at T >= 32) never fit together

	// Each LNT alone: both sites promote as 64-thread windows at the
	// winning tile 128 — 4 x 64 x 128 = 32768 B per site, 65536 B per
	// LNT, exactly the 64 KB budget.
	single := func(t *testing.T) {
		g := ir.NewGraph("kwb-single")
		_, _, _, lp, lq := addWarpWindowLNT(g, "a")
		scaffoldConstStore(g, "out")
		rec := runPass(t, g, p)
		for _, load := range []ir.Node{lp, lq} {
			sd, ok := rec.SiteDecisionOf(load)
			if !ok {
				t.Fatal("each site must carry a record entry")
			}
			if !sd.Promoted || sd.Class != ClassSMEM || sd.Bytes != 4*64*128 {
				t.Fatalf("individually-legal site = (%v, %s, %d B), want promoted SMEM windows of %d B",
					sd.Promoted, sd.Class, sd.Bytes, 4*64*128)
			}
		}
	}
	t.Run("each lnt alone is legal", single)

	t.Run("sum over budget declines the second lnt", func(t *testing.T) {
		g := ir.NewGraph("kwb-two")
		_, _, _, lpA, lqA := addWarpWindowLNT(g, "a")
		_, _, _, lpB, lqB := addWarpWindowLNT(g, "b")
		scaffoldConstStore(g, "out")
		rec := runPass(t, g, p)
		for _, load := range []ir.Node{lpA, lqA} {
			sd, _ := rec.SiteDecisionOf(load)
			if !sd.Promoted || sd.Class != ClassSMEM {
				t.Fatalf("the first LNT's site must still promote, got (%v, %s)", sd.Promoted, sd.Class)
			}
		}
		for _, load := range []ir.Node{lpB, lqB} {
			sd, ok := rec.SiteDecisionOf(load)
			if !ok {
				t.Fatal("the second LNT's sites must carry entries")
			}
			if sd.Promoted {
				t.Fatalf("the second LNT's site must be declined: the kernel-wide sum is already at MaxCache (R3)")
			}
			if sd.Reason == "" {
				t.Fatal("a declined site must record its reason")
			}
		}
		// The record is CONSISTENT: verify gates the same kernel-wide sum
		// and stays green.
		cache := opt.NewCache()
		if err := (Pass{Params: p}).Run(&opt.Context{Pass: "tiling", Cache: cache}, g); err != nil {
			t.Fatalf("tiling.Run: %v", err)
		}
		if err := (VerifyPass{Params: p}).Run(&opt.Context{Pass: "tiling-verify", Cache: cache}, g); err != nil {
			t.Fatalf("tiling-verify must be green over the kernel-wide-budgeted record: %v", err)
		}
	})
}

// TestFragBudgetGate pins F6 (MaxFragPerThread, default 256 B/thread):
// the judge declines a single FRAG site whose per-thread bytes exceed the
// cap, and the schedule-level gate accumulates the kernel-wide FRAG sum
// against the floors the caller threads in.
func TestFragBudgetGate(t *testing.T) {
	fx, lhs, _ := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p)

	// Judge level: lhs (covered {K}) at K SEQ tile 64 is a 256 B FRAG —
	// exactly the cap, promoted; tile 128 is 512 B, declined.
	soc := []SoCType{SoCTA, SoCSequential, SoCSequential}
	quantum := (p.MaxThreadsPerCTA / 32) / PortionWarp
	ctx := schedCtx{warpsPerPortion: quantum, warpPortion: AutoWarpPortion,
		coupledThreads: (PortionWarp - AutoWarpPortion) * quantum * 32}
	iters := map[ir.Node]IterDecision{
		fx.iters[0]: {SoC: SoCTA, LoC: p.MaxCTA, TileSize: 64},
		fx.iters[1]: {SoC: SoCSequential, LoC: 1, TileSize: 64},
		fx.iters[2]: {SoC: SoCSequential, LoC: 1, TileSize: 64},
	}
	j := judgeSite(siteOf(t, lf, lhs), iters, ctx, p)
	if !j.promoted || j.class != ClassFRAG || j.bytes != 256 {
		t.Fatalf("a 256 B FRAG at the cap must promote, got (%v, %s, %d B)", j.promoted, j.class, j.bytes)
	}
	iters[fx.iters[2]] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: 128}
	iters[fx.iters[1]] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: 128}
	iters[fx.iters[0]] = IterDecision{SoC: SoCTA, LoC: p.MaxCTA, TileSize: 128}
	j = judgeSite(siteOf(t, lf, lhs), iters, ctx, p)
	if j.promoted || j.reason != ReasonFragBytes || j.bytes != 512 {
		t.Fatalf("a 512 B FRAG must decline with the F6 reason, got (%v, %q, %d B)", j.promoted, j.reason, j.bytes)
	}

	// Schedule level: the same 256 B FRAG site is feasible at floor 0 but
	// infeasible once earlier LNTs already consumed 128 FRAG bytes
	// (fragFloor threads through evalSchedule — the kernel-wide F6 gate).
	_, ok, err := evalSchedule(lf, window, soc, 64, p, 0, 0)
	if err != nil || !ok {
		t.Fatalf("tile-64 FRAG at floor 0 must be feasible: ok=%v err=%v", ok, err)
	}
	_, ok, err = evalSchedule(lf, window, soc, 64, p, 0, 128)
	if err != nil {
		t.Fatalf("floor-gated evaluation must not error: %v", err)
	}
	if ok {
		t.Fatal("tile-64 FRAG with a 128-byte floor must be infeasible (128+256 > MaxFragPerThread)")
	}
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
// immediately after memopt-verify (113), and tiling-verify (115)
// immediately after tiling. The memopt sub-module is imported directly —
// the same registration opt/all accumulates (opt/all itself cannot be
// imported from this package-internal test: import cycle).
func TestRegistryOrder(t *testing.T) {
	_ = memopt.Pass{} // side import: fires memopt + memopt-verify registration
	pipe := opt.BuildPipeline()
	mv, til, tv := -1, -1, -1
	for i, p := range pipe {
		switch p.ID() {
		case "memopt-verify":
			mv = i
		case "tiling":
			til = i
		case "tiling-verify":
			tv = i
		}
	}
	if mv < 0 || til < 0 || tv < 0 {
		t.Fatalf("default pipeline missing memopt-verify (%d), tiling (%d), or tiling-verify (%d)", mv, til, tv)
	}
	if til != mv+1 {
		t.Fatalf("tiling (index %d) must land immediately after memopt-verify (index %d)", til, mv)
	}
	if tv != til+1 {
		t.Fatalf("tiling-verify (index %d) must land immediately after tiling (index %d)", tv, til)
	}
}
