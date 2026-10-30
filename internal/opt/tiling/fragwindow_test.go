// fragwindow_test.go pins fix6 §3 (quick-261030-v62, design/fix/fix6.md §3):
// the honest FRAG per-thread window pricing. A FRAG site whose RAW
// per-thread window exceeds MaxFragPerThread is PRICED as a streamed
// register block — min(raw, MaxFragPerThread) for the BUDGET — and
// PROMOTED when its traffic gate passes, exactly like every other class;
// the per-site F6 size decline (ReasonFragBytes) is gone. The TRAFFIC
// term prices the RAW window bytes × (reuse − 1) × holders, uncapped —
// the one-currency avoided-GMEM model is not weakened. The kernel-wide
// budget still binds on the block footprints: Σ block vs
// MaxFragPerThread spills by value density (ReasonEvicted, fix4 §2.6).
//
// Why (fix6 §2, the five-link deadlock measured 2026-10-29/30): on
// scale_copy const-K the all-SEQ collapse vector's covered [C,K] prices a
// 4 B × 32 × 32 = 4096 B/thread raw window > 256 B, both load sites
// declined, traffic 0, and the search could only pick the R4-penalized
// ceremony vector. These pins lock the fix: the collapse class wins.
//
// Every re-derived pin this file and its siblings touch carried the OLD
// decline behavior — which fix6 §3 identifies as the defect being fixed.
// No pin is loosened beyond what the new pricing deterministically
// produces; expected values were recorded from the actual search output
// (the "re-derive from the actual output, never weaken" rule of
// quick-261029-v9k), with the derivation in each comment.
package tiling

import (
	"math"
	"testing"

	"tcacomp/internal/ir"
)

// TestFragWindowHonestPricing pins the judge-level fix6 §3 semantics: a
// FRAG site whose raw per-thread window exceeds MaxFragPerThread
// PROMOTES (no ReasonFragBytes size decline exists anymore), the judgment
// exposes the register-block footprint for the budget (bytes ≤
// MaxFragPerThread) AND the raw window for the traffic term (rawBytes ==
// the raw factor-model window). Fixture: matmul() lhs (covered {K}) at
// K SEQ — T64 is exactly the 256 B cap (unchanged number), T128 is 512 B
// raw (declined pre-fix6 with the F6 reason — the encoding of the defect
// fix6 §3 fixes; priced 512 raw / 256 block now).
func TestFragWindowHonestPricing(t *testing.T) {
	fx, lhs, _ := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p)
	tree := treeItersOf(lf.Tree)
	defaultTile := maxTileOf(p)

	mk := func(tile int) map[ir.Node]IterDecision {
		iters := make(map[ir.Node]IterDecision, len(tree))
		for _, it := range tree {
			iters[it] = IterDecision{SoCSequential, 1, defaultTile}
		}
		for i := range window {
			iters[window[i]] = IterDecision{SoCSequential, 1, 64}
		}
		iters[window[len(window)-1]] = IterDecision{SoCSequential, 1, tile}
		return iters
	}
	ctx := blockDimCtx{blockDim: 32}

	// T64: 4 B × 64 = 256 B raw — exactly the cap. Block == raw (both
	// 256): the boundary case is unchanged by the repricing.
	j := judgeSite(siteOf(t, lf, lhs), mk(64), ctx, p)
	if !j.promoted || j.class != ClassFRAG || j.bytes != 256 || j.reason != "" {
		t.Fatalf("a 256 B FRAG at the cap must promote with its budget footprint == 256, got (promoted %v, %s, block %d, %q)",
			j.promoted, j.class, j.bytes, j.reason)
	}
	// T128: 4 B × 128 = 512 B raw — OVER the cap. Pre-fix6 this DECLINED
	// with the F6 reason (the defect: the decline zeroed every full-
	// collapse vector, fix6 §2's 4096 B evidence). Fix6 §3: PROMOTED, the
	// budget prices the 256 B register block (the search-level traffic
	// pin below proves the traffic term keeps the raw 512 B window).
	j = judgeSite(siteOf(t, lf, lhs), mk(128), ctx, p)
	if !j.promoted || j.class != ClassFRAG || j.bytes != 256 || j.reason != "" {
		t.Fatalf("a 512 B raw FRAG must PROMOTE priced as a 256 B block (fix6 §3: no per-site size decline), got (promoted %v, %s, block %d, %q)",
			j.promoted, j.class, j.bytes, j.reason)
	}
	if j.bytes > p.MaxFragPerThread {
		t.Fatalf("the block footprint %d must stay within MaxFragPerThread %d", j.bytes, p.MaxFragPerThread)
	}
}

// TestFragWindowSearchTrafficPins the search-level half of the honest
// pricing: the objective's traffic term prices the RAW window. The T128
// vector (lhs raw 512 B, reuse 64, holders 32) must score traffic
// 512 × 63 × 32 = 1,032,192 — pre-fix6 the site declined and the vector
// scored 0 (the qwo gate-3 mechanism). rhs (covered {N,K}, raw 32768 B)
// prices its 256 B block too but is the min-density site: the kernel-wide
// Σ block (256 + 256 = 512 > 256) evicts it by value density — the
// fix5 R-EVICT semantics, unchanged, now on the block footprints.
func TestFragWindowSearchTraffic(t *testing.T) {
	fx, lhs, rhs := matmul()
	p := DefaultParams()
	lf := overlapOf(fx, p)
	window := windowOf(lf, p)
	defaultTile := maxTileOf(p)

	digits := make([]schedDigit, len(window))
	for i := range window {
		digits[i] = schedDigit{soc: SoCSequential, loc: 1, tile: 64}
	}
	digits[0] = schedDigit{soc: SoCTA, loc: 16, tile: 64}
	digits[len(digits)-1] = schedDigit{soc: SoCSequential, loc: 1, tile: 128}
	st := initialPartState().place(SoCTA, 16)
	res := evalSchedule(lf, window, digits, st, defaultTile, p, 0, 0)

	if id := lhs.BasePtr().ID(); !res.sites[id].Promoted || res.sites[id].Bytes != 256 || res.sites[id].Reuse != 64 {
		t.Fatalf("lhs must promote priced at the 256 B block (raw 512), got %+v", res.sites[id])
	}
	if id := rhs.BasePtr().ID(); res.sites[id].Promoted || res.sites[id].Reason != ReasonEvicted {
		t.Fatalf("rhs must be evicted by value density (the kernel-wide block sum 512 > 256), got %+v", res.sites[id])
	}
	if res.traffic != 512*63*32 {
		t.Fatalf("the traffic term must price the RAW window: traffic = %d, want %d (512 × 63 × 32)",
			res.traffic, 512*63*32)
	}
	if res.frag != 256 {
		t.Fatalf("the post-eviction FRAG budget sum = %d, want 256 (the ONE block footprint within MaxFragPerThread)", res.frag)
	}
}

// TestFragWindowCollapseBeatsCeremony is the fix6 §3 HEADLINE pin: on the
// scale_copy const-K fixture (K const 4 — R4's tierEff bites the
// ceremony) the all-SEQ collapse vector is the searched winner and its
// score' exceeds the R4-penalized ceremony vector's. RE-DERIVED under the
// fix6 §3 ruling from the actual search output (derivation below).
//
// The winner (probe-recorded): R=CTA(16,T256) C=SEQ(1,T256) K=SEQ(1,T256)
// — blockDim 32, grid 16. Both loads' covered [C,K] prices a RAW
// 256×256×4 = 262144 B/thread FRAG window; the budget prices each at the
// 256 B register block (min(262144, 256)); the reuse is the R inside walk
// ceil(256/32) = 8 per thread; holders 32. Per-site traffic 262144×7×32 =
// 58,720,256 — the kernel-wide block sum 256+256 = 512 > 256 evicts one
// site by value density (the tied density/bytes fall to CS order), the
// survivor scores 58,720,256. tierEff: no WARP-covered dim → exactly 1.
// Occupancy (the SHARED occupancyOf): blockDim 32 → 48 CTAs/SM, capacity
// 4032, active min(16, ceil(4096/256)) = 16, fill 16/4032, idleFrac 0 →
// factor 0.15+0.85×16/4032. score' = 58,720,256 × factor ≈ 9.0061e6.
//
// The ceremony vector R=CTA(16,T256) C=SEQ(T32) K=WARP(2,T32) keeps both
// loads promoted FRAG 128 B reuse 4 (raw == block, uncapped): traffic
// 2×128×3×64 = 49,152; tierEff min(1, 4/(32·2)) = 0.0625; occupancy at
// blockDim 64 (capacity 2016, fill 16/2016) → score' ≈ 481.5 — the
// historical qwo unit-fixture number. The collapse class beats it by
// ~4 orders of magnitude: the deadlock fix6 §2 measured is unlocked.
func TestFragWindowCollapseBeatsCeremony(t *testing.T) {
	fx, x, z := scaleCopyFx(t, true) // K const 4 — R4's tierEff bites
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	// The collapse class is the argmax: K=SEQ, and no SURVIVING dim
	// carries WARP (the §3.1 prune stays intact — C must not be WARP).
	want := []IterDecision{
		{SoC: SoCTA, LoC: 16, TileSize: 256},
		{SoC: SoCSequential, LoC: 1, TileSize: 256},
		{SoC: SoCSequential, LoC: 1, TileSize: 256},
	}
	for i, w := range want {
		if d := decisionOf(t, rec, fx.iters[i]); d != w {
			t.Fatalf("fix6 §3 winner: iter %d must be %+v (the all-SEQ collapse, derivation in the comment), got %+v", i, w, d)
		}
	}
	if d := decisionOf(t, rec, fx.iters[1]); d.SoC == SoCWARP {
		t.Fatalf("the surviving dim must not carry WARP under the §3.1 prune, got %s/%d/T%d", d.SoC, d.LoC, d.TileSize)
	}
	// Exactly one site survives the kernel-wide block eviction (512 > 256
	// evicts by value density; the tie falls to CS order — site x first);
	// the survivor promotes FRAG at the 256 B block, reuse 8.
	sx, okX := rec.SiteDecisionOf(x)
	sz, okZ := rec.SiteDecisionOf(z)
	if !okX || !okZ {
		t.Fatal("both loads must carry record entries")
	}
	promoted, evicted := 0, 0
	for _, sd := range []SiteDecision{sx, sz} {
		if sd.Promoted {
			promoted++
			if sd.Class != ClassFRAG || sd.Bytes != 256 || sd.Reuse != 8 {
				t.Fatalf("the surviving collapse site must promote FRAG 256 B block reuse 8, got %+v", sd)
			}
		} else {
			evicted++
			if sd.Reason != ReasonEvicted {
				t.Fatalf("the collapsed-on site must demote with ReasonEvicted, got %+v", sd)
			}
		}
	}
	if promoted != 1 || evicted != 1 {
		t.Fatalf("exactly one of the two collapse sites must survive the block eviction, got promoted %d evicted %d", promoted, evicted)
	}

	// Score comparison by direct evalSchedule over both candidate
	// vectors — the winner's score' must exceed the ceremony's.
	lf := overlapOf(fx, p)
	lf.reduced = reducedIters(fx.g)
	window := windowOf(lf, p)
	eval := func(digits []schedDigit) schedResult {
		st := initialPartState()
		for _, d := range digits {
			st = st.place(d.soc, d.loc)
		}
		return evalSchedule(lf, window, digits, st, maxTileOf(p), p, 0, 0)
	}
	collapse := eval([]schedDigit{
		{soc: SoCTA, loc: 16, tile: 256},
		{soc: SoCSequential, loc: 1, tile: 256},
		{soc: SoCSequential, loc: 1, tile: 256},
	})
	ceremony := eval([]schedDigit{
		{soc: SoCTA, loc: 16, tile: 256},
		{soc: SoCSequential, loc: 1, tile: 32},
		{soc: SoCWARP, loc: 2, tile: 32},
	})
	if collapse.traffic != 58720256 {
		t.Fatalf("collapse traffic = %d, want 58720256 (262144 raw × 7 × 32 — the RAW window prices, uncapped)", collapse.traffic)
	}
	if ceremony.traffic != 49152 {
		t.Fatalf("ceremony traffic = %d, want 49152 (2 × 128 × 3 × 64)", ceremony.traffic)
	}
	if collapse.score <= ceremony.score {
		t.Fatalf("the collapse score' %v must EXCEED the ceremony score' %v (the fix6 §2 deadlock unlocked)", collapse.score, ceremony.score)
	}
	// The exact scores via the SHARED occupancyOf — the same composition
	// the search applies (traffic × tierEff × occupancy), never a
	// re-derived formula.
	wantCollapse := float64(58720256) * occupancyOf(collapse.iters, collapse.smem, p).factor
	wantCeremony := float64(49152) * 0.0625 * occupancyOf(ceremony.iters, ceremony.smem, p).factor
	if math.Abs(collapse.score-wantCollapse) > 1e-6*wantCollapse || math.Abs(ceremony.score-wantCeremony) > 1e-6*wantCeremony {
		t.Fatalf("score' composition mismatch: collapse %v want %v; ceremony %v want %v",
			collapse.score, wantCollapse, ceremony.score, wantCeremony)
	}
	if collapse.score < 9e6 || ceremony.score > 500 {
		t.Fatalf("score' bands drifted: collapse %v want ~9.0061e6, ceremony %v want ~481.5", collapse.score, ceremony.score)
	}
}
