// adaptive_test.go pins fix5 T1 — the R1 adaptive LoC ceiling, the R4
// tier-efficiency guard, and the K=SEQ search re-choice (quick-261029-qwo,
// design/fix/fix5.md §1 R1/R2/R4 + §5 gates).
//
// THE SINGLE SOURCE of the arithmetic under test is adaptiveLoC and
// reducedTierEff (constraints.go / judge.go) — these pins exercise those
// functions and the search's digit list directly, never re-deriving the
// formulas. The derivation comments carry the worked numbers so a pin
// failure is diagnosable without re-running the search.
//
// Effective-extent discipline (fix5 §1 R1, the SUMMARY-recorded reading):
// every formula's extent is the dim's CONST extent when one is known and
// Params.NominalExtent otherwise — the SAME discipline digitListOf's
// R-LOCCAP capOf uses, so the search's adaptive digit and tiling-verify's
// locPresetOK re-check can never disagree on a const-extent dim (const 96 /
// tile 32 yields adaptive 3, which the bare-NominalExtent formula would
// reject at verify time).
package tiling

import (
	"fmt"
	"math"
	"testing"

	"tcacomp/internal/ir"
)

// scaleCopyFx builds a scale_copy-shaped fixture (design/fix/fix5.md §0):
// two outer DYNAMIC-extent dims (R, C — TensorDimOf bounds, exactly what
// the real frontend pipeline produces for shape=[R,C,K] kernels) and one
// tiny reduced dim K. kConst selects a const K extent (4 — the real
// kernel's runtime K) vs the dynamic stand-in. Two rank-3 loads x,z cover
// [R][C][K]; a K-reduce feeds the [R][C] single-owner store.
//
// The CURRENT (pre-fix5) winner on this fixture is the fix5 §0 ceremony
// vector R=CTA(T=256,LoC=16) C=SEQ(T=32) K=WARP(T=32,K=2) with both loads
// promoted FRAG 128 B reuse 4 (probe-verified against the emitted
// scale_copy .cu: 16 CTAs x 64 threads, WarpReducer bracket).
func scaleCopyFx(t *testing.T, kConst bool) (fixture, ir.Node, ir.Node) {
	t.Helper()
	g := ir.NewGraph("scalecopy")
	u32 := ir.InternScalar(ir.TU32)
	var dims, iters []ir.Node
	for i := 0; i < 3; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	xP := g.NewInputParam("ctor", ir.RoleTensor, "x",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	zP := g.NewInputParam("ctor", ir.RoleTensor, "z",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}}))
	var parent ir.Node
	for i := 0; i < 3; i++ {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		var bound ir.Node
		if i == 2 && kConst {
			bound = g.NewConst("ctor", u32, 4) // the tiny reduced dim (scale_copy's runtime K)
		} else {
			bound = g.NewTensorDimOf("ctor", xP, i) // dynamic — the real pipeline's shape
		}
		biv := g.NewBoundedDimVar("ctor", dv, bound)
		parent = g.NewIter("ctor", biv, parent)
		iters = append(iters, parent)
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }
	x := g.NewMemLoad("ctor", xP, []ir.Node{iv(0), iv(1), iv(2)})
	z := g.NewMemLoad("ctor", zP, []ir.Node{iv(0), iv(1), iv(2)})
	prod := g.NewBinOp("ctor", ir.BinOpMul, x, z)
	acc := g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := g.NewBinOp("ctor", ir.BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	val := g.NewReduce("ctor", g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), prod,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1)}, val)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return fixture{g: g, dims: dims, out: out, iters: iters}, x, z
}

// TestAdaptiveLoCFormula pins the R1 shared formula (fix5 §1):
//
//	LoC_adaptive(d) = clamp(ceil(extent / T_d), 2, MaxCTA)          // CTA dims
//	K_adaptive(d)   = clamp(ceil(extent / T_d), 2, MaxWarpsPerCTA)  // WARP dims
//
// with extent the EFFECTIVE extent (const extent or the NominalExtent
// stand-in — the caller passes whichever applies; adaptiveLoC itself is
// extent-agnostic). Pins: the headline values, the floor clamp (tile >=
// extent -> ceil 1 -> 2), and the cap clamp (tile=8, extent=4096 -> 512
// CTA-clamped, 32 WARP-clamped).
func TestAdaptiveLoCFormula(t *testing.T) {
	p := DefaultParams() // MaxCTA 512, MaxWarpsPerCTA 32, NominalExtent 4096

	// Headline: tile=32, extent=4096 -> ceil(4096/32) = 128.
	if got := adaptiveLoC(SoCTA, 32, 4096, p); got != 128 {
		t.Fatalf("adaptiveLoC(CTA, tile 32, extent 4096) = %d, want 128 (clamp(ceil(4096/32)=128, 2, 512))", got)
	}
	// WARP cap: 128 clamped by MaxWarpsPerCTA 32.
	if got := adaptiveLoC(SoCWARP, 32, 4096, p); got != 32 {
		t.Fatalf("adaptiveLoC(WARP, tile 32, extent 4096) = %d, want 32 (clamp(128, 2, 32))", got)
	}
	// Floor: tile >= extent -> ceil 1 -> clamped to 2.
	if got := adaptiveLoC(SoCTA, 64, 32, p); got != 2 {
		t.Fatalf("adaptiveLoC(CTA, tile 64, extent 32) = %d, want 2 (clamp(ceil(32/64)=1, 2, 512))", got)
	}
	if got := adaptiveLoC(SoCWARP, 4096, 4096, p); got != 2 {
		t.Fatalf("adaptiveLoC(WARP, tile 4096, extent 4096) = %d, want 2 (floor clamp)", got)
	}
	// Cap: tile=8, extent=4096 -> ceil 512 -> CTA keeps 512 (MaxCTA), WARP clamps to 32.
	if got := adaptiveLoC(SoCTA, 8, 4096, p); got != 512 {
		t.Fatalf("adaptiveLoC(CTA, tile 8, extent 4096) = %d, want 512 (clamp(512, 2, MaxCTA 512))", got)
	}
	if got := adaptiveLoC(SoCWARP, 8, 4096, p); got != 32 {
		t.Fatalf("adaptiveLoC(WARP, tile 8, extent 4096) = %d, want 32 (clamp(512, 2, 32))", got)
	}
	// Const-extent path: extent 96, tile 32 -> ceil 3 (a NON-preset value).
	if got := adaptiveLoC(SoCTA, 32, 96, p); got != 3 {
		t.Fatalf("adaptiveLoC(CTA, tile 32, extent 96) = %d, want 3 (const-extent discipline)", got)
	}
}

// TestAdaptiveDigitListConstExtent pins the search-side emission (fix5 §2
// digitListOf): a const-extent dim (extent 96) carries its extent-derived
// adaptive top digit — 3 at tile 32, a value NO preset list contains — and
// the dedup contract: an adaptive value coinciding with a preset appears
// EXACTLY ONCE (presets win the duplicate).
func TestAdaptiveDigitListConstExtent(t *testing.T) {
	p := DefaultParams()

	count := func(digits []schedDigit, soc SoCType, loc, tile int) int {
		n := 0
		for _, d := range digits {
			if d.soc == soc && d.loc == loc && d.tile == tile {
				n++
			}
		}
		return n
	}

	// extent 96: adaptive = clamp(ceil(96/T), 2, 512) -> 3@T32, 2@T64 (dedup
	// drops the repeated 2 at T128/T256); 3 is NOT in CtaLoCPresets {2,4,...}.
	// R3 call-site update (quick-261029-v9k): the queried dim is passed as
	// its own warpDim winner — the CTA-section assertions are winner-
	// independent, and the pin keeps exercising the full per-dim list.
	fx := newChain("adapt96", 96, 4096, 4096)
	digits := digitListOf(fx.iters[0], p, nil, fx.iters[0])
	if n := count(digits, SoCTA, 3, 32); n != 1 {
		t.Fatalf("digitListOf(extent 96) must carry the adaptive (CTA, 3, T32) exactly once, got %d", n)
	}
	if n := count(digits, SoCTA, 2, 64); n != 1 {
		t.Fatalf("digitListOf(extent 96) must carry (CTA, 2, T64) exactly once (adaptive dedup), got %d", n)
	}
	// Deterministic placement: the adaptive digit sits AFTER that section's
	// preset digits (per tile, presets enumerated first).
	firstAdapt, firstPreset := -1, -1
	for i, d := range digits {
		if d.soc == SoCTA && d.loc == 3 && firstAdapt < 0 {
			firstAdapt = i
		}
		if d.soc == SoCTA && d.loc == 2 && d.tile == 32 && firstPreset < 0 {
			firstPreset = i
		}
	}
	if firstAdapt < 0 || firstPreset < 0 || firstAdapt < firstPreset {
		t.Fatalf("adaptive digit placement: preset at %d, adaptive at %d — presets must come first", firstPreset, firstAdapt)
	}

	// Dedup against presets: extent 512, tile 32 -> adaptive 16, which IS a
	// preset — the pair must appear exactly once. (R3: the dim passed as
	// its own winner — CTA assertions are winner-independent.)
	fx2 := newChain("adapt512", 512, 4096, 4096)
	if n := count(digitListOf(fx2.iters[0], p, nil, fx2.iters[0]), SoCTA, 16, 32); n != 1 {
		t.Fatalf("digitListOf(extent 512) must carry (CTA, 16, T32) exactly once (preset wins the adaptive duplicate), got %d", n)
	}
}

// TestLocPresetOKAdaptive pins the R1 acceptance path (fix5 §2
// constraints.go): the grown signature carries the tile and the effective
// extent; acceptance is preset membership OR equality with the shared
// adaptiveLoC. The (CTA, 3, T32, extent 96) case is the search/verify
// lockstep pin: the search's digitListOf emits 3 via the const-extent
// formula, so tiling-verify's re-check MUST accept it via the same formula
// (the bare-NominalExtent reading would give 128 and reject the record).
func TestLocPresetOKAdaptive(t *testing.T) {
	p := DefaultParams()

	// Adaptive acceptance: 3 == adaptiveLoC(CTA, 32, 96) and 3 is no preset.
	if !locPresetOK(SoCTA, 3, 32, 96, p) {
		t.Fatal("locPresetOK(CTA, 3, T32, extent 96) must accept via the adaptive path (adaptiveLoC = 3)")
	}
	// Same loc, different extent: adaptiveLoC(CTA, 32, 512) = 16 != 3, and 3
	// is no preset -> reject.
	if locPresetOK(SoCTA, 3, 32, 512, p) {
		t.Fatal("locPresetOK(CTA, 3, T32, extent 512) must reject (adaptive gives 16 there; 3 is no preset)")
	}
	// Preset membership unchanged: 4 is a preset regardless of extent.
	if !locPresetOK(SoCTA, 4, 32, 96, p) {
		t.Fatal("locPresetOK(CTA, 4, T32, extent 96) must accept (preset membership)")
	}
	if !locPresetOK(SoCWARP, 16, 64, 512, p) {
		t.Fatal("locPresetOK(WARP, 16, T64, extent 512) must accept (preset membership)")
	}
	// The adaptive path on a WARP dim: adaptiveLoC(WARP, 32, 96) = 3 (under
	// the MaxWarpsPerCTA cap) -> accept a non-preset 3.
	if !locPresetOK(SoCWARP, 3, 32, 96, p) {
		t.Fatal("locPresetOK(WARP, 3, T32, extent 96) must accept via the adaptive path (clamp(3, 2, 32) = 3)")
	}
	// SEQ is untouched: only the implicit 1.
	if !locPresetOK(SoCSequential, 1, 32, 96, p) {
		t.Fatal("locPresetOK(SEQ, 1) must accept")
	}
	if locPresetOK(SoCSequential, 2, 32, 96, p) {
		t.Fatal("locPresetOK(SEQ, 2) must reject (SEQ carries the implicit LoC 1)")
	}
}

// TestTierEffReduced pins the R4 tier-efficiency guard (fix5 §1 R4): a
// warp-sharded REDUCED covered dim whose effective extent cannot feed its
// lanes earns no full traffic credit —
//
//	tierEff = Π over covered dims that are reduced AND WARP-assigned of
//	          min(1, extent_k / (32 · K_k))
//
// everything else contributes exactly 1. extent_k is the const extent or
// the NominalExtent stand-in (same discipline as R1).
func TestTierEffReduced(t *testing.T) {
	p := DefaultParams()

	fx, _, _ := scaleCopyFx(t, true) // K const 4 (reduced), R/C dynamic
	r, c, k := fx.iters[0], fx.iters[1], fx.iters[2]
	decide := func(kSoc SoCType, kLoc int) map[ir.Node]IterDecision {
		return map[ir.Node]IterDecision{
			r: {SoC: SoCTA, LoC: 16, TileSize: 256},
			c: {SoC: SoCSequential, LoC: 1, TileSize: 32},
			k: {SoC: kSoc, LoC: kLoc, TileSize: 32},
		}
	}
	reduced := map[ir.Node]bool{k: true}

	// K reduced, WARP/2, const extent 4: min(1, 4/(32·2)) = 1/16 = 0.0625.
	if got := reducedTierEff(reduced, decide(SoCWARP, 2), []ir.Node{k}, p); math.Abs(got-0.0625) > 1e-12 {
		t.Fatalf("tierEff(reduced WARP/2, extent 4) = %v, want 0.0625 (min(1, 4/64))", got)
	}
	// K reduced but SEQ: exactly 1 (no lane sharding -> no fragmentation).
	if got := reducedTierEff(reduced, decide(SoCSequential, 1), []ir.Node{k}, p); got != 1 {
		t.Fatalf("tierEff(reduced SEQ) = %v, want exactly 1", got)
	}
	// Covered dim NOT reduced (C WARP-assigned): contributes 1 even though
	// it is warp-sharded — R4 guards reduced dims only.
	if got := reducedTierEff(reduced, decide(SoCWARP, 2), []ir.Node{c}, p); got != 1 {
		t.Fatalf("tierEff(non-reduced covered WARP dim) = %v, want exactly 1", got)
	}
	// Covered [C, K] with K the only reduced warp dim: 1 × 0.0625.
	if got := reducedTierEff(reduced, decide(SoCWARP, 2), []ir.Node{c, k}, p); math.Abs(got-0.0625) > 1e-12 {
		t.Fatalf("tierEff(covered [C, K]) = %v, want 0.0625 (C contributes 1)", got)
	}

	// A fed lane army: extent 64, K 2 -> min(1, 64/64) = 1.
	g64 := ir.NewGraph("teff64")
	d64 := g64.NewInputParam("ctor", ir.RoleDim, "d", nil)
	x64 := g64.NewInputParam("ctor", ir.RoleTensor, "x",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d"}}))
	dv64 := g64.NewDimVar("ctor", d64, g64.NewConst("ctor", ir.InternScalar(ir.TU32), 0), g64.NewConst("ctor", ir.InternScalar(ir.TU32), 1))
	biv64 := g64.NewBoundedDimVar("ctor", dv64, g64.NewConst("ctor", ir.InternScalar(ir.TU32), 64))
	it64 := g64.NewIter("ctor", biv64, nil)
	_ = g64.NewMemLoad("ctor", x64, []ir.Node{g64.NewInductionVariable("ctor", it64)})
	if got := reducedTierEff(map[ir.Node]bool{it64: true},
		map[ir.Node]IterDecision{it64: {SoC: SoCWARP, LoC: 2, TileSize: 32}}, []ir.Node{it64}, p); got != 1 {
		t.Fatalf("tierEff(reduced WARP/2, extent 64) = %v, want exactly 1 (min(1, 64/64))", got)
	}

	// Dynamic-extent reduced dim: the NominalExtent stand-in feeds the
	// formula — min(1, 4096/64) = 1 (no scaling; the stand-in extent feeds
	// every lane).
	fxDyn, _, _ := scaleCopyFx(t, false)
	kDyn := fxDyn.iters[2]
	if got := reducedTierEff(map[ir.Node]bool{kDyn: true},
		map[ir.Node]IterDecision{kDyn: {SoC: SoCWARP, LoC: 2, TileSize: 32}}, []ir.Node{kDyn}, p); got != 1 {
		t.Fatalf("tierEff(dynamic reduced dim) = %v, want exactly 1 (NominalExtent stand-in 4096/64)", got)
	}

	// The product semantics: two reduced WARP-sharded covered dims multiply —
	// extents 4 and 8 at K 2 each: 0.0625 × min(1, 8/64) = 0.0625 × 0.125.
	fx2 := newChain("teff2", 4, 8)
	k1, k2 := fx2.iters[0], fx2.iters[1]
	red2 := map[ir.Node]bool{k1: true, k2: true}
	dec2 := map[ir.Node]IterDecision{
		k1: {SoC: SoCWARP, LoC: 2, TileSize: 32},
		k2: {SoC: SoCWARP, LoC: 2, TileSize: 32},
	}
	if got := reducedTierEff(red2, dec2, []ir.Node{k1, k2}, p); math.Abs(got-0.0625*0.125) > 1e-12 {
		t.Fatalf("tierEff(two reduced warp dims) = %v, want %v (0.0625 × 0.125)", got, 0.0625*0.125)
	}
}

// TestSearchRechoiceCollapse pins the fix5 §4 headline: under R1+R4 the
// scale_copy-shaped LNT RE-CHOOSES away from the warp-collective ceremony
// vector to the K=SEQ collapse vector — NO WARP digit on the reduced dim
// (the one-tier ladder of R2: SeqReducer, zero ceremony), with the loads
// still promoted.
//
// Worked derivation (kConst K=4 fixture; the winner was re-derived from
// the actual search output, per the n63 "re-derive, never weaken" rule):
//
//   - Ceremony vector R=CTA(16,T256) C=SEQ(T32) K=WARP(2,T32): both loads
//     promote FRAG 128 B reuse 4 (holders = blockDim 64) -> traffic 49152;
//     R4 cuts it: tierEff = min(1, 4/(32·2)) = 0.0625 -> scaledTraffic
//     3072; grid 16 fully active, factor ≈ 0.1567 -> score' ≈ 481.5.
//   - Collapse vector R=CTA(16,T256) C=WARP(2,T32) K=SEQ(T32): K leaves
//     the warp layer, so tierEff = 1 and the FRAG windows survive (C's
//     lane chunk 1 × K's SEQ 32 × 4 B = 128 B <= 256): traffic 49152
//     UNSCALED, factor identical (same grid, same blockDim 64) -> score'
//     ≈ 7704.9 — strictly greater than every ceremony variant.
//   - The R1 adaptive digits (> 64 at T <= 64) exist in the menu but no
//     winning vector can carry them: at T_r <= 64 the reuse term
//     ceil(T_r/blockDim) with blockDim >= 64 collapses to 1 (no traffic),
//     and the blockDim-32 full-collapse declines its FRAG sites (4096 B >
//     256 B). The adaptive digit's role here is menu completeness (a
//     full-grid candidate exists at every tile), pinned by
//     TestAdaptiveDigitListConstExtent.
//
// Note the REAL suite kernels carry dynamic K extents, under which R4's
// stand-in is 1 (pinned in TestTierEffReduced) — the GPU-level consequence
// is recorded in the task SUMMARY's benchmark table.
func TestSearchRechoiceCollapse(t *testing.T) {
	fx, x, z := scaleCopyFx(t, true) // K const 4 — R4's tierEff bites
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	// The collapse property: NO WARP digit on the reduced dim.
	if !reducedIters(fx.g)[fx.iters[2]] {
		t.Fatal("fixture setup: iter 2 must be the reduced dim")
	}
	if d := decisionOf(t, rec, fx.iters[2]); d.SoC == SoCWARP {
		t.Fatalf("fix5 T1 re-choice: the reduced dim must leave the warp layer (K=SEQ collapse), got %s/%d/T%d", d.SoC, d.LoC, d.TileSize)
	}
	// The re-derived winner (derivation above): R=CTA(16,T256) C=WARP(2,T32)
	// K=SEQ(T32) — collapse with the loads still promoted.
	want := []IterDecision{
		{SoC: SoCTA, LoC: 16, TileSize: 256},
		{SoC: SoCWARP, LoC: 2, TileSize: 32},
		{SoC: SoCSequential, LoC: 1, TileSize: 32},
	}
	for i, w := range want {
		if d := decisionOf(t, rec, fx.iters[i]); d != w {
			t.Fatalf("re-choice winner: iter %d must be %+v, got %+v (derivation in the test comment)", i, w, d)
		}
	}
	// Both loads stay promoted FRAG 128 B reuse 4 — the collapse keeps the
	// traffic, it just stops paying for the warp-collective fragmentation.
	for _, ld := range []ir.Node{x, z} {
		sd, ok := rec.SiteDecisionOf(ld)
		if !ok || !sd.Promoted || sd.Class != ClassFRAG || sd.Bytes != 128 || sd.Reuse != 4 {
			t.Fatalf("re-choice winner: load %d must stay promoted FRAG 128 B reuse 4, got %+v (ok=%v)", ld.BasePtr().ID(), sd, ok)
		}
	}

	// R4 is selection-only: the raw int traffic pin survives UNSCALED.
	// Evaluate the ceremony vector directly: traffic must stay 49152 while
	// score' carries the 0.0625 tierEff cut.
	lf := overlapOf(fx, p)
	lf.reduced = reducedIters(fx.g) // Pass.Run populates this; overlapOf alone does not
	window := windowOf(lf, p)
	digits := []schedDigit{
		{soc: SoCTA, loc: 16, tile: 256},
		{soc: SoCSequential, loc: 1, tile: 32},
		{soc: SoCWARP, loc: 2, tile: 32},
	}
	st := initialPartState().place(SoCTA, 16).place(SoCSequential, 1).place(SoCWARP, 2)
	res := evalSchedule(lf, window, digits, st, maxTileOf(p), p, 0, 0)
	if res.traffic != 49152 {
		t.Fatalf("ceremony vector raw traffic = %d, want 49152 UNSCALED (bytes 128 x (reuse-1) 3 x holders 64 x 2 sites)", res.traffic)
	}
	wantScore := float64(49152) * 0.0625 * occupancyOf(res.iters, res.smem, p).factor
	if math.Abs(res.score-wantScore) > 1e-6 {
		t.Fatalf("ceremony vector score' = %v, want %v (traffic x tierEff 0.0625 x the SHARED occupancyOf factor)", res.score, wantScore)
	}
}
