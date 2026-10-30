// r3_test.go pins fix5 T2 — R3 warp/lane axis freedom with extent-aware
// placement (quick-261029-v9k, design/fix/fix5.md §1 R3 ruling + §6.3 +
// §2 digitListOf + §5 gates). R3 frees the per-LNT WARP digit from the
// reduced dim and enumerates it ONLY on the single extent-aware-chosen
// winner dim:
//
//	L1: effective extent DESC (constExtentOf const value, else
//	    Params.NominalExtent — the SAME discipline as R1's adaptiveLoC and
//	    R4's reducedTierEff)
//	L2: stride-1 set membership first — stride1Set is the union over the
//	    LNT's CS sites of the iters at each site's INNERMOST subscript
//	    position (the last non-empty AxesIters entry — the coalescing
//	    axis); in-set dims rank above out-of-set dims, and WITHIN the set
//	    the innermost WINDOW position wins
//	L3: non-reduced before reduced (the reduced axis as last-resort
//	    fallback — fires only on residuals L1/L2 left tied)
//	L4: innermost window position (final total fallback — window order is
//	    unique, so the comparator always resolves, never panics)
//
// THE SINGLE SOURCE of the ranking is warpDimOf (search.go) — these pins
// exercise the helper, the restricted digitListOf, and the search-level
// outcomes, never re-deriving the comparator. Worked derivations are in
// the pin comments so a failure is diagnosable without re-running.
//
// Worked outcomes locked here (fix5.md §1 R3 as amended, session-2 ruling):
//   - suite matmul shape, ALL dynamic (lhs[i][k], rhs[k][j], reduce k):
//     L1 ties {i,j,k} at the stand-in -> L2 stride1Set = {k (lhs
//     innermost), j (rhs innermost)} -> within-set innermost window
//     position -> k (the reduce dim, innermost) -> warps-on-K split-K
//     preserved BY CONSTRUCTION (the family-B guard pin).
//   - frontend shape, const {64,128,64} reduced=N: L1 -> N (128 largest)
//     -> the fePinScheduleSignature e2e split-k winner survives.
//   - scale_copy shape with the tiny reduced dim CONST (k=4, the spec's
//     worked example): L1 eliminates k -> {r,c} -> L2 stride1Set from the
//     3D loads is {k} -> both out -> L3 both non-reduced -> L4 innermost
//     among {r,c} -> c -> warps-on-C, K=SEQ: the T1 collapse vector wins
//     and the qwo ceremony vector is UNREPRESENTABLE.
//
// Honest dynamic-extent consequence (the gate-3 mechanism, recorded in
// the task SUMMARY): with the reduced dim ALSO dynamic, L1 ties and L2's
// stride1Set {k} selects the innermost reduced dim — the ceremony vector
// stays representable. The real suite kernels carry dynamic K, so the
// re-choice additionally needs a compile-time K extent (const-shape
// specialization is fix6's D-fix6-01, explicitly out of scope here).
package tiling

import (
	"fmt"
	"testing"

	"tcacomp/internal/ir"
)

// r3DynMatmul builds the suite-matmul shape with DYNAMIC extents (the
// family-B class): lhs[i][k], rhs[k][j], reduce over k, store out[i][j].
// All three BIV bounds are TensorDimOf nodes — constExtentOf reports
// non-const for every dim, exactly the real suite kernels' IR.
func r3DynMatmul(t *testing.T) (fixture, ir.Node, ir.Node) {
	t.Helper()
	g := ir.NewGraph("r3dynmm")
	u32 := ir.InternScalar(ir.TU32)
	var dims, iters []ir.Node
	for i := 0; i < 3; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	lhsP := g.NewInputParam("ctor", ir.RoleTensor, "lhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d2"}}))
	rhsP := g.NewInputParam("ctor", ir.RoleTensor, "rhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d2"}, {Name: "d1"}}))
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}}))
	var parent ir.Node
	for i := 0; i < 3; i++ {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewTensorDimOf("ctor", lhsP, 0)) // dynamic bound
		parent = g.NewIter("ctor", biv, parent)
		iters = append(iters, parent)
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }
	lhs := g.NewMemLoad("ctor", lhsP, []ir.Node{iv(0), iv(2)})
	rhs := g.NewMemLoad("ctor", rhsP, []ir.Node{iv(2), iv(1)})
	prod := g.NewBinOp("ctor", ir.BinOpMul, lhs, rhs)
	acc := g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := g.NewBinOp("ctor", ir.BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	val := g.NewReduce("ctor", g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), prod,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1)}, val)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return fixture{g: g, dims: dims, out: out, iters: iters}, lhs, rhs
}

// TestWarpDimRanking pins warpDimOf's four-level comparator per dim level,
// on crafted windows + SiteFacts (AxesIters is the only SiteFacts field
// the ranking reads; hand-built facts keep each level's premise exact).
// The comparator is TOTAL: every case resolves, none panics.
func TestWarpDimRanking(t *testing.T) {
	p := DefaultParams()

	// (a) L1 extent win, const extents distinct: 128 carries WARP.
	fxA := newChain("r3l1", 64, 128, 64)
	if got := warpDimOf(fxA.iters, nil, nil, false, p); got != fxA.iters[1] {
		t.Fatalf("L1: warpDimOf({64,128,64}) = iter %v, want iter %d (extent 128 largest)",
			nodeID(got), fxA.iters[1].BasePtr().ID())
	}

	// (b) L1 dynamic-tie fall-through to L2 MEMBERSHIP: every dim stands in
	// at one extent (4096), so the stride-1 set decides. Both sites'
	// innermost subscript chases the OUTER window dim r, so stride1Set =
	// {r} and the in-set OUTER dim beats the innermost out-of-set dim k.
	fxB, _, _ := scaleCopyFx(t, false) // r, c, k — all dynamic
	rB := fxB.iters[0]
	csRb := []SiteFacts{{AxesIters: [][]ir.Node{{rB}, {rB}}}, // 2D-ish load: innermost r
		{AxesIters: [][]ir.Node{{rB}, {rB}, {rB}}}} // 3D load: innermost r
	if got := warpDimOf(fxB.iters, csRb, nil, false, p); got != rB {
		t.Fatalf("L2 membership: warpDimOf must take the in-set OUTER dim r over the innermost out-of-set k, got iter %v", nodeID(got))
	}

	// (c) L2 within-set innermost-window win on the REAL matmul SiteFacts
	// (not hand-built): lhs[i][k] innermost k, rhs[k][j] innermost j ->
	// stride1Set {k, j}; window [i, j, k] -> k (innermost) wins even though
	// j is the rhs/store coalescing axis.
	fxC, _, _ := r3DynMatmul(t)
	lfC := overlapOf(fxC, p)
	lfC.reduced = reducedIters(fxC.g)
	windowC := windowOf(lfC, p)
	if got := warpDimOf(windowC, lfC.CS, reducedAxesOf(fxC.g), true, p); got != fxC.iters[2] {
		t.Fatalf("L2 within-set: warpDimOf(dyn matmul) = iter %v, want iter %d (k, innermost of the stride-1 set {k,j})",
			nodeID(got), fxC.iters[2].BasePtr().ID())
	}

	// (d) L3 reduced-last-resort demotion on an L2-empty residual: equal
	// const extents, no candidate sites, the INNER dim reduced -> the
	// non-reduced OUTER dim wins.
	fxD := newChain("r3l3", 64, 64)
	if got := warpDimOf(fxD.iters, nil, map[ir.Node]bool{fxD.iters[1]: true}, false, p); got != fxD.iters[0] {
		t.Fatalf("L3: warpDimOf({64,64}, reduced=inner) = iter %v, want iter %d (non-reduced wins)",
			nodeID(got), fxD.iters[0].BasePtr().ID())
	}

	// (e) L4 window-order residual: equal extents, no sites, no reduced ->
	// the innermost window dim wins (the total fallback).
	fxE := newChain("r3l4", 64, 64)
	if got := warpDimOf(fxE.iters, nil, nil, false, p); got != fxE.iters[1] {
		t.Fatalf("L4: warpDimOf({64,64}) = iter %v, want iter %d (innermost fallback)",
			nodeID(got), fxE.iters[1].BasePtr().ID())
	}

	// The frontend-fixture ranking lock (unit level): const {64,128,64}
	// with lhs[a][b], rhs[b][c] — L1 picks b (N=128) outright, before any
	// stride-1 consideration. The e2e fePinScheduleSignature pin (which
	// asserts N=WARP on the emitted .cu) must survive unchanged.
	fxFE := newChain("r3fe", 64, 128, 64)
	aFE, bFE, cFE := fxFE.iters[0], fxFE.iters[1], fxFE.iters[2]
	csFE := []SiteFacts{{AxesIters: [][]ir.Node{{aFE}, {bFE}}}, // lhs[a][b]
		{AxesIters: [][]ir.Node{{bFE}, {cFE}}}} // rhs[b][c]
	if got := warpDimOf(fxFE.iters, csFE, map[ir.Node]bool{bFE: true}, true, p); got != bFE {
		t.Fatalf("frontend ranking: warpDimOf({64,128,64}) = iter %v, want iter %d (N wins L1)", nodeID(got), bFE.BasePtr().ID())
	}

	// The honest dynamic-extent consequence, at ranking level: on the
	// fully-dynamic scale_copy shape the spec'd ranking itself selects the
	// innermost reduced dim k (L1 ties at the stand-in, stride1Set {k}) —
	// the family-B preservation mechanism, and the reason the elementwise
	// re-choice needs a compile-time K extent (gate-3 mechanism in the
	// task SUMMARY, qwo honest-recording precedent).
	fxSC, _, _ := scaleCopyFx(t, false)
	lfSC := overlapOf(fxSC, p)
	lfSC.reduced = reducedIters(fxSC.g)
	kSC := fxSC.iters[2]
	if got := warpDimOf(windowOf(lfSC, p), lfSC.CS, reducedAxesOf(fxSC.g), true, p); got != kSC {
		t.Fatalf("dynamic scale_copy ranking: warpDimOf = iter %v, want iter %d (k via L2)", nodeID(got), kSC.BasePtr().ID())
	}
}

// nodeID renders an iter's base id for diagnostics (nil-safe).
func nodeID(n ir.Node) string {
	if n == nil || n.BasePtr() == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%d", n.BasePtr().ID())
}

// TestWarpDimRestriction pins digitListOf's R3 gate: the WARP preset
// section AND the R1 adaptiveSection(SoCWARP) emit ONLY for the winner
// dim; every other dim's list carries ZERO SoCWARP digits; the R-REDUCED
// prune stands on top (winner reduced + ReducedDimWarp=false -> no WARP
// section at all — narrower than the pre-R3 flag=false semantics, where
// non-reduced dims kept their WARP digits: fix5 §1 R3 consequence).
func TestWarpDimRestriction(t *testing.T) {
	p := DefaultParams()
	fx, _, _ := scaleCopyFx(t, true) // r, c dynamic; k const 4 (reduced)
	r, c, k := fx.iters[0], fx.iters[1], fx.iters[2]
	reduced := map[ir.Node]bool{k: true}

	warpDigits := func(digits []schedDigit) int {
		n := 0
		for _, d := range digits {
			if d.soc == SoCWARP {
				n++
			}
		}
		return n
	}

	// Winner c: presets + the R1 adaptive digit all present.
	wc := warpDigits(digitListOf(c, p, reduced, c))
	if wc == 0 {
		t.Fatal("the winner dim's digit list must carry the WARP section (presets + R1 adaptive)")
	}
	// Non-winners r and k: ZERO WARP digits — even though k is reduced and
	// ReducedDimWarp=true (the R3 restriction binds before capability).
	if n := warpDigits(digitListOf(r, p, reduced, c)); n != 0 {
		t.Fatalf("a non-winner dim's list must carry ZERO SoCWARP digits, got %d", n)
	}
	if n := warpDigits(digitListOf(k, p, reduced, c)); n != 0 {
		t.Fatalf("the reduced non-winner dim's list must carry ZERO SoCWARP digits, got %d", n)
	}

	// R-REDUCED on top: winner reduced + ReducedDimWarp=false -> no WARP
	// section at all (the pre-R3 semantics kept WARP on non-reduced dims
	// here; R3 narrows the flag=false area to no-WARP-anywhere).
	noWarp := DefaultParams()
	noWarp.ReducedDimWarp = false
	if n := warpDigits(digitListOf(k, noWarp, reduced, k)); n != 0 {
		t.Fatalf("reduced winner under ReducedDimWarp=false must carry NO WARP digits, got %d", n)
	}
	// Sanity of the winner path under the default params: k (reduced) MAY
	// carry WARP only when it is the winner (matmul split-k class).
	if n := warpDigits(digitListOf(k, p, reduced, k)); n == 0 {
		t.Fatal("the reduced winner dim under ReducedDimWarp=true must keep its WARP section (split-k capability)")
	}
}

// TestR3SearchRechoiceConstK pins the §3.1-pruned search outcome on the
// scale_copy shape with the tiny reduced dim CONST (k=4 — the fixture
// where R4's tierEff bites). RE-DERIVED under the §3.1 ruling (author,
// 2026-10-30, iq6 candidate-1): the pre-ruling pin asserted the C=WARP
// collapse winner — a SURVIVING-WARP vector, GPU-proven broken (row_sum/
// softmax canary 1/64). Under the prune the collapse vector is
// UNREPRESENTABLE, and the honest winner is the R4-PENALIZED CEREMONY
// vector R=CTA(16,T256) C=SEQ(T32) K=WARP(2,T32): it is the only remaining
// positive-traffic class — the all-SEQ collapse's covered [C,K] prices a
// 32x32x4 = 4096 B/thread FRAG window > MaxFragPerThread 256 B (the qwo
// gate-3 mechanism, FRAG-declined -> traffic 0 -> score' 0). The K=SEQ
// collapse re-choice therefore needs fix6's work-shape model (FRAG
// windows > 1x1 pricing) IN ADDITION to the §3.1 lowering; recorded in
// the task SUMMARY, never chased out of scope. The loads stay promoted
// (qwo's pin family) — K=WARP keeps the sites' covered set legal.
func TestR3SearchRechoiceConstK(t *testing.T) {
	fx, _, _ := scaleCopyFx(t, true)
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	reduced := reducedIters(fx.g)
	if !reduced[fx.iters[2]] {
		t.Fatal("fixture setup: iter 2 (k) must be the reduced dim")
	}
	want := []IterDecision{
		{SoC: SoCTA, LoC: 16, TileSize: 256},
		{SoC: SoCSequential, LoC: 1, TileSize: 32},
		{SoC: SoCWARP, LoC: 2, TileSize: 32},
	}
	for i, w := range want {
		if d := decisionOf(t, rec, fx.iters[i]); d != w {
			t.Fatalf("pruned re-choice winner: iter %d must be %+v (the R4-penalized ceremony, derivation in the comment), got %+v", i, w, d)
		}
	}
	// The surviving-WARP class is unrepresentable: C must not carry WARP.
	if d := decisionOf(t, rec, fx.iters[1]); d.SoC == SoCWARP {
		t.Fatalf("the surviving dim must not carry WARP under the §3.1 prune, got %+v", d)
	}
}

// TestR3FamilyBGuard pins the n63 split-K preservation BY CONSTRUCTION:
// on the suite-matmul shape with ALL extents dynamic, the R3 winner is
// EXACTLY the current-HEAD winner (recorded by running the search at the
// T2 parent — the winner vector below), because the ranking's L2 picks
// the innermost reduced dim k (stride1Set {k, j}, k innermost in the
// window). If this pin is RED after GREEN, the ranking is wrong — fix the
// ranking, NEVER weaken this pin.
//
// Recorded current-HEAD winner (probe at the T2 parent fdf8ef3):
// i=CTA(128,T32) j=SEQ(T256) k=WARP(32,T256) — blockDim 1024, the split-k
// class; lhs promoted FRAG 32 B reuse 256; rhs's 8 MB K-slab window
// (K=32 x T256 x 4 B, the adaptive warp digit at T256) overflows the 64 KB
// budget and is evicted by value density (fix4 2.6). R3 leaves k's digit
// list untouched (k IS the winner) and only drops WARP sections from
// i/j's lists — sections the recorded winner never used — so the argmax
// is invariant and the vector survives bit-identically.
func TestR3FamilyBGuard(t *testing.T) {
	fx, lhs, rhs := r3DynMatmul(t)
	p := DefaultParams()
	rec := runPass(t, fx.g, p)

	want := []IterDecision{
		{SoC: SoCTA, LoC: 128, TileSize: 32},
		{SoC: SoCSequential, LoC: 1, TileSize: 256},
		{SoC: SoCWARP, LoC: 32, TileSize: 256},
	}
	for i, w := range want {
		if d := decisionOf(t, rec, fx.iters[i]); d != w {
			t.Fatalf("family-B guard: iter %d must stay %+v, got %+v (ranking regression — fix the ranking, never weaken this pin)", i, w, d)
		}
	}
	sl, ok := rec.SiteDecisionOf(lhs)
	if !ok || !sl.Promoted || sl.Class != ClassFRAG || sl.Bytes != 32 || sl.Reuse != 256 {
		t.Fatalf("family-B guard: lhs must stay promoted FRAG 32 B reuse 256, got %+v (ok=%v)", sl, ok)
	}
	sr, ok := rec.SiteDecisionOf(rhs)
	if !ok || sr.Promoted || sr.Class != ClassSMEM || sr.Bytes != 8388608 {
		t.Fatalf("family-B guard: rhs must stay evicted (SMEM 8388608 B window over budget), got %+v (ok=%v)", sr, ok)
	}
}

// TestS31SurvivingWarpPrune pins the §3.1 ruling (author, 2026-10-30,
// iq6 candidate-1 direction): the WARP digit is legal ONLY on a REDUCED
// dim. GPU-proven on the R3 re-choice kernels (row_sum/softmax, canary
// 516096/32256 = exactly 1/64 of the output space covered): a WARP digit
// on a non-reduced (surviving) dim lowers through the CTA-coupled-inside
// x lane-chunk composition whose cross product is 63/64 UNCOVERED — the
// iq6 §3.1 class (every (m,n) pair carries 16/1024 k values). fix5 §1
// R3's premise ("the layout already flows through the existing thread-map
// coordinate machinery — only the CHOICE of dim is new") is GPU-disproven;
// the prune restores the fix4 reduced-only WARP feasible area until the
// tile-lower cross-product lowering is designed, and the R3 ranking stays
// as the tie-break AMONG REDUCED candidates (multiple reduced dims).
//
// This is a LEGALITY pin, not a weakened outcome: the surviving-WARP
// class is pruned from the feasible area, so every pin asserting a
// non-reduced WARP winner (TestWarpDimRanking cases (a)/(b)/(d)/(e),
// TestWarpDimRestriction, TestR3SearchRechoiceConstK, qwo's
// TestSearchRechoiceCollapse) is re-derived to the pruned legality with
// this ruling in its comment.
func TestS31SurvivingWarpPrune(t *testing.T) {
	p := DefaultParams()

	// (a) Non-reduced candidates are EXCLUDED: on the const-K scale_copy
	// shape (reduced = {k}), the pre-prune R3 ranking picked c (L1 drops
	// the tiny reduced k, L4 innermost); under the prune the winner must
	// be a REDUCED dim — k — regardless of the extent ranking.
	fx, _, _ := scaleCopyFx(t, true) // r, c dynamic; k const 4 (reduced)
	lf := overlapOf(fx, p)
	lf.reduced = reducedIters(fx.g)
	// The prune consumes the ELIMINATED set (reducedAxesOf) — NOT lf.reduced,
	// whose reducer-consuming semantic also contains the surviving r/c.
	elim := reducedAxesOf(fx.g)
	if !elim[fx.iters[2]] || elim[fx.iters[0]] || elim[fx.iters[1]] {
		t.Fatalf("fixture setup: eliminated must be exactly {k}, got r=%v c=%v k=%v", elim[fx.iters[0]], elim[fx.iters[1]], elim[fx.iters[2]])
	}
	k := fx.iters[2]
	if got := warpDimOf(windowOf(lf, p), lf.CS, elim, true, p); got != k {
		t.Fatalf("prune: warpDimOf(const-K scale_copy) = iter %v, want the reduced iter k (surviving dims are pruned)", nodeID(got))
	}

	// (b) No reduced dim in the window -> NO WARP digit at all (nil
	// winner: digitListOf then emits no WARP section anywhere).
	fxN := newChain("s31nil", 64, 128)
	if got := warpDimOf(fxN.iters, nil, nil, true, p); got != nil {
		t.Fatalf("prune: warpDimOf(no reduced dims) = iter %v, want nil (no WARP section)", nodeID(got))
	}

	// (c) Search level: the C=WARP collapse vector is UNREPRESENTABLE on
	// the const-K scale_copy shape — C's decision must not be WARP. The
	// honest pruned winner (probe-derived, derivation below): the R4-
	// penalized CEREMONY vector R=CTA(16,T256) C=SEQ(T32) K=WARP(2,T32) —
	// the ONLY remaining positive-traffic class: the K=SEQ collapse
	// vector's covered [C,K] prices a 32x32x4 = 4096 B/thread FRAG
	// window > MaxFragPerThread 256 B (the qwo gate-3 mechanism), so it
	// scores traffic 0 and loses to the 16x-cut ceremony. The collapse
	// re-choice therefore needs fix6's work-shape model (FRAG windows
	// > 1x1 pricing) IN ADDITION to the §3.1 lowering — recorded in the
	// task SUMMARY, never chased out of scope here.
	want := []IterDecision{
		{SoC: SoCTA, LoC: 16, TileSize: 256},
		{SoC: SoCSequential, LoC: 1, TileSize: 32},
		{SoC: SoCWARP, LoC: 2, TileSize: 32},
	}
	rec := runPass(t, fx.g, p)
	for i, w := range want {
		if d := decisionOf(t, rec, fx.iters[i]); d != w {
			t.Fatalf("prune: iter %d must be %+v (the R4-penalized ceremony, derivation in the comment), got %+v", i, w, d)
		}
	}
	if d := decisionOf(t, rec, fx.iters[1]); d.SoC == SoCWARP {
		t.Fatalf("prune: the search chose C=WARP (%s/%d/T%d) — surviving-WARP vectors must be unrepresentable", d.SoC, d.LoC, d.TileSize)
	}

	// (d) The ranking still resolves AMONG REDUCED candidates: with two
	// reduced dims of distinct const extents, the L1 larger-extent one
	// wins (the R3 comparator kept as the reduced-internal tie-break).
	fx2 := newChain("s31two", 64, 128)
	r2, k2 := fx2.iters[0], fx2.iters[1]
	if got := warpDimOf(fx2.iters, nil, map[ir.Node]bool{r2: true, k2: true}, true, p); got != k2 {
		t.Fatalf("prune: warpDimOf({64,128}, both reduced) = iter %v, want iter %d (L1 extent 128)", nodeID(got), k2.BasePtr().ID())
	}
}
