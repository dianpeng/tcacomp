package tiling

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// verify_test.go pins the tiling-verify companion pass (priority 115):
// green after a clean tiling run (including the no-candidate paths), each
// planted corruption class flagged with a tiling-verify-prefixed error,
// the Requires enforcement, and the record's survival past the verify
// pass's own cache invalidation.

// fakePass is a minimal no-op pass used to satisfy Requires chains inside
// hand-built pipelines (the manager checks IDs only).
type fakePass struct{ id string }

func (f fakePass) ID() string                      { return f.id }
func (fakePass) Requires() []string                { return nil }
func (fakePass) Produces() opt.Stage               { return opt.StageConstructed }
func (fakePass) Preserved() []string               { return nil }
func (fakePass) Run(*opt.Context, *ir.Graph) error { return nil }

// runDecide runs the decide pass over g into a fresh cache and returns the
// cache plus the published record.
func runDecide(t *testing.T, g *ir.Graph, p Params) (*opt.Cache, *Record) {
	t.Helper()
	cache := opt.NewCache()
	if err := (Pass{Params: p}).Run(&opt.Context{Pass: "tiling", Cache: cache}, g); err != nil {
		t.Fatalf("tiling.Run: %v", err)
	}
	rec, ok := cache.Get(RecordName).(*Record)
	if !ok {
		t.Fatal("tiling.Run must publish *Record")
	}
	return cache, rec
}

// verifyWith runs tiling-verify over g against ctx and returns its error.
func verifyWith(g *ir.Graph, cache *opt.Cache, p Params) error {
	return (VerifyPass{Params: p}).Run(&opt.Context{Pass: "tiling-verify", Cache: cache}, g)
}

// verifyAfterTiling is the green-path stack: decide then verify.
func verifyAfterTiling(t *testing.T, g *ir.Graph, p Params) error {
	t.Helper()
	cache, _ := runDecide(t, g, p)
	return verifyWith(g, cache, p)
}

// emptyForest builds a well-formed graph with no iters at all.
func emptyForest() *ir.Graph {
	g := ir.NewGraph("noiters")
	out := tensorParam(g, "out", 1)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, nil, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return g
}

// TestVerifyGreenAfterTiling pins the green paths: the matmul fixture, a
// tree with no candidate sites, and a graph with no iters at all.
func TestVerifyGreenAfterTiling(t *testing.T) {
	t.Run("matmul", func(t *testing.T) {
		fx, _, _ := matmul()
		if err := verifyAfterTiling(t, fx.g, DefaultParams()); err != nil {
			t.Fatalf("tiling-verify must be green after tiling: %v", err)
		}
	})
	t.Run("no candidate sites", func(t *testing.T) {
		fx := newChain("plain", 64, 64)
		scaffoldStore(fx, fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
		if err := verifyAfterTiling(t, fx.g, DefaultParams()); err != nil {
			t.Fatalf("tiling-verify must be green on an empty-CS graph: %v", err)
		}
	})
	t.Run("empty forest", func(t *testing.T) {
		if err := verifyAfterTiling(t, emptyForest(), DefaultParams()); err != nil {
			t.Fatalf("tiling-verify must be green on an iter-less graph: %v", err)
		}
	})
}

// corrupt mutates a fresh matmul record and returns the verify error.
func corruptVerifyError(t *testing.T, mutate func(fx fixture, rec *Record)) error {
	t.Helper()
	fx, _, _ := matmul()
	p := DefaultParams()
	cache, rec := runDecide(t, fx.g, p)
	mutate(fx, rec)
	err := verifyWith(fx.g, cache, p)
	if err == nil {
		t.Fatal("the planted corruption must be flagged")
	}
	if !strings.HasPrefix(err.Error(), "tiling-verify:") {
		t.Fatalf("the error must be tiling-verify-prefixed, got %q", err.Error())
	}
	return err
}

// TestVerifyFlagsCorruption pins each corruption class under the fix4
// §2.7 re-check set: a deleted BIV entry, a CTA LoC outside the preset
// list, a warp LoC outside the preset list, a promoted site tied outside
// its LNT, and a record whose recomputed SMEM footprint exceeds the
// budget.
func TestVerifyFlagsCorruption(t *testing.T) {
	t.Run("deleted biv entry", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			biv := ir.DataDeps(fx.iters[0])[0]
			delete(rec.Iters, biv)
		})
		if !strings.Contains(err.Error(), "no record entry") {
			t.Fatalf("deleted-entry error must name the missing entry, got %q", err.Error())
		}
	})

	t.Run("cta loc outside presets", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				if d.SoC == SoCTA {
					d.LoC = DefaultParams().MaxCTA * 2 // over the cap AND off-list
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry a CTA entry")
		})
		if !strings.Contains(err.Error(), "preset") {
			t.Fatalf("off-list CTA LoC error must name the preset list, got %q", err.Error())
		}
	})

	t.Run("warp loc outside presets", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				if d.SoC != SoCTA {
					d.SoC = SoCWARP
					d.LoC = 3 // not a WarpLoCPresets entry
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry a non-CTA entry")
		})
		if !strings.Contains(err.Error(), "preset") {
			t.Fatalf("off-list warp LoC error must name the preset list, got %q", err.Error())
		}
	})

	t.Run("sequential loc not one", func(t *testing.T) {
		// SEQ dims carry LoC == 1 (one tile in the sequence dimension
		// = the whole dim is one tile). A sequential LoC != 1 is
		// off the implicit SEQ preset list = corruption.
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				if d.SoC == SoCSequential {
					d.LoC = 5 // not 1
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry at least one SEQ entry")
		})
		if !strings.Contains(err.Error(), "preset") {
			t.Fatalf("SEQ LoC != 1 error must name the preset list, got %q", err.Error())
		}
	})

	t.Run("promoted site tied outside its lnt", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			// Target an already-promoted entry deterministically: forcing
			// a DECLINED entry promoted would trip the budget gate first
			// (a declined entry still carries its computed tier class and
			// window bytes).
			for id, d := range rec.Sites {
				if !d.Promoted {
					continue
				}
				d.Tie = fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 1) // not an iter
				rec.Sites[id] = d
				return
			}
			t.Fatal("the matmul record must carry a promoted site entry")
		})
		if !strings.Contains(err.Error(), "tie") {
			t.Fatalf("foreign-tie error must name the tie, got %q", err.Error())
		}
	})

	t.Run("smem budget exceeded", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			// Inflate the covered dims' tiles and force both sites promoted
			// as SMEM: rhs covers {N, K}, so 4 * tile(N) * tile(K) alone
			// blows the 64 KB budget.
			for _, it := range fx.iters[1:] {
				biv := ir.DataDeps(it)[0]
				d := rec.Iters[biv]
				d.TileSize = 8192
				rec.Iters[biv] = d
			}
			for id, d := range rec.Sites {
				d.Promoted = true
				d.Class = ClassSMEM
				rec.Sites[id] = d
			}
		})
		if !strings.Contains(err.Error(), "MaxCache") {
			t.Fatalf("budget error must name MaxCache, got %q", err.Error())
		}
	})

	t.Run("tile one decision", func(t *testing.T) {
		// Gate 2 of the 2026-10-24 tile-disable ruling: a record decision at
		// TileSize 1 is corruption — tile size 1 is not a legal schedule
		// (the compiler never generates a non-tiling schedule), so verify
		// fails closed regardless of the record's other well-formedness.
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				d.TileSize = 1
				rec.Iters[biv] = d
				return
			}
			t.Fatal("the matmul record must carry an iter entry")
		})
		if !strings.Contains(err.Error(), "TileSize >= 2") {
			t.Fatalf("tile-1 error must name the TileSize >= 2 contract, got %q", err.Error())
		}
	})

	t.Run("zero cta dims", func(t *testing.T) {
		// CuMAP: every LNT must carry at least one CTA-assigned dim.
		// Demote the sole CTA dim to SEQ so the LNT has zero CTA dims.
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				if d.SoC == SoCTA {
					d.SoC = SoCSequential
					d.LoC = 1
					d.TileSize = 32
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry at least one CTA entry")
		})
		if !strings.Contains(err.Error(), "no CTA assignment") {
			t.Fatalf("zero-CTA error must name the missing CTA assignment, got %q", err.Error())
		}
	})

	t.Run("judge disagreement", func(t *testing.T) {
		// The re-derived judge must agree with the record's site
		// decision on Promoted / Class / Reuse / Bytes. Corrupting a
		// promoted site's reuse count trips the cross-check without
		// changing the iter decisions (so feasibility stays green).
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for id, d := range rec.Sites {
				if d.Promoted {
					d.Reuse = d.Reuse + 777 // wrong on purpose
					rec.Sites[id] = d
					return
				}
			}
			t.Fatal("the matmul record must carry at least one promoted site")
		})
		if !strings.Contains(err.Error(), "judge") {
			t.Fatalf("judge-disagreement error must name the judge cross-check, got %q", err.Error())
		}
	})
}

// TestVerifyNonCandidateCoverage pins the R4 fail-closed coverage: over a
// graph carrying a cross-LNT load, verify is GREEN when the record declares
// the non-candidate entries, and FAILS when such an entry is missing (a
// silently dropped load is corruption — never tolerated).
func TestVerifyNonCandidateCoverage(t *testing.T) {
	g, _, a1, _, b1 := twoChains()
	cross := g.NewMemLoad("ctor", tensorParam(g, "cross", 2),
		[]ir.Node{iv(g, a1), iv(g, b1)})

	p := DefaultParams()
	if err := verifyAfterTiling(t, g, p); err != nil {
		t.Fatalf("verify must be green when the non-candidate entries are declared: %v", err)
	}

	cache, rec := runDecide(t, g, p)
	sd, ok := rec.Sites[cross.BasePtr().ID()]
	if !ok || sd.Promoted || sd.Reason != ReasonCrossLNT {
		t.Fatalf("the decide record must declare the cross-LNT load, got (%v, %q)", ok, sd.Reason)
	}
	delete(rec.Sites, cross.BasePtr().ID())
	err := verifyWith(g, cache, p)
	if err == nil {
		t.Fatal("verify must fail when a declared non-candidate load has no record entry")
	}
	if !strings.Contains(err.Error(), "no record entry") {
		t.Fatalf("the missing non-candidate error must name the missing entry, got %q", err.Error())
	}
}

// TestVerifyForLoopStep pins the ForLoop dimvar step-positive invariant
// (enforced in tiling-verify now; the IR-verifier version lands with T4):
// a ForLoop whose dimvar step is a zero const is rejected; a positive
// const step passes.
func TestVerifyForLoopStep(t *testing.T) {
	mkFor := func(g *ir.Graph, step float64) ir.Node {
		f32 := ir.InternScalar(ir.TU32)
		d := g.NewInputParam("ctor", ir.RoleDim, "fd", nil)
		dv := g.NewDimVar("ctor", d, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, step))
		return g.NewForLoop("ctor", dv, g.NewConst("ctor", f32, 64),
			g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	}

	t.Run("zero step rejected", func(t *testing.T) {
		fx, _, _ := matmul()
		mkFor(fx.g, 0)
		cache, _ := runDecide(t, fx.g, DefaultParams())
		err := verifyWith(fx.g, cache, DefaultParams())
		if err == nil {
			t.Fatal("a ForLoop with a zero dimvar step must be rejected (an infinite loop)")
		}
		if !strings.Contains(err.Error(), "step") {
			t.Fatalf("the error must name the step, got %q", err.Error())
		}
	})

	t.Run("positive const step green", func(t *testing.T) {
		fx, _, _ := matmul()
		mkFor(fx.g, 4)
		if err := verifyAfterTiling(t, fx.g, DefaultParams()); err != nil {
			t.Fatalf("a positive-const step must pass: %v", err)
		}
	})
}

// TestVerifyKernelWideFragGate pins the F6 kernel-wide gate at the verify
// side: a genuine FRAG-promoting record verifies green under the mirrored
// params, and the recomputed kernel-wide FRAG sum is gated against THIS
// pass's MaxFragPerThread — a 31-byte budget flags it.
//
// §2.4 sweep (2026-10-26 amendment) — the pin STAYS at 32 B, and here is
// the full hand derivation why (the plan's round-2 draft expected a flip
// to a 4 B K=WARP winner; that draft forgot the §2.6 eviction collateral,
// see the SUMMARY deviation note):
//
//   - The K=WARP vector (M=CTA/…/8, N=SEQ/1/8, K=WARP/K_d/8) judges lhs
//     as a 4 B lane-chunk FRAG, reuse 8, holders K_d·32 — traffic
//     4×7×(K_d·32) = 896·K_d (28672 at K_d=32), and rhs as an
//     8×(K_d·8)×4 = 256·K_d B SMEM window (≥ 512 B > MaxCache 100).
//     BUT eviction is part of the score: density = reuse − 1 ties at 7
//     (lhs per-thread 8, rhs collective T_M = 8), and the UNCHANGED
//     §2.6 tie-break (smaller bytes first) drops the 4 B lhs BEFORE the
//     8192 B rhs — the vector evacuates BOTH sites and scores 0.
//   - The all-SEQ-covered vector (M=CTA/64/8, N=SEQ/1/8, K=SEQ/1/8,
//     blockDim 32) judges lhs as a 32 B FRAG at reuse 8 (the N walk),
//     traffic 32×7×32 = 7168; rhs is a 256 B FRAG (≤ MaxFragPerThread —
//     ReasonFragBytes cannot fire) that declines ReasonLowReuse on the M
//     CTA dim (per-thread ceil(8/32) = 1 trip). No SMEM exists, nothing
//     evicts, score 7168 > 0 — TRAFFIC decides; the partitionCount
//     tiebreak never comes into play.
//
// So the searched winner keeps exactly one 32 B promoted FRAG site, and
// the 31-byte tight budget still fires (32 > 31). The amendment DID move
// this fixture's arithmetic (rhs now declines for low reuse instead of
// pricing out; K=WARP vectors now die in eviction) — but not the sum.
func TestVerifyKernelWideFragGate(t *testing.T) {
	fx, _, _ := matmul()
	decideP := DefaultParams()
	decideP.TileSizes = []int{8} // single tile preset: every dim tiles at 8
	decideP.MaxCache = 100       // SMEM windows never fit: the FRAG tier wins

	cache, rec := runDecide(t, fx.g, decideP)
	var promotedFrag int
	for _, sd := range rec.Sites {
		if sd.Promoted && sd.Class == ClassFRAG {
			promotedFrag += sd.Bytes
		}
	}
	if promotedFrag != 32 {
		t.Fatalf("the fixture must promote exactly one 32 B FRAG site (the K=WARP challenger scores 0 after §2.6 eviction — see the derivation above), got %d B", promotedFrag)
	}
	if err := verifyWith(fx.g, cache, decideP); err != nil {
		t.Fatalf("verify must be green under the mirrored params: %v", err)
	}
	tight := decideP
	tight.MaxFragPerThread = 31
	err := verifyWith(fx.g, cache, tight)
	if err == nil {
		t.Fatal("the kernel-wide FRAG gate must fire under a 31-byte budget")
	}
	if !strings.Contains(err.Error(), "MaxFragPerThread") {
		t.Fatalf("the FRAG budget error must name MaxFragPerThread, got %q", err.Error())
	}
}

// TestVerifyMultiCTAGreen pins the v0-policy deletion: a record carrying
// TWO CTA dims (well-formed preset LoCs, products within bounds) now
// VERIFIES GREEN — multi-dim parallel schedules are legal records under
// the fix4 model (tile-lower's templates fail closed on them pending
// fix4 T4, but the record side is sound).
func TestVerifyMultiCTAGreen(t *testing.T) {
	g := ir.NewGraph("multi-verify")
	u32 := ir.InternScalar(ir.TU32)
	var dims, iters []ir.Node
	for i := 0; i < 2; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	var parent ir.Node
	for i := 0; i < 2; i++ {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, 1024))
		parent = g.NewIter("ctor", biv, parent)
		iters = append(iters, parent)
	}
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, nil, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: map[uint64]SiteDecision{}}
	for i, it := range iters {
		biv := ir.DataDeps(it)[0]
		rec.Iters[biv] = IterDecision{SoC: SoCTA, LoC: 2, TileSize: 32}
		_ = i
	}
	cache := opt.NewCache()
	cache.Put(rec)
	if err := verifyWith(g, cache, DefaultParams()); err != nil {
		t.Fatalf("a two-CTA-dim record must verify green under the fix4 re-check set, got %v", err)
	}
}

// TestVerifyCTAProductOverMaxCTA pins the MALoCC CTA-product check: a
// record carrying two CTA dims whose LoCs are each in the preset list
// but whose product exceeds MaxCTA must fail verification. The two dims
// use in-list LoCs (64, 16) so the preset-list check passes and the
// product check is what fires (64 × 16 = 1024 > 512 = MaxCTA).
func TestVerifyCTAProductOverMaxCTA(t *testing.T) {
	g := ir.NewGraph("cta-prod-over")
	u32 := ir.InternScalar(ir.TU32)
	var dims, iters []ir.Node
	for i := 0; i < 2; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	var parent ir.Node
	for i := 0; i < 2; i++ {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, 1024))
		parent = g.NewIter("ctor", biv, parent)
		iters = append(iters, parent)
	}
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, nil, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	p := DefaultParams()
	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: map[uint64]SiteDecision{}}
	// Both dims CTA, LoCs both in CtaLoCPresets, product = 64 × 16 = 1024 > 512.
	rec.Iters[ir.DataDeps(iters[0])[0]] = IterDecision{SoC: SoCTA, LoC: 64, TileSize: 32}
	rec.Iters[ir.DataDeps(iters[1])[0]] = IterDecision{SoC: SoCTA, LoC: 16, TileSize: 32}
	cache := opt.NewCache()
	cache.Put(rec)
	err := verifyWith(g, cache, p)
	if err == nil {
		t.Fatal("a two-CTA-dim record with product > MaxCTA must fail verification")
	}
	if !strings.Contains(err.Error(), "CTA factorization product") {
		t.Fatalf("the error must name the CTA factorization product, got %q", err.Error())
	}
}

// TestVerifyWarpProductOverMaxWarps pins the MALoCC warp-product check:
// a record carrying two WARP dims whose Ks are each in the preset list
// but whose product exceeds MaxWarpsPerCTA must fail verification. Both
// Ks are in WarpLoCPresets (16, 4) so the preset check passes and the
// product check is what fires (16 × 4 = 64 > 32 = MaxWarpsPerCTA).
func TestVerifyWarpProductOverMaxWarps(t *testing.T) {
	g := ir.NewGraph("warp-prod-over")
	u32 := ir.InternScalar(ir.TU32)
	var dims, iters []ir.Node
	for i := 0; i < 3; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	var parent ir.Node
	for i := 0; i < 3; i++ {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, 1024))
		parent = g.NewIter("ctor", biv, parent)
		iters = append(iters, parent)
	}
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, nil, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	p := DefaultParams()
	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: map[uint64]SiteDecision{}}
	// One CTA dim (CuMAP: at least one) + two WARP dims.
	// Warp product = 16 × 4 = 64 > 32 = MaxWarpsPerCTA.
	rec.Iters[ir.DataDeps(iters[0])[0]] = IterDecision{SoC: SoCTA, LoC: 8, TileSize: 32}
	rec.Iters[ir.DataDeps(iters[1])[0]] = IterDecision{SoC: SoCWARP, LoC: 16, TileSize: 32}
	rec.Iters[ir.DataDeps(iters[2])[0]] = IterDecision{SoC: SoCWARP, LoC: 4, TileSize: 32}
	cache := opt.NewCache()
	cache.Put(rec)
	err := verifyWith(g, cache, p)
	if err == nil {
		t.Fatal("a two-WARP-dim record with product > MaxWarpsPerCTA must fail verification")
	}
	if !strings.Contains(err.Error(), "warp factorization product") {
		t.Fatalf("the error must name the warp factorization product, got %q", err.Error())
	}
}

// TestVerifyRequiresTiling pins the Requires enforcement: a pipeline of
// just [tiling-verify] is rejected by the manager's precondition check.
func TestVerifyRequiresTiling(t *testing.T) {
	fx, _, _ := matmul()
	pm := opt.NewManager([]opt.Pass{VerifyPass{Params: DefaultParams()}})
	err := pm.Run(fx.g)
	if err == nil {
		t.Fatal("a [tiling-verify]-only pipeline must be rejected")
	}
	if !strings.Contains(err.Error(), "requires") {
		t.Fatalf("the manager's requires message must surface, got %q", err.Error())
	}
}

// TestRecordSurvivesVerify pins the Preserved whitelist: after a pipeline
// of [simplify-lower (fake), tiling, tiling-verify], the record is still
// retrievable from the manager's cache — the verify pass's invalidation
// does not wipe it.
func TestRecordSurvivesVerify(t *testing.T) {
	fx, _, _ := matmul()
	pm := opt.NewManager([]opt.Pass{
		fakePass{id: "simplify-lower"},
		Pass{Params: DefaultParams()},
		VerifyPass{Params: DefaultParams()},
	})
	if err := pm.Run(fx.g); err != nil {
		t.Fatalf("pipeline run: %v", err)
	}
	if got := pm.Cache().Get(RecordName); got == nil {
		t.Fatal("the tiling record must survive the tiling-verify pass's invalidation")
	}
}
