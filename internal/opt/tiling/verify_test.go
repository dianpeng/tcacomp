package tiling

import (
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

// TestVerifyFlagsCorruption pins each corruption class: a deleted BIV
// entry, an inflated CTA LoC, a v0-policy violation (a second CTA dim), a
// warp LoC that is not a portion multiple, a promoted site tied outside
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

	t.Run("inflated cta loc", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			for biv, d := range rec.Iters {
				if d.SoC == SoCTA {
					d.LoC = DefaultParams().MaxCTA * 2
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry a CTA entry")
		})
		if !strings.Contains(err.Error(), "MaxCTA") {
			t.Fatalf("inflated-LoC error must name MaxCTA, got %q", err.Error())
		}
	})

	t.Run("v0 policy second cta dim", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			// Flip one sequential entry into a well-formed second CTA dim
			// (LoC == MaxCTA), so only the policy count fires.
			for biv, d := range rec.Iters {
				if d.SoC == SoCSequential {
					d.SoC = SoCTA
					d.LoC = DefaultParams().MaxCTA
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry a sequential entry")
		})
		if !strings.Contains(err.Error(), "v0 policy") {
			t.Fatalf("second-CTA error must name the v0 policy, got %q", err.Error())
		}
	})

	t.Run("warp loc not portion multiple", func(t *testing.T) {
		err := corruptVerifyError(t, func(fx fixture, rec *Record) {
			quantum := (DefaultParams().MaxThreadsPerCTA / 32) / PortionWarp
			for biv, d := range rec.Iters {
				if d.SoC != SoCTA {
					d.SoC = SoCWARP
					d.LoC = quantum + 1 // not a quantum multiple
					rec.Iters[biv] = d
					return
				}
			}
			t.Fatal("the matmul record must carry a non-CTA entry")
		})
		if !strings.Contains(err.Error(), "portion") {
			t.Fatalf("non-portion warp LoC error must name the portion quantum, got %q", err.Error())
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
// side: a genuine FRAG-promoting record (32 B at tile 8) verifies green
// under the mirrored params, and the recomputed kernel-wide FRAG sum is
// gated against THIS pass's MaxFragPerThread — a 31-byte budget flags it.
func TestVerifyKernelWideFragGate(t *testing.T) {
	fx, _, _ := matmul()
	decideP := DefaultParams()
	decideP.TileSizes = []int{8} // FRAG-friendly: lhs = 4·8 = 32 B, reuse 8
	decideP.MaxCache = 100       // SMEM windows never fit: the FRAG tier wins

	cache, rec := runDecide(t, fx.g, decideP)
	var promotedFrag int
	for _, sd := range rec.Sites {
		if sd.Promoted && sd.Class == ClassFRAG {
			promotedFrag += sd.Bytes
		}
	}
	if promotedFrag != 32 {
		t.Fatalf("the fixture must promote exactly one 32 B FRAG site, got %d B", promotedFrag)
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

// TestVerifyCoupledTileGate pins HI-03 gate 2: a record whose CTA dim
// carries a TileSize over the coupled inside-walk thread count (resolved
// from Params exactly as the rewrite's coupledThreads does — never the
// record's warp LoC) fails the per-LNT feasibility sweep with a named
// error; a tile exactly at the bound verifies green.
func TestVerifyCoupledTileGate(t *testing.T) {
	fx := newChain32("ctagate", 4096)
	bodyStore(fx)
	mk := func(tile int) *Record {
		return recordOf(t, fx, map[int]IterDecision{
			0: {SoC: SoCTA, LoC: 100, TileSize: tile},
		}, nil)
	}
	cache := opt.NewCache()
	cache.Put(mk(512))
	err := verifyWith(fx.g, cache, DefaultParams())
	if err == nil {
		t.Fatal("a CTA TileSize over the coupled thread count must fail the feasibility sweep")
	}
	if !strings.Contains(err.Error(), "coupled") {
		t.Fatalf("the gate-2 error must name the coupled walk, got %q", err.Error())
	}
	if !strings.Contains(err.Error(), "512") || !strings.Contains(err.Error(), "256") {
		t.Fatalf("the gate-2 error must name the tile and the coupled count, got %q", err.Error())
	}
	cacheOK := opt.NewCache()
	cacheOK.Put(mk(256))
	if err := verifyWith(fx.g, cacheOK, DefaultParams()); err != nil {
		t.Fatalf("a CTA TileSize exactly at the coupled count must verify green, got %v", err)
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
