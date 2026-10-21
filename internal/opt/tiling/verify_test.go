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

func (f fakePass) ID() string                   { return f.id }
func (fakePass) Requires() []string             { return nil }
func (fakePass) Produces() opt.Stage            { return opt.StageConstructed }
func (fakePass) Preserved() []string            { return nil }
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
			for id, d := range rec.Sites {
				d.Promoted = true
				d.Tie = fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 1) // not an iter
				rec.Sites[id] = d
				return
			}
			t.Fatal("the matmul record must carry a site entry")
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
