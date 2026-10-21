package tiling

import (
	"fmt"
	"testing"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
)

// overlap_test.go pins the MemLoad overlap analysis of
// doc/optana/loop_tiling.md ("MemLoad Overlap Analysis" plus the "Cache Tie
// Semantics" rules as revised by the 2026-10-20 rulings): findINS, SET-level
// bucketing at the innermost REFERENCED level, the admission filters
// (set-size >= 2, tensor rank >= CacheDims, the MaxCS cap), and TIE-level
// placement (the operational tie rule, whole-dimension coverage, and the
// CacheDims clamp). The doc's matmul worked example is the pinned fixture:
// SET {lhs, rhs} @ iterK; lhs tie iterN covering [·][K]; rhs above the
// outermost loop covering [N][K].

// fixture is one loopforest-shaped test graph: dim params, an out tensor,
// and a chain of normalized iters (root first) bounded by const maxes.
type fixture struct {
	g     *ir.Graph
	dims  []ir.Node
	out   ir.Node
	iters []ir.Node // root first, innermost last
}

// newChain builds an n-deep iter chain over dim params d0..dn-1, level i
// bounded by the const extent maxes[i].
func newChain(name string, maxes ...float64) fixture {
	g := ir.NewGraph(name)
	f32 := ir.InternScalar(ir.TF32)
	var fx fixture
	fx.g = g
	for i := range maxes {
		fx.dims = append(fx.dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	fx.out = g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	var parent ir.Node
	for i := range maxes {
		dv := g.NewDimVar("ctor", fx.dims[i], g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", f32, maxes[i]))
		parent = g.NewIter("ctor", biv, parent)
		fx.iters = append(fx.iters, parent)
	}
	return fx
}

// tensorParam allocates a rank-dimensional f32 tensor parameter.
func tensorParam(g *ir.Graph, name string, rank int) ir.Node {
	dims := make([]ir.DimRef, rank)
	for i := range dims {
		dims[i] = ir.DimRef{Name: fmt.Sprintf("%s_d%d", name, i)}
	}
	return g.NewInputParam("ctor", ir.RoleTensor, name, ir.InternTensor(ir.TF32, dims))
}

// iv builds the InductionVariable of iter.
func iv(g *ir.Graph, iter ir.Node) ir.Node {
	return g.NewInductionVariable("ctor", iter)
}

// scaffoldStore wires the loopforest-test-shaped store+return observation
// so the graph is well-formed construction, not floating debris.
func scaffoldStore(fx fixture, val ir.Node) {
	env := ir.NewEnv()
	st := fx.g.NewMemStoreChained("ctor", env, fx.out, nil, val)
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, st)
}

// matmul builds the doc's worked example: M->N->K chain, lhs[m][k],
// rhs[n][k]. Returns the fixture plus the two loads.
func matmul() (fx fixture, lhs, rhs ir.Node) {
	fx = newChain("matmul", 1024, 2048, 512) // M, N, K
	lhs = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[2])})
	rhs = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "rhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	scaffoldStore(fx, fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	return fx, lhs, rhs
}

// overlapOf runs the analysis stack over fx and returns the single tree's
// facts (the chain fixtures have exactly one root).
func overlapOf(fx fixture, p Params) lntFacts {
	se := sideeffect.Analyze(fx.g)
	f := loopforest.Analyze(fx.g)
	all := analyzeOverlap(fx.g, f, se, p)
	if len(all) != 1 {
		panic(fmt.Sprintf("chain fixture must yield exactly one LNT, got %d", len(all)))
	}
	return all[0]
}

// siteOf finds the CS entry of load.
func siteOf(t *testing.T, facts lntFacts, load ir.Node) SiteFacts {
	t.Helper()
	for _, s := range facts.CS {
		if s.Load == load {
			return s
		}
	}
	t.Fatalf("load node %d is not in the candidate set", load.BasePtr().ID())
	return SiteFacts{}
}

// assertIterSet checks that got holds exactly the want iters (node identity,
// order-insensitive) and nothing else.
func assertIterSet(t *testing.T, label string, got, want []ir.Node) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d iters, want %d", label, len(got), len(want))
	}
	for _, w := range want {
		found := false
		for _, gn := range got {
			if gn == w {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("%s: iter %d missing from %v-by-id", label, w.BasePtr().ID(), idsOf(got))
		}
	}
}

func idsOf(nodes []ir.Node) []uint64 {
	out := make([]uint64, 0, len(nodes))
	for _, n := range nodes {
		out = append(out, n.BasePtr().ID())
	}
	return out
}

// TestMatmulOverlap pins the doc's worked example end to end at the overlap
// level: findINS over both loads, ONE co-traversed set at iterK, the lhs tie
// at iterN covering [·][K], and the rhs tie above the outermost loop
// covering [N][K].
func TestMatmulOverlap(t *testing.T) {
	fx, lhs, rhs := matmul()
	se := sideeffect.Analyze(fx.g)

	insL := findINS(lhs, se)
	assertIterSet(t, "findINS(lhs)", insL, []ir.Node{fx.iters[0], fx.iters[2]})
	insR := findINS(rhs, se)
	assertIterSet(t, "findINS(rhs)", insR, []ir.Node{fx.iters[1], fx.iters[2]})

	facts := overlapOf(fx, DefaultParams())
	if len(facts.CS) != 2 {
		t.Fatalf("the matmul set {lhs, rhs} @ iterK must admit both loads, got %d CS entries", len(facts.CS))
	}

	sl := siteOf(t, facts, lhs)
	if sl.SetLevel != fx.iters[2] {
		t.Fatalf("lhs SET level must be iterK (innermost referenced), got node %d", sl.SetLevel.BasePtr().ID())
	}
	if sl.Tie != fx.iters[1] {
		t.Fatalf("lhs TIE must be iterN, got %v", tieLabel(sl.Tie))
	}
	assertIterSet(t, "covered(lhs)", sl.Covered, []ir.Node{fx.iters[2]})

	sr := siteOf(t, facts, rhs)
	if sr.SetLevel != fx.iters[2] {
		t.Fatalf("rhs SET level must be iterK, got node %d", sr.SetLevel.BasePtr().ID())
	}
	if sr.Tie != nil {
		t.Fatalf("rhs TIE must be above the outermost loop (nil), got node %d", sr.Tie.BasePtr().ID())
	}
	assertIterSet(t, "covered(rhs)", sr.Covered, []ir.Node{fx.iters[1], fx.iters[2]})
}

func tieLabel(tie ir.Node) string {
	if tie == nil {
		return "nil"
	}
	return fmt.Sprintf("node %d", tie.BasePtr().ID())
}

// TestCompoundAxisINS pins the Input-list chase through composed scalar
// expressions: an axis (ivM*K + ivK) collects BOTH iters into INS.
func TestCompoundAxisINS(t *testing.T) {
	fx := newChain("compound", 64, 64)
	f32 := ir.InternScalar(ir.TF32)
	k := fx.g.NewConst("ctor", f32, 64)
	ivA, ivB := iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[1])
	axis := fx.g.NewBinOp("ctor", ir.BinOpAdd,
		fx.g.NewBinOp("ctor", ir.BinOpMul, ivA, k), ivB)
	load := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "t", 2), []ir.Node{axis})
	scaffoldStore(fx, fx.g.NewConst("ctor", f32, 0))

	ins := findINS(load, sideeffect.Analyze(fx.g))
	assertIterSet(t, "findINS(compound axis)", ins, fx.iters)
}

// TestAdmissionFilters pins the three doc-ordered filters: a singleton set
// (no co-traversed partner) is skipped whole, rank-1 tensor loads are
// dropped from an otherwise-admitted set, and MaxCS stops admission beyond
// the cap.
func TestAdmissionFilters(t *testing.T) {
	t.Run("singleton set skipped", func(t *testing.T) {
		fx, lhs, rhs := matmul()
		// w references only iterN: its own bucket at iterN is a singleton.
		w := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "w", 2),
			[]ir.Node{iv(fx.g, fx.iters[1])})
		_ = w
		facts := overlapOf(fx, DefaultParams())
		if len(facts.CS) != 2 {
			t.Fatalf("the singleton {w} @ iterN must be skipped, got %d CS entries", len(facts.CS))
		}
		siteOf(t, facts, lhs)
		siteOf(t, facts, rhs)
	})

	t.Run("rank-1 loads dropped", func(t *testing.T) {
		fx, lhs, rhs := matmul()
		// y1/y2 co-traverse iterN but are rank 1 < CacheDims 2.
		y1 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "y1", 1), []ir.Node{iv(fx.g, fx.iters[1])})
		y2 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "y2", 1), []ir.Node{iv(fx.g, fx.iters[1])})
		_ = y1
		_ = y2
		facts := overlapOf(fx, DefaultParams())
		if len(facts.CS) != 2 {
			t.Fatalf("both rank-1 loads must be dropped from the iterN set, got %d CS entries", len(facts.CS))
		}
		siteOf(t, facts, lhs)
		siteOf(t, facts, rhs)
	})

	t.Run("maxcs caps admission", func(t *testing.T) {
		fx, lhs, _ := matmul()
		p := DefaultParams()
		p.MaxCS = 1
		facts := overlapOf(fx, p)
		if len(facts.CS) != 1 {
			t.Fatalf("MaxCS 1 must stop admission at one site, got %d CS entries", len(facts.CS))
		}
		if facts.CS[0].Load != lhs {
			t.Fatalf("the first-allocated load (lhs) must be the admitted one, got node %d",
				facts.CS[0].Load.BasePtr().ID())
		}
	})
}

// TestTieClampRank3 pins the CacheDims clamp: a rank-3 load whose INS spans
// the whole LNT gets its tie moved inward until covered stays within
// CacheDims dims.
func TestTieClampRank3(t *testing.T) {
	fx := newChain("clamp", 128, 128, 128)
	lhs3 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs3", 3),
		[]ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	rhs3 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "rhs3", 2),
		[]ir.Node{iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	scaffoldStore(fx, fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))

	facts := overlapOf(fx, DefaultParams())
	if len(facts.CS) != 2 {
		t.Fatalf("the {lhs3, rhs3} @ iterC set must admit both loads, got %d CS entries", len(facts.CS))
	}
	sl := siteOf(t, facts, lhs3)
	// Full-span INS {A,B,C} ties above the outermost, then the clamp moves
	// the tie inward one level at a time: nil -> A leaves covered {B,C}.
	if sl.Tie != fx.iters[0] {
		t.Fatalf("the clamped lhs3 tie must be iterA (the outermost level), got %s", tieLabel(sl.Tie))
	}
	assertIterSet(t, "covered(lhs3)", sl.Covered, fx.iters[1:])
	sr := siteOf(t, facts, rhs3)
	if sr.Tie != nil {
		t.Fatalf("rhs3 (INS {B,C}, missing A) ties above the outermost loop, got %s", tieLabel(sr.Tie))
	}
	assertIterSet(t, "covered(rhs3)", sr.Covered, fx.iters[1:])
}

// TestConstValueOf pins the new ir read face: consts report (value, true),
// everything else reports (0, false).
func TestConstValueOf(t *testing.T) {
	f32 := ir.InternScalar(ir.TF32)
	g := ir.NewGraph("consts")
	c := g.NewConst("ctor", f32, 42.5)
	if v, ok := ir.ConstValueOf(c); !ok || v != 42.5 {
		t.Fatalf("ConstValueOf(const) = (%v, %v), want (42.5, true)", v, ok)
	}
	b := g.NewBinOp("ctor", ir.BinOpAdd, c, c)
	if v, ok := ir.ConstValueOf(b); ok || v != 0 {
		t.Fatalf("ConstValueOf(binop) = (%v, %v), want (0, false)", v, ok)
	}
	if v, ok := ir.ConstValueOf(nil); ok || v != 0 {
		t.Fatalf("ConstValueOf(nil) = (%v, %v), want (0, false)", v, ok)
	}
}

// TestParamsValidate pins DefaultParams' cleanliness and every Validate
// error class — the portion divisibility cases mirror the runtime
// assert-and-trap pre-conditions.
func TestParamsValidate(t *testing.T) {
	if err := DefaultParams().Validate(); err != nil {
		t.Fatalf("DefaultParams must validate clean, got %v", err)
	}
	cases := []struct {
		name string
		mut  func(*Params)
	}{
		{"threads not warp multiple", func(p *Params) { p.MaxThreadsPerCTA = 1023 }},
		{"threads below one warp", func(p *Params) { p.MaxThreadsPerCTA = 16 }},
		{"threads below portion minimum", func(p *Params) { p.MaxThreadsPerCTA = 64 }},
		{"cta not portion multiple", func(p *Params) { p.MaxCTA = 99 }},
		{"cta zero", func(p *Params) { p.MaxCTA = 0 }},
		{"warp portion negative", func(p *Params) { p.WarpPortion = -1 }},
		{"warp portion above cap", func(p *Params) { p.WarpPortion = PortionWarp + 1 }},
		{"zero tile size", func(p *Params) { p.TileSizes = []int{32, 0} }},
		{"empty tile list", func(p *Params) { p.TileSizes = nil }},
		{"windowk zero", func(p *Params) { p.WindowK = 0 }},
		{"cachedims zero", func(p *Params) { p.CacheDims = 0 }},
		{"maxcache negative", func(p *Params) { p.MaxCache = -1 }},
		{"maxcs zero", func(p *Params) { p.MaxCS = 0 }},
		{"nominalextent zero", func(p *Params) { p.NominalExtent = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := DefaultParams()
			tc.mut(&p)
			if err := p.Validate(); err == nil {
				t.Fatalf("%s must fail Validate", tc.name)
			}
		})
	}
}
