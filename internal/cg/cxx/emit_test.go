package cxx

import (
	"fmt"
	"strings"
	"testing"

	_ "tcacomp/internal/opt/all"

	"tcacomp/internal/cg"
	"tcacomp/internal/golden"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/finallower"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling"
	"tcacomp/internal/opt/typeinfer"
	"tcacomp/internal/pipeline"
)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

// matmulStoreGraph builds the worked store-chain matmul (the schedule
// pipeline test's shape): u32 extents 1024/1024/1024 over the M→N→K iter
// chain, InductionVariable-wired lhs[m][k] and rhs[n][k] loads, and a
// chained store of lhs*rhs over [m,n,k]. The searched tiling promotes lhs
// into an SMEM window, so the emitted kernel exercises the mappings,
// CTA/thr/seq iterators, the dynamic-smem pool, and the cooperative fill.
func matmulStoreGraph(name string) *ir.Graph {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "d0", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d1", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d2", nil),
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d2"}}))
	rhs := g.NewInputParam("ctor", ir.RoleTensor, "rhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d1"}, {Name: "d2"}}))
	var parent ir.Node
	iters := make([]ir.Node, 3)
	for i, extent := range []float64{1024, 1024, 1024} {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extent))
		parent = g.NewIter("ctor", biv, parent)
		iters[i] = parent
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }
	l := g.NewMemLoad("ctor", lhs, []ir.Node{iv(0), iv(2)}) // lhs[m][k]
	r := g.NewMemLoad("ctor", rhs, []ir.Node{iv(1), iv(2)}) // rhs[n][k]
	prod := g.NewBinOp("ctor", ir.BinOpMul, l, r)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1), iv(2)}, prod)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return g
}

// runPipeline runs the full default pipeline (the real searched tiling
// under DefaultParams) and fails the test on any error.
func runPipeline(t *testing.T, g *ir.Graph) {
	t.Helper()
	// fix3 D-o: the REAL composition is the pipeline orchestrator's —
	// HIR + LOWER + the TAIL (const dedup, memopt, AST-roots dce). The
	// registry pipeline keeps its catalog role; the goldens pin the path
	// users actually compile through (the facade wires pipeline.Run).
	if err := pipeline.Run(g); err != nil {
		t.Fatalf("default pipeline run = %v", err)
	}
}

// chainFx bundles a small hand-built iter chain for the crafted-record
// fixtures (the finallower test idiom: pkg-built graphs cannot traverse the
// pipeline — the schedule gate refuses materialized graphs).
type chainFx struct {
	g     *ir.Graph
	u32   ir.Type
	dims  []ir.Node
	iters []ir.Node
}

func newChain(name string, extents ...float64) *chainFx {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	f := &chainFx{g: g, u32: u32}
	for i := range extents {
		f.dims = append(f.dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	var parent ir.Node
	for i := range extents {
		dv := g.NewDimVar("ctor", f.dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extents[i]))
		parent = g.NewIter("ctor", biv, parent)
		f.iters = append(f.iters, parent)
	}
	return f
}

func (f *chainFx) tensor(name string, dims ...int) ir.Node {
	refs := make([]ir.DimRef, len(dims))
	for i, d := range dims {
		refs[i] = ir.DimRef{Name: fmt.Sprintf("d%d", d)}
	}
	return f.g.NewInputParam("ctor", ir.RoleTensor, name, ir.InternTensor(ir.TF32, refs))
}

func (f *chainFx) iv(i int) ir.Node { return f.g.NewInductionVariable("ctor", f.iters[i]) }

func (f *chainFx) bodyStore(out ir.Node, val ir.Node) {
	env := ir.NewEnv()
	axes := make([]ir.Node, len(f.iters))
	for i := range f.iters {
		axes[i] = f.iv(i)
	}
	st := f.g.NewMemStoreChained("ctor", env, out, axes, val)
	f.g.NewReturnChained("ctor", env, []ir.Node{out}, st)
}

// runTileLowerSchedule runs the real tile-lower + schedule passes with a
// crafted verified record (one decision per chain iter, by index), the
// finallower fixture idiom.
func runTileLowerSchedule(t *testing.T, f *chainFx, decisions []tiling.IterDecision) *opt.Cache {
	t.Helper()
	rec := &tiling.Record{Iters: map[ir.Node]tiling.IterDecision{}, Sites: map[uint64]tiling.SiteDecision{}, Verified: true}
	for i, it := range f.iters {
		biv := ir.DataDeps(it)[0]
		if biv == nil {
			t.Fatalf("iter %d carries no BIV", i)
		}
		rec.Iters[biv] = decisions[i]
	}
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (tiling.LowerPass{Params: tiling.DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("tile-lower: %v", err)
	}
	if err := (schedule.Scheduler{}).Run(nil, f.g); err != nil {
		t.Fatalf("schedule: %v", err)
	}
	if cache.Get(tiling.ProvenanceName) == nil {
		t.Fatal("fixture setup: the provenance must be published and preserved past schedule")
	}
	return cache
}

// reduceWarpGraph builds the tiled reduce chain (the final-lower test's
// proven shape): a 2-dim chain with m CTA T=32 / k WARP T=64, lhs[m][k]
// load → map (as=[m,k]) → reduce(init=0.0f, +) → out[m] store. Its golden
// exercises the WarpReducer, the D7 slot reads, the hoisted combiner
// lambda, and the epilogue re-iteration.
func reduceWarpGraph(t *testing.T) *ir.Graph {
	t.Helper()
	f := newChain("redwarp", 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0)}, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	return f.g
}

// reduce3Graph builds the rank-3 last-axis reduce (the HI-01 proof shape):
// u32 extents 64/32/512, lhs[d0,d1,d2] load → map (as=[d0,d1,d2]) →
// reduce(0.0f, +) → out[d0,d1] store, decisions {CTA,100,32}/{SEQ,1,64}/
// {WARP,8,64} — TWO between dims (the >=2-between-dims R-A case).
func reduce3Graph(t *testing.T) *ir.Graph {
	t.Helper()
	f := newChain("red3", 64, 32, 512)
	lhs := f.tensor("lhs", 0, 1, 2)
	out := f.tensor("out", 0, 1)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1), f.iv(2)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0), f.iv(1)}, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 64},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	return f.g
}

// reduceT1Graph builds the minimal-legal-tile endpoint (the HI-02 proof
// shape, retargeted by the 2026-10-24 tile-disable ruling; the identifier
// is historical): the redwarp graph at {CTA,100,2}/{WARP,8,2} — the floor
// tile 2 everywhere, the compiler's flat endpoint — X is the reduced dim's
// tile-between loop; the Combine local and the moved store must share the
// element loop's lexical scope.
func reduceT1Graph(t *testing.T) *ir.Graph {
	t.Helper()
	f := newChain("redt1", 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0)}, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 2},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 2},
	})
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	return f.g
}

// emitOptions mirrors tiling.DefaultParams — the same totals the pipeline
// ran under (doc/cg.md §1.3: the driver passes them explicitly).
func emitOptions() cg.Options {
	return cg.Options{MaxCTA: 100, MaxThreadsPerCTA: 512, TargetSM: 86}
}

// ---------------------------------------------------------------------------
// goldens
// ---------------------------------------------------------------------------

// TestGoldenMatmulStore pins the emitted trio over the full default
// pipeline: the searched tiling's real product (SMEM promotion, mappings,
// the CTA/thr/seq iterator chain) rendered end-to-end.
func TestGoldenMatmulStore(t *testing.T) {
	g := matmulStoreGraph("mm")
	runPipeline(t, g)
	res, err := Emit(g, emitOptions())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	golden.Assert(t, "testdata/matmul_store.cu", res.Files["mm.cu"])
	golden.Assert(t, "testdata/matmul_store_cmake.txt", res.Files["CMakeLists.txt"])
}

// TestGoldenReduceWarp pins the reducer surface: the hoisted combiner
// lambda, the WarpReducer construction carrying (init, lambda), the
// Init/Reduce/Combine bracket over k's tile-between loop with slot IVs,
// and the D7 slot reads inside the epilogue re-iteration. The R-A audit
// (R-B, 2026-10-24): the byte-identical regeneration proves the committed
// golden conformant, and ScanBracketLaw re-proves the bracket law
// mechanically over the emitted text (== the golden bytes).
func TestGoldenReduceWarp(t *testing.T) {
	g := reduceWarpGraph(t)
	res, err := Emit(g, emitOptions())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	golden.Assert(t, "testdata/reduce_warp.cu", res.Files["redwarp.cu"])
	if err := ScanBracketLaw(string(res.Files["redwarp.cu"])); err != nil {
		t.Fatalf("the committed reduce golden must satisfy the R-A bracket law mechanically: %v", err)
	}
}

// TestReduceShapesEmit pins the two new reduce shapes at the emitted-text
// level (the compile-level oracle: pre-fix, the >=2-between shape failed
// emission at the dimvar-scope gate): both EMIT successfully, each .cu
// carries the WarpReducer ctor with the 4-operand face, exactly one Init
// call site, exactly one Combine local, and the mechanical R-A bracket-law
// scan is green (for shape B including the same-lexical-scope pin: no
// scope closes between the Combine local and the gated store).
func TestReduceShapesEmit(t *testing.T) {
	t.Run("rank3 multi-between", func(t *testing.T) {
		res, err := Emit(reduce3Graph(t), emitOptions())
		if err != nil {
			t.Fatalf("Emit over the rank-3 reduce (the HI-01 oracle — pre-fix this failed the dimvar-scope gate): %v", err)
		}
		cu := string(res.Files["red3.cu"])
		assertReducerSurface(t, cu)
		if err := ScanBracketLaw(cu); err != nil {
			t.Fatalf("the rank-3 reduce text must satisfy R-A: %v", err)
		}
	})
	t.Run("minimal-legal-tile endpoint", func(t *testing.T) {
		res, err := Emit(reduceT1Graph(t), emitOptions())
		if err != nil {
			t.Fatalf("Emit over the minimal-tile reduce (the HI-02 oracle): %v", err)
		}
		cu := string(res.Files["redt1.cu"])
		assertReducerSurface(t, cu)
		if err := ScanBracketLaw(cu); err != nil {
			t.Fatalf("the minimal-tile reduce text must satisfy R-A (incl. the Combine-local/gated-store same-scope pin): %v", err)
		}
	})
}

// assertReducerSurface pins the common reducer emission facts: the
// WarpReducer ctor with the 4-operand face, exactly one Init call site,
// and exactly one Combine local.
func assertReducerSurface(t *testing.T, cu string) {
	t.Helper()
	if !strings.Contains(cu, "tca::rt::WarpReducer<float> reducer0(") {
		t.Fatal("the .cu must construct the WarpReducer")
	}
	if !strings.Contains(cu, "SmemView<float, 1u>(reinterpret_cast<float*>(tca_smem") {
		t.Fatal("the ctor face must carry the staging SmemView operand")
	}
	if got := strings.Count(cu, "reducer0.Init();"); got != 1 {
		t.Fatalf("exactly one Init call site expected, got %d", got)
	}
	if got := strings.Count(cu, "reducer0.Combine();"); got != 1 {
		t.Fatalf("exactly one Combine local expected, got %d", got)
	}
}

// TestDeterminism pins twin-graph byte identity (the repo's standing
// determinism discipline).
func TestDeterminism(t *testing.T) {
	g1, g2 := matmulStoreGraph("mm"), matmulStoreGraph("mm")
	runPipeline(t, g1)
	runPipeline(t, g2)
	r1, err := Emit(g1, emitOptions())
	if err != nil {
		t.Fatalf("Emit 1: %v", err)
	}
	r2, err := Emit(g2, emitOptions())
	if err != nil {
		t.Fatalf("Emit 2: %v", err)
	}
	for name, want := range r1.Files {
		if got := string(r2.Files[name]); got != string(want) {
			t.Fatalf("twin emission of %s differs", name)
		}
	}
}

// TestKernelMetaNames pins quick-261023-cgq: the KindKernelMeta
// instrumentation node carries the frontend-specified kernel and launcher
// symbols through the whole pipeline to codegen (anchored in the
// Prototype, never swept by DCE, never a parameter).
func TestKernelMetaNames(t *testing.T) {
	g := matmulStoreGraph("mm")
	g.NewKernelMeta("ctor", "MatmulScale", "MatmulScaleLaunch")
	runPipeline(t, g)
	res, err := Emit(g, emitOptions())
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	cu := string(res.Files["mm.cu"])
	// The kernel is anonymous-namespace internal; the Tensor-protocol
	// launcher is the exported surface (quick-261023-ten).
	for _, want := range []string{
		"namespace {",
		"__global__ void __launch_bounds__(512) MatmulScale(",
		"void MatmulScaleLaunch(",
		"const tca::Tensor<float, 2u>& lhs",
		"tca::Tensor<float, 3u>& out",
	} {
		if !strings.Contains(cu, want) {
			t.Fatalf("emitted .cu missing %q", want)
		}
	}
	for _, stale := range []string{"tca_kernel_", "tca_mm_launch", "tca_lambda_"} {
		if strings.Contains(cu, stale) {
			t.Fatalf("emitted .cu still carries the old symbol family %q", stale)
		}
	}
	if !strings.Contains(string(res.Files["CMakeLists.txt"]), "add_library(MatmulScale OBJECT") {
		t.Fatal("the CMake target must follow the kernel symbol")
	}
}

// TestKernelMetaValidation pins the fail-closed identifier gate: a
// user-specified name that cannot compile is a hard error, never silent.
func TestKernelMetaValidation(t *testing.T) {
	g := matmulStoreGraph("mm")
	g.NewKernelMeta("ctor", "bad kernel name!", "")
	runPipeline(t, g)
	if _, err := Emit(g, emitOptions()); err == nil || !strings.Contains(err.Error(), "not a valid C++ identifier") {
		t.Fatalf("an invalid kernel name must fail closed, got %v", err)
	}
}

// TestGates pins the fail-closed entry surface.
func TestGates(t *testing.T) {
	if _, err := Emit(nil, emitOptions()); err == nil || !strings.Contains(err.Error(), "nil graph") {
		t.Fatalf("nil graph must fail closed, got %v", err)
	}
	g := matmulStoreGraph("mm") // constructed, pipeline not run
	if _, err := Emit(g, emitOptions()); err == nil || !strings.Contains(err.Error(), "no attached ASTKernel") {
		t.Fatalf("an unscheduled graph must fail closed naming the AST, got %v", err)
	}
	g2 := matmulStoreGraph("mm")
	runPipeline(t, g2)
	bad := emitOptions()
	bad.MaxCTA = 0
	if _, err := Emit(g2, bad); err == nil || !strings.Contains(err.Error(), "launch totals") {
		t.Fatalf("zero CTAs must fail closed, got %v", err)
	}
}
