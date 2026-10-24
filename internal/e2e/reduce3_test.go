// reduce3_test.go — the 261024-bz9 e2e expansion: two full-pipeline
// GENERATED reduce kernels (TCACOMP_E2E-gated, doc/rt.md §8.3), built from
// the live compiler (graph → typeinfer → crafted-record tile-lower →
// schedule → final-lower → cxx.Emit), not from committed goldens:
//
//	Shape A (HI-01): the rank-3 last-axis reduce out[i,j] = sum_k
//	a[i,j,k] with TWO T>1 non-reduced dims — the >=2-between-dims R-A
//	bracket shape (decisions {CTA,100,32}/{SEQ,1,64}/{WARP,8,64}).
//	Shape B (HI-02): the T==1 sweep endpoint (the redwarp graph at
//	{CTA,100,1}/{WARP,8,1}) — the T==1 flat bracket's Combine local and
//	the moved store share one C++ lexical scope, and the WARP-single
//	reduced dim rides the single-form mirror gate.
//
// R-C discipline: reduce comparisons are tolerance-based ONLY (canary 0,
// max abs diff <= 2e-3); the fold order is irrelevant to correctness and
// never pinned. Sanitizers (memcheck + racecheck) run on both fixtures —
// racecheck is the decided gate for the 2026-10-24 single-barrier WarpReducer
// protocol's bracket-reuse question.
package e2e

import (
	"fmt"
	"testing"

	"tcacomp/internal/cg"
	"tcacomp/internal/cg/cxx"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/finallower"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling"
	"tcacomp/internal/opt/typeinfer"
)

// ---------------------------------------------------------------------------
// local fixture helpers (the emit_test.go chain idiom, self-contained)
// ---------------------------------------------------------------------------

type r3fx struct {
	g     *ir.Graph
	dims  []ir.Node
	iters []ir.Node
}

func newR3Chain(name string, extents ...float64) *r3fx {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	f := &r3fx{g: g}
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

func (f *r3fx) tensor(name string, dims ...int) ir.Node {
	refs := make([]ir.DimRef, len(dims))
	for i, d := range dims {
		refs[i] = ir.DimRef{Name: fmt.Sprintf("d%d", d)}
	}
	return f.g.NewInputParam("ctor", ir.RoleTensor, name, ir.InternTensor(ir.TF32, refs))
}

func (f *r3fx) iv(i int) ir.Node { return f.g.NewInductionVariable("ctor", f.iters[i]) }

// runReduce3Pipeline drives the real tile-lower + schedule + final-lower
// over a crafted verified record and emits the kernel trio.
func runReduce3Pipeline(t *testing.T, name string, extents []float64, decisions []tiling.IterDecision, kernel, launcher string) []byte {
	t.Helper()
	f := newR3Chain(name, extents...)
	lhsRank := len(extents)
	lhsRefs := make([]int, lhsRank)
	for i := range lhsRefs {
		lhsRefs[i] = i
	}
	lhs := f.tensor("lhs", lhsRefs...)
	outRefs := make([]int, lhsRank-1)
	for i := range outRefs {
		outRefs[i] = i
	}
	out := f.tensor("out", outRefs...)
	env := ir.NewEnv()
	axes := make([]ir.Node, len(f.iters))
	for i := range f.iters {
		axes[i] = f.iv(i)
	}
	load := f.g.NewMemLoadChained("ctor", env, lhs, axes)
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	outAxes := make([]ir.Node, lhsRank-1)
	for i := range outAxes {
		outAxes[i] = f.iv(i)
	}
	store := f.g.NewMemStoreChained("ctor", env, out, outAxes, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	// The naming metadata rides AFTER typeinfer on the crafted-record path
	// (the full pipeline's pre-105 passes handle it; the standalone pass's
	// catalog is closed) and before schedule, which anchors it in the
	// Prototype.
	f.g.NewKernelMeta("ctor", kernel, launcher)
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
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	res, err := cxx.Emit(f.g, cg.Options{MaxCTA: 100, MaxThreadsPerCTA: 512, TargetSM: 86})
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	cu, ok := res.Files[name+".cu"]
	if !ok {
		keys := make([]string, 0, len(res.Files))
		for k := range res.Files {
			keys = append(keys, k)
		}
		t.Fatalf("emit result carries no %s.cu (keys: %v)", name, keys)
	}
	// The mechanical R-A bracket-law proof over the emitted text (the same
	// scanner the reduce golden gate runs).
	if err := cxx.ScanBracketLaw(string(cu)); err != nil {
		t.Fatalf("the emitted %s kernel must satisfy the R-A bracket law: %v", kernel, err)
	}
	return cu
}

// ---------------------------------------------------------------------------
// runners
// ---------------------------------------------------------------------------

const r3RunnerA = runnerPrologue + `
void Reduce3Launch(const tca::Tensor<float, 3u>& lhs, tca::Tensor<float, 2u>& out, cudaStream_t tca_stream);

int main() {
  const unsigned d0 = 64u, d1 = 32u, d2 = 512u;
  const unsigned nLhs = d0 * d1 * d2, nOut = d0 * d1;

  float* hLhs = static_cast<float*>(std::malloc(nLhs * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hLhs == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nLhs; ++i) hLhs[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dLhs = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dLhs, nLhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dLhs, hLhs, nLhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto lhsT = tca::Tensor<float, 3u>::From(dLhs, d0, d1, d2);
  auto outT = tca::Tensor<float, 2u>::From(dOut, d0, d1);
  Reduce3Launch(lhsT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double sum = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    sum += static_cast<double>(hOut[i]);
  }
  std::printf("KERNEL Reduce3\n");
  std::printf("CANARY %u\n", canary);
  std::printf("CHECKSUM %.17g\n", sum);
  std::printf("OUT");
  for (unsigned i = 0u; i < nOut; ++i) std::printf(" %a", static_cast<double>(hOut[i]));
  std::printf("\n");
  return 0;
}
`

const r3RunnerB = runnerPrologue + `
void ReduceT1Launch(const tca::Tensor<float, 2u>& lhs, tca::Tensor<float, 1u>& out, cudaStream_t tca_stream);

int main() {
  const unsigned d0 = 1024u, d1 = 512u;
  const unsigned nLhs = d0 * d1, nOut = d0;

  float* hLhs = static_cast<float*>(std::malloc(nLhs * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hLhs == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nLhs; ++i) hLhs[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dLhs = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dLhs, nLhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dLhs, hLhs, nLhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto lhsT = tca::Tensor<float, 2u>::From(dLhs, d0, d1);
  auto outT = tca::Tensor<float, 1u>::From(dOut, d0);
  ReduceT1Launch(lhsT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double sum = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    sum += static_cast<double>(hOut[i]);
  }
  std::printf("KERNEL ReduceT1\n");
  std::printf("CANARY %u\n", canary);
  std::printf("CHECKSUM %.17g\n", sum);
  std::printf("OUT");
  for (unsigned i = 0u; i < nOut; ++i) std::printf(" %a", static_cast<double>(hOut[i]));
  std::printf("\n");
  return 0;
}
`

// ---------------------------------------------------------------------------
// fixtures + references + tests
// ---------------------------------------------------------------------------

func buildReduce3Fixture(t *testing.T) string {
	t.Helper()
	cu := runReduce3Pipeline(t, "red3", []float64{64, 32, 512}, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 64},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	}, "Reduce3", "Reduce3Launch")
	files := map[string][]byte{
		"red3.cu":        cu,
		"runner.cu":      []byte(r3RunnerA),
		"CMakeLists.txt": []byte(fmt.Sprintf(fixtureCMake, "Reduce3E2E", "reduce3_k", "red3.cu", "reduce3_e2e")),
	}
	return BuildFixture(t, files, "reduce3_e2e", runtimeDirAbs(t))
}

func buildReduceT1Fixture(t *testing.T) string {
	t.Helper()
	cu := runReduce3Pipeline(t, "redt1", []float64{1024, 512}, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 1},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 1},
	}, "ReduceT1", "ReduceT1Launch")
	files := map[string][]byte{
		"redt1.cu":       cu,
		"runner.cu":      []byte(r3RunnerB),
		"CMakeLists.txt": []byte(fmt.Sprintf(fixtureCMake, "ReduceT1E2E", "reducet1_k", "redt1.cu", "reducet1_e2e")),
	}
	return BuildFixture(t, files, "reducet1_e2e", runtimeDirAbs(t))
}

// reduce3Reference computes sequential fp32 out[i][j] = sum_k lhs[i][j][k].
func reduce3Reference() []float32 {
	const d0, d1, d2 = 64, 32, 512
	lhs := genInput(d0 * d1 * d2)
	out := make([]float32, d0*d1)
	for i := 0; i < d0; i++ {
		for j := 0; j < d1; j++ {
			var acc float32
			base := (i*d1 + j) * d2
			for k := 0; k < d2; k++ {
				acc += lhs[base+k]
			}
			out[i*d1+j] = acc
		}
	}
	return out
}

// TestReduce3E2E is the HI-01 proof: the rank-3 multi-between reduce runs
// on the A6000 with canary 0 and tolerance-only numerics against the Go
// sequential reference.
func TestReduce3E2E(t *testing.T) {
	skipUnlessE2E(t)
	bin := buildReduce3Fixture(t)
	out, err := RunCmd(t, "reduce3_e2e", bin)
	if err != nil {
		t.Fatalf("e2e: reduce3 fixture run failed: %v\n%s", err, out)
	}
	canary, _, _, full := ParseFixtureOutput(t, out)
	if len(full) != 64*32 {
		t.Fatalf("e2e: expected %d dumped outputs, got %d", 64*32, len(full))
	}
	if canary != 0 {
		t.Errorf("e2e: reduce3 canary count = %d, want 0 (unwritten out elements)", canary)
	}
	ref := reduce3Reference()
	var maxAbs float64
	worst := -1
	for i, got := range full {
		diff := float64(float32(got)) - float64(ref[i])
		if d := abs(diff); d > maxAbs {
			maxAbs, worst = d, i
		}
	}
	t.Logf("e2e: reduce3 canary %d, max abs diff %g (tolerance %g)", canary, maxAbs, reduceTol)
	if maxAbs > reduceTol {
		i := worst
		t.Errorf("e2e: reduce3 max abs diff = %g > tolerance %g at flat %d: got %g, want %g",
			maxAbs, reduceTol, i, full[i], ref[i])
	}
}

// TestReduceT1E2E is the HI-02 proof: the T==1 sweep endpoint compiles
// under nvcc (the same-scope Combine local/store) and matches the
// sequential reference within tolerance.
func TestReduceT1E2E(t *testing.T) {
	skipUnlessE2E(t)
	bin := buildReduceT1Fixture(t)
	out, err := RunCmd(t, "reducet1_e2e", bin)
	if err != nil {
		t.Fatalf("e2e: reducet1 fixture run failed: %v\n%s", err, out)
	}
	canary, _, _, full := ParseFixtureOutput(t, out)
	if len(full) != 1024 {
		t.Fatalf("e2e: expected %d dumped outputs, got %d", 1024, len(full))
	}
	if canary != 0 {
		t.Errorf("e2e: reducet1 canary count = %d, want 0 (unwritten out elements)", canary)
	}
	ref := reduceReference() // the same 1024x512 row sums as the golden shape
	var maxAbs float64
	worst := -1
	for i, got := range full {
		diff := float64(float32(got)) - float64(ref[i])
		if d := abs(diff); d > maxAbs {
			maxAbs, worst = d, i
		}
	}
	t.Logf("e2e: reducet1 canary %d, max abs diff %g (tolerance %g)", canary, maxAbs, reduceTol)
	if maxAbs > reduceTol {
		i := worst
		t.Errorf("e2e: reducet1 max abs diff = %g > tolerance %g at row %d: got %g, want %g",
			maxAbs, reduceTol, i, full[i], ref[i])
	}
}

// TestReduce3Sanitizer runs memcheck + racecheck over both new fixtures —
// racecheck is the decided gate for the 2026-10-24 single-barrier WarpReducer
// protocol's bracket-reuse question.
func TestReduce3Sanitizer(t *testing.T) {
	skipUnlessE2E(t)
	san, ok := FindTool("compute-sanitizer", "/usr/local/cuda-12.2/bin/compute-sanitizer")
	if !ok {
		t.Skip("compute-sanitizer not found")
	}
	fixtures := []struct{ name, bin string }{
		{"reduce3", buildReduce3Fixture(t)},
		{"reducet1", buildReduceT1Fixture(t)},
	}
	for _, fx := range fixtures {
		for _, tool := range []string{"memcheck", "racecheck"} {
			out, err := RunCmd(t, "sanitizer:"+fx.name+":"+tool, san, "--tool", tool, fx.bin)
			if err != nil {
				t.Fatalf("e2e: %s %s run failed: %v\n%s", fx.name, tool, err, out)
			}
			requireZeroSanitizerErrors(t, tool, out)
		}
	}
}
