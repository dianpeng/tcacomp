// frontend_test.go — the design-Task-6 acceptance test (doc/syntax-design.md
// §6): the Appendix A matmul_scale program in the NEW frontend syntax,
// compiled by the real compiler pipeline (parse → semacheck → lower →
// opt → emit — NOT a checked-in golden) and run on the GPU against the CPU
// reference. Two fixtures exercise the written-clause form and the
// fully-omitted as-inference form (§7.4). TCACOMP_E2E-gated via
// skipUnlessE2E; the .tca fixtures live in t.TempDir (the loader resolves
// @include relative to the including file), and the emitted .cu joins the
// content-addressed BuildFixture map with a pinned-signature runner.
package e2e

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tcacomp/internal/cg"
	"tcacomp/internal/cg/cxx"
	"tcacomp/internal/compiler"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/pipeline"
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// Frontend runner dims: small enough for a fast GPU run, large enough to
// exercise the tiled path (T=32 tiles on every dim).
const (
	feM, feN, feQ = 64, 128, 64
	feScale       = 1.5
)

// frontendCommon is the minimal @include target (may be empty).
const frontendCommon = `// common — frontend e2e include fixture
`

// frontendMatmulWritten is Appendix A verbatim: every clause written.
const frontendMatmulWritten = `@include "tca-common.tca";
@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="MatMulScale")
kernel matmul_scale(
  lhs: tensor<dtype=f32, shape=[M, N]>,
  rhs: tensor<dtype=f32, shape=[N, Q]>,
  scale: f32,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);
  let output = map
    on(lhs[a][b], rhs[b][c])
    as (a, b, c)
    (l: f32, r: f32) -> f32:
      l * r
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (a, c) -> f32:
      acc + %0
  | map
    on()
    [scale] (v: f32) -> f32:
      v * scale
  ;
  return output;
}
`

// frontendMatmulOmitted drops every inferable clause (§7.4: root map as =
// first-binding-appearance order; piped map as = live axes in order). The
// reduce keeps its explicit as (a, c) — the surviving-dims clause is the
// chain's semantic contract, and the written-clause fixture already covers
// the explicit form.
const frontendMatmulOmitted = `@include "tca-common.tca";
@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="MatMulScale")
kernel matmul_scale(
  lhs: tensor<dtype=f32, shape=[M, N]>,
  rhs: tensor<dtype=f32, shape=[N, Q]>,
  scale: f32,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);
  let output = map
    on(lhs[a][b], rhs[b][c])
    (l: f32, r: f32) -> f32:
      l * r
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (a, c) -> f32:
      acc + %0
  | map
    on()
    [scale] (v: f32) -> f32:
      v * scale
  ;
  return output;
}
`

// The pinned launcher prototype the emitter derives from the graph's param
// order (tensor params in signature order, then the output, then scalars —
// see internal/syntax/build/build.go's declaration order and
// internal/cg/cxx/launch.go's single-pass parameter walk).
const feRunnerTail = `
void MatMulScaleLaunch(const tca::Tensor<float, 2u>& lhs,
                       const tca::Tensor<float, 2u>& rhs,
                       tca::Tensor<float, 2u>& output, float scale,
                       cudaStream_t tca_stream);

int main() {
  const unsigned dM = %[1]du, dN = %[2]du, dQ = %[3]du;
  const unsigned nLhs = dM * dN, nRhs = dN * dQ, nOut = dM * dQ;

  float* hLhs = static_cast<float*>(std::malloc(nLhs * sizeof(float)));
  float* hRhs = static_cast<float*>(std::malloc(nRhs * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hLhs == nullptr || hRhs == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nLhs; ++i) hLhs[i] = elem(i);
  for (unsigned i = 0u; i < nRhs; ++i) hRhs[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dLhs = nullptr;
  float* dRhs = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dLhs, nLhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dRhs, nRhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dLhs, hLhs, nLhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dRhs, hRhs, nRhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto lhsT = tca::Tensor<float, 2u>::From(dLhs, dM, dN);
  auto rhsT = tca::Tensor<float, 2u>::From(dRhs, dN, dQ);
  auto outT = tca::Tensor<float, 2u>::From(dOut, dM, dQ);
  MatMulScaleLaunch(lhsT, rhsT, outT, %[4]g, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
  }
  std::printf("KERNEL FrontendMatmulScale\n");
  std::printf("CANARY %%u\n", canary);
  std::printf("OUT");
  for (unsigned i = 0u; i < nOut; ++i) std::printf(" %%a", static_cast<double>(hOut[i]));
  std::printf("\n");
  return 0;
}
`

// compileFrontendKernel runs the real compiler (the internal/compiler
// facade — the single syntax→backend composition, fix2 R-F) over one fixture
// and returns the emitted files (the .cu — the emitted CMakeLists is
// dropped: the fixture composes its own, mirroring the golden path).
func compileFrontendKernel(t *testing.T, src string) map[string][]byte {
	t.Helper()
	dir := t.TempDir()
	for name, content := range map[string]string{
		"tca-common.tca":   frontendCommon,
		"matmul_scale.tca": src,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("e2e: write fixture %s: %v", name, err)
		}
	}
	mod, err := syntax.ParseModule(lexer.NewFileSet(), filepath.Join(dir, "matmul_scale.tca"))
	if err != nil {
		t.Fatalf("e2e: frontend parse/semacheck failed: %v", err)
	}
	if len(mod.Kernels) != 1 {
		t.Fatalf("e2e: want 1 kernel, got %d", len(mod.Kernels))
	}
	res, err := compiler.CompileKernel(mod.Kernels[0], cg.Options{MaxCTAPolicy: 512, TargetSM: 86})
	if err != nil {
		t.Fatalf("e2e: compile through the facade failed: %v", err)
	}
	files := map[string][]byte{}
	for name, content := range res.Files {
		if strings.HasSuffix(name, ".cu") {
			files[name] = content
		}
	}
	if len(files) == 0 {
		t.Fatalf("e2e: emit produced no .cu (files: %v)", res.Files)
	}
	return files
}

// frontendMatmulScaleReference computes out[m,q] = scale * Σ_n lhs[m,n]*rhs[n,q]
// in fp32 (sequential order — the GPU's tree order diverges within reduceTol).
func frontendMatmulScaleReference(m, n, q int, scale float32) []float32 {
	lhs := genInput(m * n)
	rhs := genInput(n * q)
	out := make([]float32, m*q)
	for mi := 0; mi < m; mi++ {
		for qi := 0; qi < q; qi++ {
			var acc float32
			for ni := 0; ni < n; ni++ {
				acc += lhs[mi*n+ni] * rhs[ni*q+qi]
			}
			out[mi*q+qi] = acc * scale
		}
	}
	return out
}

// feCollapseLine collapses whitespace runs to single spaces and trims —
// whitespace-tolerant line matching so pure formatting drift cannot dodge a
// pin (split lines, collapse, strings.Contains).
func feCollapseLine(line string) string {
	return strings.Join(strings.Fields(line), " ")
}

// feLineContains reports whether any collapsed line of src contains the
// collapsed want.
func feLineContains(src, want string) bool {
	want = feCollapseLine(want)
	for _, line := range strings.Split(src, "\n") {
		if strings.Contains(feCollapseLine(line), want) {
			return true
		}
	}
	return false
}

// fePinScheduleSignature pins the fix4 §3.5 SPLIT-K schedule class on the
// emitted frontend fixture (quick-261027-kzv T8a: the Wave-2 flip makes the
// feM=64, feN=128, feQ=64 matmul_scale the dedicated split-k kernel — the
// fix4 §8 staged item). The selected schedule is the §3.5 vector
// M=CTA(T=256,LoC=64) × N(reduced)=WARP(K=2,T=32) × Q=SEQ(T=256):
//
//   - gridDim 64 = Π CTA LoC over the LNT's CTA dims (M=CTA/64), emitted
//     as a literal dim3 in both the launch and ValidateLaunch;
//   - blockDim 64 = (Π warp K=2)×32 — two physical warps split the reduced
//     N dim (the split-k vector);
//   - smem 65544u = 65536 B rhs SMEM window (the MaxCache cap) + 8 B
//     reducer staging row (`tca_smem + 65536u`, extent 2u = #warps) at a
//     disjoint offset — the v0 pool never overlaps;
//   - 512u — the MaxCTAPolicy constant (cg.Options), the 4th ValidateLaunch
//     arg of the 4-arg contract (fix4 §4.3);
//   - __launch_bounds__(64) — the derived blockDim (fix4 §5.1);
//   - the WarpReducer bracket: construction with the combine function +
//     thread map + staging view, per-element [Init() → N tile-between
//     ThrIterator rounds (K_d·T_d = 2×32 = 64 stride) with lane-chunk
//     inside walks → Reduce(value) per element → Combine()], and the
//     single-owner store gated on is_first_in_cta().
//
// Hermetic by design: the pin catches schedule-class drift in every
// `go test` pass, no GPU required; the numerical acceptance of this exact
// schedule runs in TestFrontendMatmulScale on the A6000 (the dedicated
// fix4 §3.5 split-k matmul e2e). If a legitimate defect fix changes the
// emission, re-derive from the freshly emitted .cu and re-pin in the same
// commit — never force a schedule to match a pin.
func fePinScheduleSignature(t *testing.T, files map[string][]byte) {
	t.Helper()
	pins := []struct{ name, want string }{
		{"derived __launch_bounds__ (blockDim 64 = 2 warps)", "__launch_bounds__(64)"},
		{"4-arg ValidateLaunch (literal derived dim3 grid/block, smem 65544u, MaxCTAPolicy 512u)",
			"ValidateLaunch(dim3(64u, 1u, 1u), dim3(64u, 1u, 1u), 65544u, 512u)"},
		{"literal dim3 kernel launch", "<<<dim3(64u, 1u, 1u), dim3(64u, 1u, 1u), 65544u"},
		{"§3.5 WarpReducer construction (combine fn, thread map, disjoint staging row at 65536)",
			"tca::rt::WarpReducer<float> reducer0("},
		{"§3.5 reducer staging row: #warps x sizeof(T) at tca_smem + 65536, extent 2",
			"tca_smem + 65536u), 2u)"},
		{"§3.5 reduced-dim N tile-between rounds, K_d*T_d = 64 stride",
			"ThrIterator(thread_map1, 0u, 2u, 0u, 64u,"},
	}
	for name, content := range files {
		src := string(content)
		for _, pin := range pins {
			if !feLineContains(src, pin.want) {
				t.Fatalf("e2e: emitted %s lost the fix4 §3.5 split-k schedule signature — %s: no line matches %q\n"+
					"M=CTA/64 x N(reduced)=WARP/2 x Q=SEQ; smem 65544 = 65536 rhs window + 8 reducer staging",
					name, pin.name, pin.want)
			}
		}
	}
}

// TestFrontendMatmulScalePipeline is the HERMETIC positive pin of
// quick-261024-sdk Step 7.1: both Appendix A variants compile through the
// real pipeline — tile-lower's recast §8 fill checks (ORDERING + LAYERING-IV)
// included, the gate the pre-fix1 body model failed on this exact shape —
// and emit a non-empty .cu. No GPU and no nvcc: the assertion ends at the
// emitted files, so this runs in every hermetic `go test` pass. The written
// variant additionally pins the fix4 §3.5 split-k launch/reducer signature
// class (fePinScheduleSignature). The GPU numerical acceptance lives in
// TestFrontendMatmulScale (fix3 D-p: green, no skip — both variants within
// reduceTol, canary 0).
func TestFrontendMatmulScalePipeline(t *testing.T) {
	variants := []struct {
		name string
		src  string
	}{
		{"written", frontendMatmulWritten},
		{"omitted", frontendMatmulOmitted},
	}
	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			files := compileFrontendKernel(t, v.src)
			for name, content := range files {
				if len(content) == 0 {
					t.Fatalf("e2e: emitted %s is empty", name)
				}
				// fix3 D-p downstream pin: the bracket law (R-A) holds over
				// the emitted frontend fixture — assert-only, no code change.
				if err := cxx.ScanBracketLaw(string(content)); err != nil {
					t.Fatalf("e2e: bracket law violated by emitted %s: %v", name, err)
				}
			}
			if v.name == "written" {
				fePinScheduleSignature(t, files)
			}
		})
	}
}

func TestFrontendMatmulScale(t *testing.T) {
	skipUnlessE2E(t)
	variants := []struct {
		name string
		src  string
	}{
		{"written", frontendMatmulWritten},
		{"omitted", frontendMatmulOmitted},
	}
	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			files := compileFrontendKernel(t, v.src)
			runner := runnerPrologue + fmt.Sprintf(feRunnerTail, feM, feN, feQ, feScale)
			var srcFile string
			for name := range files {
				srcFile = name
			}
			files["runner.cu"] = []byte(runner)
			files["CMakeLists.txt"] = []byte(fmt.Sprintf(fixtureCMake,
				"FeMatmulE2E", "fe_mm", srcFile, "fe_mm_e2e"))
			bin := BuildFixture(t, files, "fe_mm_e2e", runtimeDirAbs(t))

			out, err := RunCmd(t, "fe_mm_e2e", bin)
			if err != nil {
				t.Fatalf("e2e: frontend matmul fixture run failed: %v\n%s", err, out)
			}
			canary, _, _, full := ParseFixtureOutput(t, out)
			if len(full) != feM*feQ {
				t.Fatalf("e2e: expected %d dumped outputs, got %d", feM*feQ, len(full))
			}
			if canary != 0 {
				t.Errorf("e2e: frontend matmul canary count = %d, want 0 (unwritten out elements)", canary)
			}

			ref := frontendMatmulScaleReference(feM, feN, feQ, feScale)
			var maxAbs float64
			worst := -1
			for i, got := range full {
				diff := float64(float32(got)) - float64(ref[i])
				if d := abs(diff); d > maxAbs {
					maxAbs = d
					worst = i
				}
			}
			if maxAbs > reduceTol {
				t.Errorf("e2e: frontend matmul max abs diff = %g > tolerance %g at flat %d: got %g, want %g",
					maxAbs, reduceTol, worst, full[worst], ref[worst])
			}
		})
	}
}

// TestFrontendMatmulScaleSanitizer extends the sanitizer coverage to the
// frontend fixture (fix3 D-p): memcheck + racecheck over the real compiled
// kernel, zero errors — the derived fill placement and the composition's
// ordinary memory-chain ordering are race-freedom-relevant surfaces.
func TestFrontendMatmulScaleSanitizer(t *testing.T) {
	skipUnlessE2E(t)
	san, ok := FindTool("compute-sanitizer", "/usr/local/cuda-12.2/bin/compute-sanitizer")
	if !ok {
		t.Skip("compute-sanitizer not found")
	}
	files := compileFrontendKernel(t, frontendMatmulWritten)
	var srcFile string
	for name := range files {
		srcFile = name
	}
	files["runner.cu"] = []byte(runnerPrologue + fmt.Sprintf(feRunnerTail, feM, feN, feQ, feScale))
	files["CMakeLists.txt"] = []byte(fmt.Sprintf(fixtureCMake,
		"FeMatmulSan", "fe_mm_san", srcFile, "fe_mm_san_e2e"))
	bin := BuildFixture(t, files, "fe_mm_san_e2e", runtimeDirAbs(t))
	for _, tool := range []string{"memcheck", "racecheck"} {
		out, err := RunCmd(t, "sanitizer:"+tool, san, "--tool", tool, bin)
		if err != nil {
			t.Fatalf("e2e: frontend matmul %s run failed: %v\n%s", tool, err, out)
		}
		requireZeroSanitizerErrors(t, tool, out)
	}
}

// ---------------------------------------------------------------------------
// fix3 D-p battery: graph-level (hermetic) pins over the frontend fixture
// ---------------------------------------------------------------------------

// compileFrontendGraph drives the real pipeline to the GRAPH (not the
// emission): parse → semacheck → Build → Dematerialize → the pass windows.
// tail selects the cleanup tail: true runs the exact user path
// (pipeline.Run); false stops after LOWER (fix3 D-p battery negative (a)
// pins what the lower output alone guarantees).
func compileFrontendGraph(t *testing.T, src string, tail bool) *ir.Graph {
	t.Helper()
	dir := t.TempDir()
	for name, content := range map[string]string{
		"tca-common.tca":   frontendCommon,
		"matmul_scale.tca": src,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("e2e: write fixture %s: %v", name, err)
		}
	}
	mod, err := syntax.ParseModule(lexer.NewFileSet(), filepath.Join(dir, "matmul_scale.tca"))
	if err != nil {
		t.Fatalf("e2e: frontend parse/semacheck failed: %v", err)
	}
	if len(mod.Kernels) != 1 {
		t.Fatalf("e2e: want 1 kernel, got %d", len(mod.Kernels))
	}
	g, err := mod.Kernels[0].Build()
	if err != nil {
		t.Fatalf("e2e: kernel build failed: %v", err)
	}
	if err := g.Dematerialize(); err != nil {
		t.Fatalf("e2e: dematerialize failed: %v", err)
	}
	if tail {
		if err := pipeline.Run(g); err != nil {
			t.Fatalf("e2e: pipeline run failed: %v", err)
		}
		return g
	}
	passes := append(pipeline.HIRPasses(), pipeline.LOWERPasses()...)
	if err := opt.NewManager(passes).Run(g); err != nil {
		t.Fatalf("e2e: HIR+LOWER run failed: %v", err)
	}
	return g
}

// feWalkItems visits every AST item node with its containing block.
func feWalkItems(blk *ir.ASTBlock, fn func(n ir.Node, containing *ir.ASTBlock)) {
	for _, it := range blk.Items() {
		if it.Node != nil {
			fn(it.Node, blk)
		}
		if it.Block != nil {
			feWalkItems(it.Block, fn)
		}
	}
}

// feBlockContains reports whether root is target or an ancestor of it.
func feBlockContains(root, target *ir.ASTBlock) bool {
	if root == target {
		return true
	}
	for _, it := range root.Items() {
		if it.Block != nil && feBlockContains(it.Block, target) {
			return true
		}
	}
	return false
}

// feOperandClosure collects n's transitive Input-operand closure (the R-I
// coordinate closure — data-dependency edges only).
func feOperandClosure(n ir.Node) map[ir.Node]bool {
	cl := map[ir.Node]bool{}
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || cl[x] {
			return
		}
		cl[x] = true
		for _, op := range ir.OperandsOf(x) {
			visit(op)
		}
	}
	visit(n)
	return cl
}

// feImpactClosure collects n's transitive ImpactBy (memory-chain
// predecessor) closure — the ordinary effect chain, walked backwards.
func feImpactClosure(n ir.Node) map[ir.Node]bool {
	cl := map[ir.Node]bool{}
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || cl[x] {
			return
		}
		cl[x] = true
		for _, u := range ir.ImpactBy(x) {
			visit(u.Def())
		}
	}
	visit(n)
	return cl
}

// TestFrontendMatmulScaleLowerNoTailStillVerifies is fix3 D-p battery
// negative (a): with the TAIL removed, the lower output still verifies —
// the sea may hold garbage (here: the duplicate literal consts the tail's
// dedup exists to erase, the fix2 D-e measurement's drift class) while the
// items stay clean. The verifier and the anchor are liveness-scoped: dead
// weight in the sea is never a violation.
func TestFrontendMatmulScaleLowerNoTailStillVerifies(t *testing.T) {
	g := compileFrontendGraph(t, frontendMatmulWritten, false)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("the TAIL-less lower output must verify clean (garbage in the sea, not in the items), got: %v", vs)
	}
	type constKey struct {
		ty   *ir.NumericType
		bits uint64
	}
	counts := map[constKey]int{}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindConst {
			continue
		}
		v, ok := ir.ConstValueOf(n)
		if !ok {
			continue
		}
		ty := ir.ConstTypeOf(n)
		if ty == nil {
			ty = ir.TypeOf(n)
		}
		nt, ok := ty.(*ir.NumericType)
		if !ok || nt == nil {
			continue
		}
		counts[constKey{nt, math.Float64bits(v)}]++
	}
	dupes := 0
	for _, c := range counts {
		if c > 1 {
			dupes++
		}
	}
	if dupes == 0 {
		t.Fatal("the no-TAIL run must leave garbage in the sea (live duplicate consts) — otherwise negative (a) pins nothing")
	}
}

// TestFrontendMatmulScaleFillCompositionScope is fix3 D-p battery negative
// (c): the re-placed fill composition (D-i's MemNew + tile-shaped GMEM
// load + class store) lands DOMINATED-BY-SCOPE — inside the body of every
// surviving loop whose IV its coordinate closure references (the frontend
// between-round loop X; D-j closing fix2's D-c) — and the buffer's window
// loads chain AFTER the store (the ordinary memory chain intact).
func TestFrontendMatmulScaleFillCompositionScope(t *testing.T) {
	g := compileFrontendGraph(t, frontendMatmulWritten, true)
	k := g.AST()

	// The fill compositions: a class store whose target is a MemNew buffer
	// and whose value is a GMEM load.
	type fillSite struct {
		store  ir.Node
		buffer ir.Node
	}
	var fills []fillSite
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		if class, ok := ir.StorageClassOf(n); !ok || class == ir.StorageGMEM {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) < 2 || ops[0] == nil || ops[1] == nil {
			continue
		}
		if ops[1].KindOf() != ir.KindMemNew || ops[1].BasePtr().Erased() {
			continue
		}
		if ops[0].KindOf() != ir.KindMemLoad {
			continue
		}
		if class, ok := ir.StorageClassOf(ops[0]); !ok || class != ir.StorageGMEM {
			continue
		}
		fills = append(fills, fillSite{store: n, buffer: ops[1]})
	}
	if len(fills) == 0 {
		t.Fatal("the frontend matmul must carry at least one fill composition (the between-block cache)")
	}

	for _, fs := range fills {
		closure := feOperandClosure(fs.store)

		// (i) dominated-by-scope (the SchedBlock pin): the store's block is
		// inside the body of every loop whose IV the composition's
		// coordinate closure references.
		var storeBlk *ir.ASTBlock
		feWalkItems(k.Body, func(n ir.Node, containing *ir.ASTBlock) {
			if n.BasePtr() == fs.store.BasePtr() {
				storeBlk = containing
			}
		})
		if storeBlk == nil {
			t.Fatalf("fill store %d is not an AST item — the composition must be placed", fs.store.BasePtr().ID())
		}
		if sb := fs.store.BasePtr().SchedBlock(); sb != storeBlk {
			t.Fatalf("fill store %d's SchedBlock stamp %v != its containing block %v", fs.store.BasePtr().ID(), sb, storeBlk)
		}
		refLoops := 0
		var scope func(blk *ir.ASTBlock)
		scope = func(blk *ir.ASTBlock) {
			for _, it := range blk.Items() {
				if it.Block == nil {
					continue
				}
				if iv := it.Block.IV(); iv != nil && !iv.BasePtr().Erased() && closure[iv] {
					refLoops++
					if !feBlockContains(it.Block, storeBlk) {
						t.Fatalf("fill store %d's block is NOT dominated by the scope of the loop its closure references (IV node %d) — the D-j re-placement regressed",
							fs.store.BasePtr().ID(), iv.BasePtr().ID())
					}
				}
				scope(it.Block)
			}
		}
		scope(k.Body)
		if refLoops == 0 {
			t.Fatalf("fill store %d's closure references no surviving loop — the R-I derivation had no anchor", fs.store.BasePtr().ID())
		}

		// (ii) memory-chain order intact: every live window load of the
		// buffer chains (transitively) after the fill store.
		windows := 0
		for _, n := range g.AllocatedNodes() {
			if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemLoad {
				continue
			}
			ops := ir.OperandsOf(n)
			if len(ops) == 0 || ops[0] == nil || ops[0].BasePtr() != fs.buffer.BasePtr() {
				continue
			}
			windows++
			if !feImpactClosure(n)[fs.store] {
				t.Fatalf("window load %d of buffer %d does not chain after the fill store %d — the composition's memory-chain order broke",
					n.BasePtr().ID(), fs.buffer.BasePtr().ID(), fs.store.BasePtr().ID())
			}
		}
		if windows == 0 {
			t.Fatalf("buffer %d carries no window loads — the fill would be dead", fs.buffer.BasePtr().ID())
		}
	}
}
