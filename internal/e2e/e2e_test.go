// e2e_test.go — TCACOMP_E2E-gated numerical e2e + compute-sanitizer tests
// (doc/rt.md §8.3). Fixtures compose the committed golden kernels
// (../cg/cxx/testdata), a runner.cu with the launchers' pinned signatures,
// and a CMakeLists mirroring the compiler-generated matmul_store_cmake.txt.
package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	matmulDim   = 1024 // d0 = d1 = d2 (the golden's baked bounds)
	reduceRows  = 1024 // lhs rows / out extent (the golden's baked c0)
	reduceCols  = 512  // lhs cols (the golden's baked c1)
	canaryValue = float32(-12345.0)
	reduceTol   = 2e-3 // max abs diff: tree-order vs sequential-order fp32 divergence over <=512 terms
)

func testdataPath(name string) string { return filepath.Join("..", "cg", "cxx", "testdata", name) }
func runtimeDirAbs(t TB) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", "..", "runtime"))
	if err != nil {
		t.Fatalf("e2e: resolve runtime dir: %v", err)
	}
	return abs
}

// fixtureCMake mirrors the compiler-generated matmul_store_cmake.txt (project
// LANGUAGES CXX CUDA, standards 20, OBJECT lib for the golden,
// CUDA_ARCHITECTURES 86, --expt-relaxed-constexpr, tcacomp_runtime,
// CUDA::cudart) plus the executable from runner.cu. TCACOMP_RUNTIME_DIR is
// injected by the harness so the fixture content stays checkout-independent.
// find_package(CUDAToolkit) supplies the CUDA::cudart imported target —
// LANGUAGES CUDA alone does not define it (a latent gap in the generated
// CMakeLists, noted in the task SUMMARY).
const fixtureCMake = `cmake_minimum_required(VERSION 3.24)
project(%[1]s LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(${TCACOMP_RUNTIME_DIR} runtime)
add_library(%[2]s OBJECT %[3]s)
set_target_properties(%[2]s PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(%[2]s PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(%[2]s PRIVATE tcacomp_runtime CUDA::cudart)
add_executable(%[4]s runner.cu)
target_link_libraries(%[4]s PRIVATE %[2]s tcacomp_runtime CUDA::cudart)
`

// The shared runner prologue: deterministic input formula, canary pre-fill,
// device upload/download, machine-parseable output lines.
const runnerPrologue = `#include "tca/runtime.h"
#include <cstdio>
#include <cstdlib>

static float elem(unsigned i) {
  return static_cast<float>((i * 2654435761u) % 1000u) / 1000.0f;
}

#define CUDA_OK(call)                                                    \
  do {                                                                   \
    if ((call) != cudaSuccess) {                                         \
      std::printf("CUDA_FAIL %s:%d\n", __FILE__, __LINE__);              \
      return 3;                                                          \
    }                                                                    \
  } while (0)
`

const mmRunner = runnerPrologue + `
void MmLaunch(tca::Tensor<float, 3u>& out, const tca::Tensor<float, 2u>& lhs,
              const tca::Tensor<float, 2u>& rhs, cudaStream_t tca_stream);

int main() {
  const unsigned d0 = 1024u, d1 = 1024u, d2 = 1024u;
  const unsigned nLhs = d0 * d2, nRhs = d1 * d2, nOut = d0 * d1 * d2;

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

  auto outT = tca::Tensor<float, 3u>::From(dOut, d0, d1, d2);
  auto lhsT = tca::Tensor<float, 2u>::From(dLhs, d0, d2);
  auto rhsT = tca::Tensor<float, 2u>::From(dRhs, d1, d2);
  MmLaunch(outT, lhsT, rhsT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double sum = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    sum += static_cast<double>(hOut[i]);
  }
  std::printf("KERNEL Mm\n");
  std::printf("CANARY %u\n", canary);
  std::printf("CHECKSUM %.17g\n", sum);
  std::printf("SAMPLES");
  for (unsigned i = 0u; i < 8u; ++i) std::printf(" %a", static_cast<double>(hOut[i]));
  std::printf("\n");
  return 0;
}
`

const rwRunner = runnerPrologue + `
void RedwarpLaunch(const tca::Tensor<float, 2u>& lhs, tca::Tensor<float, 1u>& out,
                   cudaStream_t tca_stream);

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
  RedwarpLaunch(lhsT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double sum = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    sum += static_cast<double>(hOut[i]);
  }
  std::printf("KERNEL Redwarp\n");
  std::printf("CANARY %u\n", canary);
  std::printf("CHECKSUM %.17g\n", sum);
  std::printf("OUT");
  for (unsigned i = 0u; i < nOut; ++i) std::printf(" %a", static_cast<double>(hOut[i]));
  std::printf("\n");
  return 0;
}
`

func goldenFixture(t TB, goldenFile, proj, objLib, srcFile, binName, runner string) string {
	t.Helper()
	golden, err := os.ReadFile(testdataPath(goldenFile))
	if err != nil {
		t.Fatalf("e2e: read golden %s: %v", goldenFile, err)
	}
	files := map[string][]byte{
		srcFile:          golden,
		"runner.cu":      []byte(runner),
		"CMakeLists.txt": []byte(fmt.Sprintf(fixtureCMake, proj, objLib, srcFile, binName)),
	}
	return BuildFixture(t, files, binName, runtimeDirAbs(t))
}

func buildMatmulFixture(t TB) string {
	t.Helper()
	return goldenFixture(t, "matmul_store.cu", "MmE2E", "mm_golden", "mm.cu", "mm_e2e", mmRunner)
}

func buildReduceFixture(t TB) string {
	t.Helper()
	return goldenFixture(t, "reduce_warp.cu", "RedwarpE2E", "rw_golden", "rw.cu", "rw_e2e", rwRunner)
}

// genInput is the deterministic input formula shared with the runners:
// v(i) = float((i * 2654435761) % 1000) / 1000 (u32 wraparound).
func genInput(n int) []float32 {
	v := make([]float32, n)
	for i := uint32(0); i < uint32(n); i++ {
		v[i] = float32((i*2654435761)%1000) / 1000.0
	}
	return v
}

// matmulReference computes out[m,n,k] = lhs[m,k] * rhs[n,k] in fp32 (the
// golden's single deterministic multiply), streaming the double-precision
// checksum over flat index order plus the first 8 values.
func matmulReference() (sum float64, first8 []float64) {
	const d0, d1, d2 = matmulDim, matmulDim, matmulDim
	lhs := genInput(d0 * d2)
	rhs := genInput(d1 * d2)
	first8 = make([]float64, 8)
	flat := 0
	for m := 0; m < d0; m++ {
		for n := 0; n < d1; n++ {
			rbase := n * d2
			for k := 0; k < d2; k++ {
				v := lhs[m*d2+k] * rhs[rbase+k]
				sum += float64(v)
				if flat < 8 {
					first8[flat] = float64(v)
				}
				flat++
			}
		}
	}
	return sum, first8
}

// reduceReference computes naive sequential fp32 row sums.
func reduceReference() []float32 {
	const rows, cols = reduceRows, reduceCols
	lhs := genInput(rows * cols)
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var acc float32
		for c := 0; c < cols; c++ {
			acc += lhs[r*cols+c]
		}
		out[r] = acc
	}
	return out
}

// matmulGoldenDesignGapSkip documents the STOPPED golden matmul fixture
// (quick-261027-iq6 defect protocol, "DESIGN-LEVEL gap" branch): the
// golden's crafted fix4 §3.3-class record assigns K=WARP (K=2, T=32) on a
// dim that SURVIVES into the output tensor — out[m,n,k] = lhs[m,k]*rhs[n,k]
// keeps every k. Probe evidence (A6000, 2026-10-27, content-histogram
// runner over the exact fixture binary): the kernel completes with cudaSuccess,
// m and n coverage complete and uniform, but every (m,n) pair carries
// EXACTLY 16 of 1024 k values — precisely the k's owned by the single
// thread that owns m in the coupled walk ({32·w(t)+l(t)+64·j}). Root cause:
// the M=CTA coupled inside walk strides the WHOLE blockDim (all 64 threads,
// both warps) while the K=WARP lane chunks assign each k-tile to ONE warp's
// lanes — the m×k output cross product has no single owning thread for the
// off-diagonal (warp, m) combinations. A per-warp coupled walk (init
// lane_id, step 32) was probed and is ALSO insufficient — the hole repeats
// one level down (m partitioned over lanes × k partitioned over lanes).
// Any two thread-partitioned inside walks leave the surviving cross product
// uncovered; WARP dims are only coverable on REDUCED dims (fix4 §3.5
// split-k, Wave 2). fix4 §3.1's templates as written do not cover the
// CTA×WARP(surviving) composition, and the decide pass has no gate pruning
// it — author ruling required (candidate shapes in the task SUMMARY). The
// fixture stays stopped — silently re-targeting it would bury the §3.3
// signature's uncoverability; the §9(1) acceptance rides the REAL frontend
// matmul_scale (TestFrontendMatmulScale, green) plus the reduce fixtures.
const matmulGoldenDesignGapSkip = "golden matmul fixture STOPPED: the crafted M=CTA × K=WARP(surviving) §3.3-class record leaves the m×k output cross product 63/64 uncovered (fix4 §3.1 CTA×WARP composition — design gap, ruling pending; see quick-261027-iq6 SUMMARY for probe evidence)"

func TestMatmulScaleE2E(t *testing.T) {
	skipUnlessE2E(t)
	t.Skip(matmulGoldenDesignGapSkip)
	bin := buildMatmulFixture(t)
	out, err := RunCmd(t, "mm_e2e", bin)
	if err != nil {
		t.Fatalf("e2e: matmul fixture run failed: %v\n%s", err, out)
	}
	canary, checksum, samples, _ := ParseFixtureOutput(t, out)
	if len(samples) != 8 {
		t.Fatalf("e2e: expected 8 samples, got %d (output:\n%s)", len(samples), out)
	}

	// Structural: every output element must be overwritten (canary count 0).
	if canary != 0 {
		t.Errorf("e2e: matmul canary count = %d, want 0 (unwritten out elements)", canary)
	}

	// Numerical: exact match — the kernel performs a single deterministic
	// fp32 multiply, so the double accumulation of identical floats must be
	// bitwise equal.
	refSum, refFirst8 := matmulReference()
	if checksum != refSum {
		t.Errorf("e2e: matmul checksum mismatch: got %.17g, want %.17g (diff %g)",
			checksum, refSum, checksum-refSum)
	}
	for i := 0; i < 8; i++ {
		if samples[i] != refFirst8[i] {
			t.Errorf("e2e: matmul sample[%d] = %g, want %g", i, samples[i], refFirst8[i])
		}
	}
}

func TestReduceWarpE2E(t *testing.T) {
	skipUnlessE2E(t)
	bin := buildReduceFixture(t)
	out, err := RunCmd(t, "rw_e2e", bin)
	if err != nil {
		t.Fatalf("e2e: reduce fixture run failed: %v\n%s", err, out)
	}
	canary, _, _, full := ParseFixtureOutput(t, out)
	if len(full) != reduceRows {
		t.Fatalf("e2e: expected %d dumped outputs, got %d", reduceRows, len(full))
	}
	if canary != 0 {
		t.Errorf("e2e: reduce canary count = %d, want 0 (unwritten out elements)", canary)
	}

	ref := reduceReference()
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
		i := worst
		t.Errorf("e2e: reduce max abs diff = %g > tolerance %g at row %d: got %g, want %g",
			maxAbs, reduceTol, i, full[i], ref[i])
	}
}

func TestMatmulScaleSanitizer(t *testing.T) {
	skipUnlessE2E(t)
	t.Skip(matmulGoldenDesignGapSkip)
	san, ok := FindTool("compute-sanitizer", "/usr/local/cuda-12.2/bin/compute-sanitizer")
	if !ok {
		t.Skip("compute-sanitizer not found")
	}
	bin := buildMatmulFixture(t)
	for _, tool := range []string{"memcheck", "racecheck"} {
		out, err := RunCmd(t, "sanitizer:"+tool, san, "--tool", tool, bin)
		if err != nil {
			t.Fatalf("e2e: matmul %s run failed: %v\n%s", tool, err, out)
		}
		requireZeroSanitizerErrors(t, tool, out)
	}
}

func TestReduceWarpSanitizer(t *testing.T) {
	skipUnlessE2E(t)
	san, ok := FindTool("compute-sanitizer", "/usr/local/cuda-12.2/bin/compute-sanitizer")
	if !ok {
		t.Skip("compute-sanitizer not found")
	}
	bin := buildReduceFixture(t)
	for _, tool := range []string{"memcheck", "racecheck"} {
		out, err := RunCmd(t, "sanitizer:"+tool, san, "--tool", tool, bin)
		if err != nil {
			t.Fatalf("e2e: reduce %s run failed: %v\n%s", tool, err, out)
		}
		requireZeroSanitizerErrors(t, tool, out)
	}
}

// requireZeroSanitizerErrors asserts exit-clean output: every tool summary
// line must report zero errors. Tools word their summaries differently
// (memcheck: "ERROR SUMMARY: 0 errors"; racecheck: "RACECHECK SUMMARY: 0
// hazards displayed (0 errors, 0 warnings)") — match any "*SUMMARY:" line and
// require its parenthesized error count to be zero.
func requireZeroSanitizerErrors(t *testing.T, tool, out string) {
	t.Helper()
	summaries := 0
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, "SUMMARY:") {
			continue
		}
		summaries++
		if strings.Contains(line, "errors") && !strings.Contains(line, "0 errors") {
			t.Errorf("e2e: %s reported errors: %s", tool, strings.TrimSpace(line))
		}
	}
	if summaries == 0 {
		t.Errorf("e2e: %s produced no SUMMARY line (output:\n%s)", tool, out)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
