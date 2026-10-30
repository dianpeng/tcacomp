// fix5t2_one_tier_test.go — quick-261029-v9k follow-up (author ruling): the
// R3 collapse re-choice (warps on a NON-reduced element dim, the reduced
// dim SEQ — the one-tier SeqReducer ladder) exposed two latent defects on
// the real row_sum/softmax kernels (GPU-proven: canary unwritten outputs
// + memcheck Invalid __local__ read). These EMIT-LEVEL pins lock the fixed
// emission shape hermetically (no GPU — the qwo e88f049 precedent: a
// compile-level/emit-level pin is the RED gate; the orchestrator suite is
// the numerical acceptance):
//
//   - Defect A: the cg prescan gated EVERY Combine()-closure store on
//     is_first_in_cta() — correct for the CTA-collective multi-tier
//     WarpReducer (one elected writer stores the combined result), WRONG
//     for the one-tier SeqReducer whose Combine is an identity over EACH
//     THREAD'S OWN accumulator: every thread owns distinct output
//     elements, so the gate drops every non-first thread's stores (the
//     observed canary). The one-tier epilogue store must be UNGATED.
//   - Defect B: finallower's buildElementLoop rebuilt the output-element
//     loops as fresh UNIFORM init-0 walks (correct for multi-tier — every
//     thread must walk every element; its WARP sharding lives on the
//     REDUCED dim inside the bracket, never rebuilt). For the one-tier
//     collapse the WARP dim IS an element dim: its lane-chunk walk's
//     lane-origin init (lane x ceil(T/32)) is what makes the chunk-local
//     FRAG coordinate and the store coordinate correct, and the uniform
//     rebuild reset it to 0 — the FRAG access subtracted the lane origin
//     from the now-origin-less local index (negative for lanes >= 1 — the
//     observed Invalid __local__ read). The element loops must keep their
//     original per-thread walks (lane-origin init intact).
//
// The fixture is the real testing/kernels/runtime/row_sum.tca shape
// ([R,K,C] -> [R,C], reduce the MIDDLE dim — the layout whose coalescing
// axis is the non-reduced C, which R3 re-chooses onto the collapse).
package e2e

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"tcacomp/internal/cg"
	"tcacomp/internal/compiler"
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// rowSumOneTierSrc is testing/kernels/runtime/row_sum.tca verbatim (the
// [R,K,C] middle-reduce layout R3 re-chooses to the one-tier collapse:
// C=thread/warp, K=sequential, SeqReducer).
const rowSumOneTierSrc = `@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="RowSum")
kernel row_sum(
  x: tensor<dtype=f32, shape=[R, K, C]>,
  z: tensor<dtype=f32, shape=[R, K, C]>,
) -> tensor<dtype=f32, shape=[R, C]>
{
  dimvar (i, k, j);
  let out = map
    on(x[i][k][j], z[i][k][j])
    (v: f32, w: f32) -> f32:
      v * w
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (i, j) -> f32:
      acc + %0
  ;
  return out;
}
`

// compileTcaSource runs the real compiler facade over one standalone .tca
// source (no @include) and returns the emitted .cu text — the same
// hermetic path as compileFrontendKernel, generalized past the
// matmul_scale fixture naming.
func compileTcaSource(t *testing.T, fname, src string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, fname)
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatalf("e2e: write fixture: %v", err)
	}
	mod, err := syntax.ParseModule(lexer.NewFileSet(), p)
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
	for name, content := range res.Files {
		if strings.HasSuffix(name, ".cu") {
			return string(content)
		}
	}
	t.Fatalf("e2e: emit produced no .cu (files: %v)", res.Files)
	return ""
}

// laneOriginInitRe matches a walk iterator whose init is a NON-CONST
// lane-derived product (the hoisted `(vN * cM)` lane-chunk origin form —
// the same rendering the working multi-tier K-inside walk produces, e.g.
// `SeqIterator((v7 * c3), 1u, v23)` in the pre-R3 row_sum). The uniform
// rebuild renders a bare `SeqIterator(0u, 1u, ...)` — the defect-B shape.
var laneOriginInitRe = regexp.MustCompile(`SeqIterator\(\(v\d+ \* c\d+\), 1u,`)

// TestFix5T2OneTierCollapseEmission pins the two fixed emission properties
// of the R3 one-tier collapse kernel (the row_sum [R,K,C] shape):
//
//  A. NO is_first_in_cta() anywhere: the one-tier SeqReducer's per-thread
//     accumulator makes every thread the owner of ITS OWN output elements
//     — the single-owner gate (R2/R3, correct for the CTA-collective
//     WarpReducer) would drop every non-first thread's stores (GPU-proven:
//     canary = unwritten outputs on row_sum/softmax).
//  B. The WARP-covered element walk keeps its LANE-ORIGIN init: the
//     chunk-local FRAG coordinate (insideIV - lane x c) and the store
//     coordinate (betweenOrigin + insideIV) are only correct when the
//     walk IV carries the lane-chunk init (GPU-proven: memcheck Invalid
//     __local__ read, iq6 class, on row_sum/softmax).
//
// Positive controls: the kernel IS the one-tier class (SeqReducer present,
// no WarpReducer) and the chunk-local FRAG subtraction survives.
func TestFix5T2OneTierCollapseEmission(t *testing.T) {
	cu := compileTcaSource(t, "row_sum.tca", rowSumOneTierSrc)

	if !strings.Contains(cu, "tca::rt::SeqReducer<float>") {
		t.Fatal("fixture control: the R3 re-choice must select the one-tier SeqReducer class")
	}
	if strings.Contains(cu, "tca::rt::WarpReducer<") || strings.Contains(cu, "tca::rt::GlobalReducer<") {
		t.Fatal("fixture control: the one-tier collapse must not construct a Warp/GlobalReducer")
	}

	// Pin A — the epilogue store is ungated (no single-owner gate for the
	// per-thread ladder). RED pre-fix: the cg prescan gates every
	// Combine()-closure store.
	if strings.Contains(cu, "is_first_in_cta()") {
		t.Fatal("defect A: the one-tier SeqReducer's epilogue store must NOT be gated on is_first_in_cta() — every thread owns its own output elements (GPU-proven canary on row_sum/softmax)")
	}

	// Pin B — a lane-origin (non-const, lane-derived) walk init survives in
	// the emitted element nest. RED pre-fix: the uniform element-loop
	// rebuild renders every walk as SeqIterator(0u, 1u, ...).
	if !laneOriginInitRe.MatchString(cu) {
		t.Fatal("defect B: the WARP-covered element walk lost its lane-origin init (SeqIterator((vN * cM), 1u, ...)) — the uniform rebuild reset it to 0, breaking the chunk-local FRAG coordinate and the store coordinate (GPU-proven Invalid __local__ read on row_sum/softmax)")
	}

	// Control: the chunk-local FRAG access form survives (insideIV - lane x c).
	chunkLocal := false
	for _, line := range strings.Split(cu, "\n") {
		c := feCollapseLine(line)
		if strings.Contains(c, ".Get(") && strings.Contains(c, " - v") {
			chunkLocal = true
			break
		}
	}
	if !chunkLocal {
		t.Fatal("fixture control: the C1b chunk-local FRAG access (insideIV - lane x c) must survive on the collapse path")
	}
}
