// fix5t2_one_tier_test.go — quick-261029-v9k follow-up, RE-DERIVED under the
// §3.1 ruling (author, 2026-10-30, iq6 candidate-1). History: the R3
// re-choice (warps on a NON-reduced dim) exposed two latent one-tier-path
// defects (GPU-proven: canary unwritten outputs + memcheck Invalid
// __local__ reads on row_sum/softmax) — the executor's cg/finallower fixes
// (RED 722a652 / GREEN fd72179) are real and stay locked at unit level
// (the reduce_warp golden + TestReducerOneTierSeqLadder). The GPU failure
// itself was then root-caused deeper: the C=WARP re-choice was the
// SURVIVING-WARP class — the iq6 §3.1 CTA-coupled-inside × lane-chunk
// cross-product hole (canary exactly 1/64 of the output space, the
// "16/1024 k values" class) — a LOWERING defect, not an emission defect.
// The ruling prunes the class (warpDimOf §3.1 prune + the tile-lower
// mirror), so the real row_sum returns bit-identically to its pre-T2
// multi-tier schedule (sha256 72db430…, GPU-verified 14/14 at every prior
// battery). THIS pin now locks the pruned state:
//
//   - the emitted row_sum is the multi-tier WarpReducer class (the
//     restored pre-T2 schedule);
//   - §3.1: NO surviving dim carries a WARP digit (the LNT factorization
//     puts thread/warp on the ELIMINATED dim K only);
//   - the one-tier SeqReducer class is unreachable from real kernels
//     under the prune — when the §3.1 cross-product lowering is designed
//     and the fences lift, re-instate the one-tier fixture control and
//     pins A/B below (they are the ready-made RED gate for that lifting).
//
// The two emission properties the fixes landed (ungated one-tier store,
// lane-origin walk init) remain pinned hermetically at unit level and
// apply the moment the one-tier class is reachable again.

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

// TestFix5T2OneTierCollapseEmission pins the row_sum emission class.
// RE-DERIVED under fix6 §3 (quick-261030-v62): the pre-fix6 pin locked
// the MULTI-TIER WarpReducer schedule ("the restored pre-T2 class") —
// under the F6 per-site FRAG decline the all-SEQ collapse was
// unrepresentable in the search (its covered window priced over
// MaxFragPerThread and declined, the qwo gate-3 mechanism). fix6 §3
// prices the window honestly, and the honest winner on the [R,K,C]
// middle-reduce shape is now the ALL-SEQ COLLAPSE: R=CTA(16,T256),
// C=SEQ(1,T256), K=SEQ(1,T256), blockDim 32 — the per-thread K reduction
// lowers through the ONE-TIER SeqReducer (no WarpReducer anywhere), with
// the per-thread Frag<float, 2, 256, 256> windows (256 KB/thread — local
// memory until the fix6 register-blocking codegen lands; legality is the
// suite + memcheck's job per the fix6 §5.3 protocol, performance is
// recorded honestly in the task SUMMARY). The §3.1 legality pin is
// UNTOUCHED and load-bearing as ever: no SURVIVING dim carries
// thread/warp (C stays sequential — the GPU-broken surviving-WARP class
// stays unrepresentable).
//
// Re-instatement contract: when the §3.1 cross-product lowering lands and
// the warpDimOf prune + rejectWarpSurvivingDims mirror lift, this test
// re-gains its one-tier fixture control (SeqReducer, no WarpReducer) and
// the pins A (ungated epilogue store) / B (lane-origin walk init) — see
// the file header.
func TestFix5T2OneTierCollapseEmission(t *testing.T) {
	cu := compileTcaSource(t, "row_sum.tca", rowSumOneTierSrc)

	// 1. The fix6 §3 winner: the all-SEQ collapse lowers the per-thread K
	//    reduction through the ONE-TIER SeqReducer — the multi-tier
	//    WarpReducer ceremony class lost the search to the honest block
	//    pricing (was: "must emit WarpReducer" while the collapse was
	//    F6-declined — that decline is the defect fix6 §3 removes).
	//    RE-DERIVED under fix7 §4 slice 1b (quick-261101-mt9): the qualified
	//    one-tier flat-bracket shape is AMORTIZED — the SeqReducer ceremony
	//    (Init/Reduce/Combine per output element) is replaced by the
	//    chunk-wide Frag accumulator (init walk + inlined combiner +
	//    store walk), so the collapse now emits ZERO reducer objects of
	//    either class; the one-tier fact this pin guards (no WarpReducer
	//    anywhere) is kept UNWEAKENED, and the zero-ceremony assertion
	//    pins the amortization (red evidence: red-task2.txt + the
	//    TestFix7Bracket pin in fix7bracket_test.go).
	if strings.Contains(cu, "tca::rt::WarpReducer<float>") {
		t.Fatal("fix6 §3: the honest winner is the all-SEQ collapse (one-tier SeqReducer) — the multi-tier WarpReducer ceremony must not appear")
	}
	if strings.Contains(cu, "tca::rt::SeqReducer") {
		t.Fatal("fix7 §4 1b: the qualified collapse is amortized — the SeqReducer ceremony must be gone (zero Init/Reduce/Combine sites)")
	}
	if !regexp.MustCompile(`tca::rt::Frag<float, 1u, \d+u> tca_f\d+;`).MatchString(cu) {
		t.Fatal("fix7 §4 1b: the amortized collapse must declare the chunk-wide acc Frag block")
	}

	// 2. §3.1: a surviving dim carrying thread/warp is the GPU-broken
	//    class (canary 1/64) and must be unrepresentable. Under the fix6
	//    §3 collapse there is no WARP digit at all (K reduces
	//    per-thread) — trivially pruned; the pin stays load-bearing for
	//    the day the WARP digit re-enters: it may sit ONLY on the
	//    eliminated K dim.
	for _, line := range strings.Split(cu, "\n") {
		if !strings.Contains(line, "LNT #0:") {
			continue
		}
		if strings.Contains(line, "C=thread/warp") {
			t.Fatalf("§3.1: the surviving C dim must not carry a WARP digit, LNT line: %s", strings.TrimSpace(line))
		}
	}
}
