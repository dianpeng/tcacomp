package cxx

// ---------------------------------------------------------------------------
// fix7 §5 slice 1c RED pin (quick-261101-oqg, design/fix/fix7.md §5):
// the EMISSION pin — alignas(16) on per-thread FRAG fill destinations.
// ---------------------------------------------------------------------------
//
// The loader's 128-bit vec16 path (runtime/tca/loader.h copy_row) checks
// BOTH endpoints per row; the destination side is a per-thread Frag's
// Data() + row*trailing. Frag::data_ (runtime/tca/view.h) carries NO
// member alignas, so the class alignment is alignof(T) = 4 for float and
// the dst-side check fails on EVERY row regardless of the emitted shape —
// unless the DECLARATION carries alignas(16) (valid on a variable
// declaration; the object then reports 16-byte alignment, and for the
// emitted rank-2 fill windows RowMajorShape<2> is 16 B, so data_ sits at
// offset 16 and Data() + row*trailing is 16B-clean exactly when
// trailing*elemBytes % 16 == 0). Slice 1c's decl rule: the emitter computes
// the set of FRAG buffers that are per-thread fill destinations
// (LoadTileThread dsts — the identical FRAG-class v==0 classification
// memory.go's stmtFill uses) in a pre-pass, and prints "alignas(16) "
// before tca::rt::Frag< iff the buffer is in that set AND its trailing
// extent is 16B-clean. EVERYTHING else — SMEM views, the 1b acc Frag (not
// a fill dst), chunk-1 layernorm/transpose fills, all matmul-family decls
// — prints byte-unchanged (the matmul/layernorm/transpose sha256
// bit-identity gate is a hard Task-3 gate).
//
// The fixtures reuse fragchunk_test.go's fragChunkGraph (the 1a/1b
// fixture idiom): the const-extent arrangement is exactly the 1b acc/fill
// layout — a per-thread LoadTileThread into Frag<float, 2u, 16u, 4u>
// (trailing 4 = 16B-clean) PLUS the amortized rank-1 acc block
// Frag<float, 1u, 16u> that must render UNCHANGED (it is not a fill dst).
// The odd-trailing subtest guards the rule's condition: a fill destination
// whose trailing extent is NOT 16B-clean gains NO alignas (the scalar
// fallback is the mechanism's contract — an unconditioned alignas would
// change bytes on kernels that cannot benefit).
//
// RED gate (pre-1c): no alignas anywhere — the positive assertion and the
// byte golden fail; the acc-unchanged and odd-trailing assertions PASS at
// RED (they pin the current shape — regression guards at GREEN). RED
// evidence: red-task2.txt in the task dir.

import (
	"regexp"
	"strings"
	"testing"

	"tcacomp/internal/golden"
)

var (
	// vecFillAlignedRe is the 1c target: the per-thread fill destination
	// with a 16B-clean trailing extent carries the alignas(16) spelling.
	vecFillAlignedRe = regexp.MustCompile(`alignas\(16\) tca::rt::Frag<float, 2u, 16u, 4u> tca_f\d+;`)
	// vecAccPlainRe pins the 1b accumulator block UNCHANGED: not a fill
	// dst, so no alignas — the exact 1b decl line.
	vecAccPlainRe = regexp.MustCompile(`(?m)^  tca::rt::Frag<float, 1u, 16u> tca_f\d+;$`)
	// vecAccAlignedRe is the regression guard: the acc must never gain the
	// spelling (a rank-1 Frag's data_ sits at offset 8 — the spelling
	// would not even make it eligible; only fill destinations qualify).
	vecAccAlignedRe = regexp.MustCompile(`alignas\(16\) tca::rt::Frag<float, 1u`)
	// vecOddDeclRe is the odd-trailing fill destination (3*4 = 12 B row,
	// not 16B-clean): the rule's condition must hold — no alignas.
	vecOddDeclRe = regexp.MustCompile(`(?m)^  tca::rt::Frag<float, 2u, 3u, 3u> tca_f\d+;$`)
)

// TestFix7VecFillDecl is fix7 §5 slice 1c's unit emission pin: the
// per-thread FRAG fill destination with a 16B-clean trailing extent
// renders "alignas(16) tca::rt::Frag<...>", the 1b acc block renders
// byte-unchanged, and a fill destination whose trailing extent is not
// 16B-clean gains no alignas. The byte golden pins the whole emission.
func TestFix7VecFillDecl(t *testing.T) {
	t.Run("fill dst alignas, acc unchanged", func(t *testing.T) {
		// The 1b const-extent arrangement (fragchunk_test.go idiom):
		// covered [C, e4], budget 256 B -> chunk [16, 4]; the reduce
		// amortizes (1b) into the acc block + the fill-dst window.
		g := fragChunkGraph(t, "fix7vec", "constextent", 64, 256, 256)
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the 1c vec fixture: %v", err)
		}
		cu := string(res.Files["fix7vec.cu"])
		if !vecFillAlignedRe.MatchString(cu) {
			t.Fatalf("fix7 §5 1c: the per-thread fill destination with a 16B-clean trailing extent (Frag<float, 2u, 16u, 4u>) must render 'alignas(16) tca::rt::Frag<...>' — the loader's vec16 dst-side check holds by construction, not by luck; no match of %s in:\n%s", vecFillAlignedRe, cu)
		}
		if !vecAccPlainRe.MatchString(cu) {
			t.Fatalf("fix7 §5 1c: the 1b accumulator block must render its 1b decl line byte-unchanged; no match of %s in:\n%s", vecAccPlainRe, cu)
		}
		if vecAccAlignedRe.MatchString(cu) {
			t.Fatalf("fix7 §5 1c: the accumulator block is NOT a fill destination — the alignas spelling is scoped to LoadTileThread dsts only; found %s in:\n%s", vecAccAlignedRe, cu)
		}
		golden.Assert(t, "testdata/fix7vec.cu", res.Files["fix7vec.cu"])
	})

	t.Run("odd trailing stays scalar", func(t *testing.T) {
		// Dynamic [3,3] window (36 B <= the 256 B budget — no halving):
		// trailing 3 -> 12 B rows, NOT 16B-clean. The fill destination must
		// render its plain decl — the rule's condition, not an unconditioned
		// prefix (this guards the matmul/layernorm bit-identity gate).
		g := fragChunkGraph(t, "fix7vecodd", "dynamic", 3, 3, 256)
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the odd-trailing fixture: %v", err)
		}
		cu := string(res.Files["fix7vecodd.cu"])
		if !vecOddDeclRe.MatchString(cu) {
			t.Fatalf("fix7 §5 1c (negative control): the odd-trailing fill destination (Frag<float, 2u, 3u, 3u>, 12 B rows) must still render its plain decl; no match of %s in:\n%s", vecOddDeclRe, cu)
		}
		if strings.Contains(cu, "alignas(16)") {
			t.Fatalf("fix7 §5 1c (negative control): a trailing extent whose row is not 16B-clean must gain NO alignas (the vec16 gate is trailing %% 4 == 0 — the spelling would be dead bytes); found alignas(16) in:\n%s", cu)
		}
	})
}
