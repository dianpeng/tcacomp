// fix7vec_test.go — quick-261101-oqg (fix7 §5 slice 1c) e2e pin: the
// emission-side eligibility chain through the REAL optimizer, byte-pinned
// against the 1b sha256 manifest.
//
// For the six 1b-amortized kernels (row_sum, softmax, scale_copy, axpy3,
// dup_expr, dup_const) every per-thread FRAG fill destination with a
// 16B-clean trailing extent must carry "alignas(16) " on its decl (the
// loader's vec16 dst-side check holds by construction), and stripping
// exactly that spelling must reproduce the 1b emission BYTE-FOR-BYTE —
// the sha256 of the stripped text equals the 1b manifest hash, proving
// the ONLY delta vs 1b is the alignas decl lines (any chunk-dim change
// fails this check: the Task-2 bias must be a no-op on the suite).
// The matmul family + mlp + layernorm + transpose stay bit-identical to
// the 1b manifest VERBATIM (the slice touches no matmul scheduling; their
// fill windows are SMEM-class or their trailing extents are not
// 16B-clean).
//
// The 1b manifest hashes come from quick-261101-mt9's
// after-sha256-manifest.txt (the 1b emission IS this task's BEFORE — the
// 1b precedent of copying the prior slice's table verbatim). The compiler
// facade options here (MaxCTAPolicy 512, TargetSM 86) are exactly
// cmd/tcacomp's sm86Options, so the emitted bytes are the manifest bytes.
//
// RED gate (pre-1c): the six carry no alignas — the positive assertion
// fails (the strip check passes trivially at RED: stripping nothing is
// the identity, so it is a GREEN-only proof); the eight unchanged kernels
// pass at RED and GREEN alike (regression guards).

package e2e

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

// fix7VecSuite maps every suite kernel to its suite-tree source and its
// 1b sha256 (after-sha256-manifest.txt of quick-261101-mt9).
var fix7VecSuite = []struct {
	kernel   string
	group    string
	file     string
	hash1b   string
	amortized bool
}{
	// The six 1b-amortized kernels — the only emissions slice 1c may change.
	{"row_sum", "runtime", "row_sum.tca", "22ecbed531bea38d50266dce135dc68581ce11e993f86d55979990a3b4fb7c4f", true},
	{"softmax", "fusion", "softmax.tca", "00c5cc8972a046af006cc9dc1de1a890b93561f91da47ad2b09eb864ad426c72", true},
	{"scale_copy", "runtime", "scale_copy.tca", "b3d03c46286b9f717faca1653ab6896d7f81402857a38d7f4719a8f1820cbb59", true},
	{"axpy3", "fusion", "axpy3.tca", "d1b1c0ba8ab3eaca21a7f5ff9d49fe5669776db2dd23bd9ed2183646026daae8", true},
	{"dup_expr", "dedup", "dup_expr.tca", "f09566c59a0c2860e7b1696253249ce85b1dceec65dc1c16d070568096c7dda7", true},
	{"dup_const", "dedup", "dup_const.tca", "2ba4722758f25cd3ce0c36754c9d3d61e44dcf36bfbccea189c7c79ccd6e71d3", true},
	// The unchanged set — bit-identical to the 1b manifest by hard gate.
	{"gemm", "llm", "gemm.tca", "e90a57f99470511728eb15856e8d71b120654afcf5eac6a8b956da605779c04d", false},
	{"mlp", "fusion", "mlp.tca", "f32f4f7a970fff6ab70842de618eeb8a77042b0c3436552deb9ec51ab88d2cef", false},
	{"mm_small", "tiling", "mm_small.tca", "06f12d4859219632eb59de7398fefa4c892fb1c501f7c9a8fd2ce7f2a44b89f6", false},
	{"mm_mid", "tiling", "mm_mid.tca", "ca0d787e9b76075968209a2cd4874f0b7a983a97107a5ebd5a0cc021490e5b7c", false},
	{"mm_splitk", "tiling", "mm_splitk.tca", "daf32e95f555e39dfa6c68a2a5a66bd88003b65b26695aad89869c8a85945b18", false},
	{"qk_score", "llm", "qk_score.tca", "ada248c0f560f86ce9a9b57b83b03fe52787c758a06f7a459d7bff01d7c1cf61", false},
	{"layernorm", "llm", "layernorm.tca", "5ec89e29ee4990b3949ec6a6719d2b5ff52c3478fdc917a3bee05af0cf569174", false},
	{"transpose", "tiling", "transpose.tca", "83724012ff67aa4bd5de8b80dc147c79302ec0867333638b84f3887e25d1145d", false},
}

// vecAlignasPrefix is the exact decl-line delta slice 1c introduces.
const vecAlignasPrefix = "alignas(16) tca::rt::Frag<"

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// TestFix7VecEligibility is fix7 §5 slice 1c's e2e pin through the real
// optimizer: the amortized six gain alignas(16) on exactly the qualifying
// fill-dst decls (strip-proof: the stripped text hash-equals the 1b
// manifest), and the unchanged eight are bit-identical to the 1b manifest.
func TestFix7VecEligibility(t *testing.T) {
	for _, k := range fix7VecSuite {
		k := k
		t.Run(k.kernel, func(t *testing.T) {
			cu := compileSuiteKernel(t, k.group, k.file)
			if k.amortized {
				if !strings.Contains(cu, vecAlignasPrefix) {
					t.Fatalf("fix7 §5 1c (%s): at least one per-thread FRAG fill destination with a 16B-clean trailing extent must carry %q (the loader's vec16 dst-side check holds by construction); none found", k.kernel, vecAlignasPrefix)
				}
				stripped := strings.ReplaceAll(cu, vecAlignasPrefix, "tca::rt::Frag<")
				if got := sha256Hex(stripped); got != k.hash1b {
					t.Fatalf("fix7 §5 1c (%s): stripping exactly the alignas(16) decl prefix must reproduce the 1b emission byte-for-byte (the ONLY delta is the decl lines — a chunk-dim change means the Task-2 bias was not a no-op); stripped sha256 %s != 1b manifest %s", k.kernel, got, k.hash1b)
				}
				return
			}
			if got := sha256Hex(cu); got != k.hash1b {
				t.Fatalf("fix7 §5 1c (%s regression guard): the emission must stay bit-identical to the 1b manifest; sha256 %s != %s", k.kernel, got, k.hash1b)
			}
		})
	}
	// A summary line so the RED evidence shows the kernel-by-kernel state.
	t.Logf("fix7 §5 1c suite sweep complete: %d kernels", len(fix7VecSuite))
}
