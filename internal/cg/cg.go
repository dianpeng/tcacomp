// Package cg is the code generation driver surface: the backend-neutral
// options and result types every backend (cxx today, mlir later) shares
// (doc/cg.md §4). Codegen is NOT a pipeline pass — it is a consumer invoked
// after (*opt.Manager).Run returns, over the graph plus its attached
// ASTKernel; it never mutates the graph.
package cg

// Options configures one backend emission. The launch dimensions are NOT
// driver-supplied anymore (fix4 §5.1, T5 — supersedes cg.md LR-15): the
// backend DERIVES gridDim.x and blockDim.x from the emitted loops'
// soc/dosoc/loc annotations over the ASTKernel's LNTs — gridDim = max
// over LNTs of Π CTA LoC (CuMAP guarantees ≥ 1 CTA dim), blockDim = (max
// over LNTs of Π warp K) × 32, floor 32 — and bakes them as literal
// dim3(...) values in the launcher and __launch_bounds__. The launch
// config is still not an IR node (LR-15); it is annotation-derived at
// Emit time.
type Options struct {
	// KernelName overrides the emitted kernel symbol base name; empty uses
	// the graph's name (the KindKernelMeta node, when present, takes
	// precedence over both — quick-261023-cgq).
	KernelName string
	// MaxCTAPolicy is the §4.3 compiler-emitted POLICY constant — the
	// gridDim cap ValidateLaunch checks against (the MALoC CTA bound,
	// never a launch total). Zero resolves to 512 (= tiling
	// DefaultParams.MaxCTA).
	MaxCTAPolicy int
	// TargetSM is the targeted compute capability major-minor packed as
	// major*100+minor (86 = sm_86, the A6000). Pins the generated
	// CMakeLists CUDA_ARCHITECTURES — never "native".
	TargetSM int
	// Comments selects the generated-code comment instrumentation level
	// (quick-261023-cgq). The zero value means CommentSections — the
	// goldens pin it; CommentNone strips every comment.
	Comments CommentLevel
}

// CommentLevel is the generated-code comment instrumentation level: section
// comments are synthesized from facts the IR already carries (soc/dosoc/
// loc/tile annotations, dim names, loader variants, launch totals) so a
// reviewer can map semantic regions to the final code without reading the
// compiler.
type CommentLevel int

const (
	// CommentDefault resolves to CommentSections (the zero value).
	CommentDefault CommentLevel = iota
	// CommentNone emits no instrumentation comments at all.
	CommentNone
	// CommentSections emits one comment per semantic section (LNT headers,
	// loop lines, fills, reducer brackets, the launch) plus a kernel
	// header block.
	CommentSections
	// CommentVerbose additionally annotates hoisted locals with the pass
	// that minted them (node PassOrigin provenance).
	CommentVerbose
)

// Result is one emission's output files, keyed by file name (e.g.
// "matmul.cu", "CMakeLists.txt"). Deterministic for twin graphs.
type Result struct {
	Files map[string][]byte
}

// Resolved normalizes the zero value to CommentSections (the goldens' level).
func (l CommentLevel) Resolved() CommentLevel {
	if l == CommentDefault {
		return CommentSections
	}
	return l
}
