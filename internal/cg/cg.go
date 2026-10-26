// Package cg is the code generation driver surface: the backend-neutral
// options and result types every backend (cxx today, mlir later) shares
// (doc/cg.md §4). Codegen is NOT a pipeline pass — it is a consumer invoked
// after (*opt.Manager).Run returns, over the graph plus its attached
// ASTKernel; it never mutates the graph.
package cg

// Options configures one backend emission. The launch totals are supplied
// by the driver (doc/cg.md §1.3); in the fix4 factorization model they
// should match the tiling provenance's derived launch dimensions — the
// driver reads them from the tiling record when it builds the pipeline
// and passes them here (the launch config is not an IR node, LR-15).
//
// NOTE (fix4 T5): cg.Options will be redesigned in T5 to read launch
// dimensions from the graph's tiling provenance directly; for now these
// fields are passed in by the driver / test harness and must match what
// the tiling pass decided.
type Options struct {
	// KernelName overrides the emitted kernel symbol base name; empty uses
	// the graph's name (the KindKernelMeta node, when present, takes
	// precedence over both — quick-261023-cgq).
	KernelName string
	// MaxCTA is gridDim.x (positive). In the fix4 model this is the
	// product of CTA-assigned dims' LoC values; the driver supplies it
	// from the tiling provenance.
	MaxCTA int
	// MaxThreadsPerCTA is blockDim.x (a whole-warp multiple, i.e. a
	// multiple of 32). In the fix4 model this is (Π warp K_d) × 32
	// derived from the tiling record; the driver supplies it.
	MaxThreadsPerCTA int
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
