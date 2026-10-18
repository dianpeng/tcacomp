package pkg

import (
	"io"
	"strings"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/visualization"
)

// DefaultInstrumentations returns the standard compiler instrumentation set —
// VerifyEach, IRDump (with the real emitters wired), PassTiming — writing to
// w (Phase 5's cmd/ passes os.Stderr). It is the one constructor the CLI
// consumes to get dump/verify/timing behavior gated by flags.F:
//
//   - flags.F.VerifyEach gates per-pass and pipeline-end ir.Verify.
//   - flags.F.DumpAfter ("all" or comma-separated pass IDs) selects which
//     passes dump; flags.F.DumpGraphviz routes the dump to .dot emission
//     instead of canonical text.
//   - flags.F.PassTiming accumulates per-pass durations and reports them in
//     sorted ID order at pipeline end.
//
// The dump emitters are bound to the canonical printers built in plans
// 01-03: TextEmitter is ir.Print (TOOL-04 text dump) and DotEmitter adapts
// visualization.EmitDot into a string (TOOL-03 graphviz dump). Without this
// wiring the IRDump instrumentation would emit placeholder comments in place
// of real IR content (TOOL-06).
func DefaultInstrumentations(w io.Writer) []opt.Instrumentation {
	dump := opt.NewIRDump(w)
	dump.TextEmitter = ir.Print
	dump.DotEmitter = func(g *ir.Graph) string {
		var b strings.Builder
		// EmitDot is emit-only: it returns nil for a nil graph and cannot
		// fail against a strings.Builder. Should that ever change, the
		// partial buffer is returned rather than panicking — a dump must
		// never abort the pipeline (T-1-05-03).
		_ = visualization.EmitDot(&b, g)
		return b.String()
	}
	return []opt.Instrumentation{
		opt.NewVerifyEach(w),
		dump,
		opt.NewPassTiming(w),
	}
}
