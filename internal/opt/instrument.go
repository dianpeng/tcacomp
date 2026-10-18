package opt

import (
	"fmt"
	"io"
	"os"
	"slices"
	"time"

	"tcacomp/internal/flags"
	"tcacomp/internal/ir"
)

// Instrumentation is the PM callback surface (PASS-02): verify, dump, and
// timing run as callbacks, never inside passes, so flags.F can disable them
// without editing pass code. BeforePass fires in registration order,
// AfterPass in reverse registration order (stack-like, MLIR-verified);
// AfterPass receives the pass's error on the failure path.
type Instrumentation interface {
	BeforePass(p Pass, g *ir.Graph)
	AfterPass(p Pass, g *ir.Graph, err error)
}

// PipelineEnd is the optional pipeline-end hook an Instrumentation may
// additionally implement. Manager.Run dispatches AfterPipeline after the
// last pass — even with an empty pass list — and returns the first non-nil
// error ("pipeline end: %w"), never swallowed.
type PipelineEnd interface {
	AfterPipeline(*ir.Graph) error
}

// VerifyEach runs ir.Verify after every pass (AfterPass) and once at
// pipeline end (AfterPipeline), the single authoritative end-of-pipeline
// verification (D-10 substrate): an empty pass list is still verified
// exactly once. Violations render as
//
//	verify: node <id> (last touched by "<pass>"): <invariant>: <msg>
//
// lines to the writer; AfterPass has no error path, AfterPipeline returns
// violations as an error the manager surfaces. No-op when
// flags.F.VerifyEach is false (the --verify-each=0 escape hatch, TOOL-06).
type VerifyEach struct {
	W io.Writer // violation output; nil means os.Stderr
}

// NewVerifyEach returns a VerifyEach writing to w (os.Stderr when nil).
func NewVerifyEach(w io.Writer) *VerifyEach {
	if w == nil {
		w = os.Stderr
	}
	return &VerifyEach{W: w}
}

func (v *VerifyEach) render(vs []ir.Violation) {
	for _, vi := range vs {
		fmt.Fprintf(v.W, "verify: node %d (last touched by %q): %s: %s\n",
			vi.NodeID, vi.LastTouchedBy, vi.Invariant, vi.Msg)
	}
}

// BeforePass is a no-op: verification happens after passes.
func (v *VerifyEach) BeforePass(Pass, *ir.Graph) {}

// AfterPass runs ir.Verify(g) after every pass and renders violations.
func (v *VerifyEach) AfterPass(_ Pass, g *ir.Graph, _ error) {
	if !flags.F.VerifyEach {
		return
	}
	v.render(ir.Verify(g))
}

// AfterPipeline runs the FINAL ir.Verify(g) — the one authoritative check
// even when no passes ran — and returns violations as an error.
func (v *VerifyEach) AfterPipeline(g *ir.Graph) error {
	if !flags.F.VerifyEach {
		return nil
	}
	vs := ir.Verify(g)
	v.render(vs)
	if len(vs) == 0 {
		return nil
	}
	return fmt.Errorf("ir verify failed with %d violation(s), first: node %d (last touched by %q): %s: %s",
		len(vs), vs[0].NodeID, vs[0].LastTouchedBy, vs[0].Invariant, vs[0].Msg)
}

// IRDump writes the IR after passes whose ID flags.F.DumpAfter selects
// ("all" or a comma-separated list). TextEmitter/DotEmitter are nil in this
// plan; plan 01-05 wires them to ir.Print / visualization.EmitDot
// post-merge — until then the pinned placeholder comments stand in.
type IRDump struct {
	W           io.Writer              // dump output; nil means os.Stderr
	TextEmitter func(*ir.Graph) string // canonical IR text; nil = placeholder
	DotEmitter  func(*ir.Graph) string // .dot text; nil = placeholder
}

// NewIRDump returns an IRDump writing to w (os.Stderr when nil).
func NewIRDump(w io.Writer) *IRDump {
	if w == nil {
		w = os.Stderr
	}
	return &IRDump{W: w}
}

func dumpAfterSelected(passID string) bool {
	want := flags.F.DumpAfter
	if want == "" {
		return false
	}
	if want == "all" {
		return true
	}
	return slices.Contains(splitComma(want), passID)
}

func splitComma(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	return out
}

// BeforePass is a no-op.
func (d *IRDump) BeforePass(Pass, *ir.Graph) {}

// AfterPass dumps the graph when the pass ID is selected by DumpAfter.
func (d *IRDump) AfterPass(p Pass, g *ir.Graph, _ error) {
	if !dumpAfterSelected(p.ID()) {
		return
	}
	fmt.Fprintf(d.W, "// ---- after %s ----\n", p.ID())
	if flags.F.DumpGraphviz {
		if d.DotEmitter != nil {
			fmt.Fprintln(d.W, d.DotEmitter(g))
			return
		}
		fmt.Fprintf(d.W, "// [dot dump: graph %s after %s — no dot emitter wired]\n", g.Name(), p.ID())
		return
	}
	if d.TextEmitter != nil {
		fmt.Fprintln(d.W, d.TextEmitter(g))
		return
	}
	fmt.Fprintf(d.W, "// [ir dump: graph %s after %s — no text emitter wired]\n", g.Name(), p.ID())
}

// PassTiming accumulates per-pass durations and reports them at pipeline
// end, one "pass <id>: <duration>" line per pass in sorted ID order
// (slices.Sort before reporting — never map iteration, RESEARCH Pitfall 1).
// No-op when flags.F.PassTiming is false (TOOL-06).
type PassTiming struct {
	W         io.Writer // report output; nil means os.Stderr
	durations map[string]time.Duration
	started   time.Time
}

// NewPassTiming returns a PassTiming writing to w (os.Stderr when nil).
func NewPassTiming(w io.Writer) *PassTiming {
	if w == nil {
		w = os.Stderr
	}
	return &PassTiming{W: w, durations: make(map[string]time.Duration)}
}

// BeforePass starts the timer for the running pass.
func (t *PassTiming) BeforePass(Pass, *ir.Graph) {
	if !flags.F.PassTiming {
		return
	}
	t.started = time.Now()
}

// AfterPass adds the elapsed duration to the running pass's ID.
func (t *PassTiming) AfterPass(p Pass, _ *ir.Graph, _ error) {
	if !flags.F.PassTiming || t.started.IsZero() {
		return
	}
	t.durations[p.ID()] += time.Since(t.started)
	t.started = time.Time{}
}

// AfterPipeline writes the sorted per-pass timing report.
func (t *PassTiming) AfterPipeline(*ir.Graph) error {
	if !flags.F.PassTiming {
		return nil
	}
	ids := make([]string, 0, len(t.durations))
	for id := range t.durations {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		fmt.Fprintf(t.W, "pass %s: %s\n", id, t.durations[id])
	}
	return nil
}
