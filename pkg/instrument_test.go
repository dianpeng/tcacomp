package pkg_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/flags"
	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
	"tcacomp/pkg"
)

// nopPass is a stub pipeline step for instrumentation tests: ID "nop", no
// requirements, no preservation, Run is a no-op.
type nopPass struct{}

func (nopPass) ID() string                         { return "nop" }
func (nopPass) Requires() []string                 { return nil }
func (nopPass) Produces() pass.Stage               { return pass.StageConstructed }
func (nopPass) Preserved() []string                { return nil }
func (nopPass) Run(*pass.Context, *ir.Graph) error { return nil }

func setDumpAfter(t *testing.T, v string) {
	t.Helper()
	old := flags.F.DumpAfter
	flags.F.DumpAfter = v
	t.Cleanup(func() { flags.F.DumpAfter = old })
}

func setDumpGraphviz(t *testing.T, v bool) {
	t.Helper()
	old := flags.F.DumpGraphviz
	flags.F.DumpGraphviz = v
	t.Cleanup(func() { flags.F.DumpGraphviz = old })
}

func setPassTiming(t *testing.T, v bool) {
	t.Helper()
	old := flags.F.PassTiming
	flags.F.PassTiming = v
	t.Cleanup(func() { flags.F.PassTiming = old })
}

func setVerifyEach(t *testing.T, v bool) {
	t.Helper()
	old := flags.F.VerifyEach
	flags.F.VerifyEach = v
	t.Cleanup(func() { flags.F.VerifyEach = old })
}

// returnSumGraph builds the minimal verifiable graph: one statement,
// `return 1.0 + 2.0` over two typed constants. The return stage root is
// what makes the printer render real content (a bare scalar-expr root
// prints only the statement header — chainOf walks stages only).
func returnSumGraph() *ir.Graph {
	g := ir.NewGraph("t")
	f32 := ir.InternScalar(ir.TF32)
	sum := g.NewBinOp("test", ir.BinOpAdd,
		g.NewConst("test", f32, 1), g.NewConst("test", f32, 2))
	g.AddStatement(g.NewReturn("test", sum))
	return g
}

func runNop(t *testing.T, g *ir.Graph, ins ...pass.Instrumentation) error {
	t.Helper()
	pm := pass.NewManager([]pass.Pass{nopPass{}})
	for _, in := range ins {
		pm.AddInstrumentation(in)
	}
	return pm.Run(g)
}

// TestInstrumentationsRealDumpOutput (TOOL-06) asserts that the standard
// instrumentation set emits REAL dump content end-to-end through Manager.Run:
// text dumps route through ir.Print (a real statement header, not the
// unwired-emitter placeholder line) and graphviz dumps through
// visualization.EmitDot (`digraph ` + `subgraph cluster_` lines, not the
// placeholder).
func TestInstrumentationsRealDumpOutput(t *testing.T) {
	t.Run("text dump", func(t *testing.T) {
		setDumpAfter(t, "all")
		setDumpGraphviz(t, false)
		g := returnSumGraph()
		var buf bytes.Buffer
		if err := runNop(t, g, pkg.DefaultInstrumentations(&buf)...); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "// ---- after nop ----") {
			t.Errorf("dump missing pass header: %q", out)
		}
		// The statement header number comes from the graph's single
		// monotonic ID counter shared with node IDs (IR-05), so assert it
		// dynamically rather than as a literal: the load-bearing check is
		// that REAL ir.Print content follows the header — the statement's
		// stage text `return 1.0 + 2.0` — never the placeholder comment.
		stmtID := g.Block().Statements()[0].ID()
		if !strings.Contains(out, fmt.Sprintf("statement #%d:", stmtID)) {
			t.Errorf("dump missing real ir.Print statement header: %q", out)
		}
		if !strings.Contains(out, "return 1.0 + 2.0") {
			t.Errorf("dump missing real ir.Print stage content: %q", out)
		}
		// The placeholder an unwired emitter would print is a comment line
		// starting "// [ir dump:" (see internal/pass/instrument.go); it must
		// never appear once DefaultInstrumentations has bound ir.Print.
		if strings.Contains(out, "// [ir dump:") {
			t.Errorf("dump contains placeholder line — TextEmitter not wired: %q", out)
		}
	})

	t.Run("dot dump", func(t *testing.T) {
		setDumpAfter(t, "all")
		setDumpGraphviz(t, true)
		var buf bytes.Buffer
		if err := runNop(t, returnSumGraph(), pkg.DefaultInstrumentations(&buf)...); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "digraph ") {
			t.Errorf("dot dump missing digraph line: %q", out)
		}
		if !strings.Contains(out, "subgraph cluster_") {
			t.Errorf("dot dump missing statement cluster: %q", out)
		}
		// The unwired-emitter placeholder is a "// [dot dump:" comment line;
		// it must never appear once EmitDot is bound.
		if strings.Contains(out, "// [dot dump:") {
			t.Errorf("dot dump contains placeholder line — DotEmitter not wired: %q", out)
		}
	})
}

// TestDefaultInstrumentationsComposition asserts the returned slice carries
// the standard set — VerifyEach, IRDump, PassTiming — and that each one is
// behaviorally live: the verify gate surfaces violations at pipeline end,
// the dump fires under DumpAfter, and PassTiming reports a sorted
// `pass nop: <duration>` line.
func TestDefaultInstrumentationsComposition(t *testing.T) {
	ins := pkg.DefaultInstrumentations(&bytes.Buffer{})
	if len(ins) != 3 {
		t.Fatalf("DefaultInstrumentations returned %d instrumentations, want 3 (VerifyEach, IRDump, PassTiming)", len(ins))
	}
	var haveVerify, haveDump, haveTiming bool
	for _, in := range ins {
		switch in.(type) {
		case *pass.VerifyEach:
			haveVerify = true
		case *pass.IRDump:
			haveDump = true
		case *pass.PassTiming:
			haveTiming = true
		default:
			t.Errorf("unexpected instrumentation type %T", in)
		}
	}
	if !haveVerify || !haveDump || !haveTiming {
		t.Errorf("composition incomplete: VerifyEach=%v IRDump=%v PassTiming=%v", haveVerify, haveDump, haveTiming)
	}

	t.Run("verify gate live", func(t *testing.T) {
		setVerifyEach(t, true)
		var buf bytes.Buffer
		broken := ir.NewGraph("broken")
		broken.AddStatement(broken.NewNode(ir.KindBinOp, "test", nil, nil))
		err := runNop(t, broken, pkg.DefaultInstrumentations(&buf)...)
		if err == nil || !strings.Contains(err.Error(), "pipeline end") {
			t.Fatalf("verify gate: Run error = %v, want pipeline-end verification failure", err)
		}
		if !strings.Contains(buf.String(), "verify:") {
			t.Errorf("verify gate: no violation rendered: %q", buf.String())
		}
	})

	t.Run("dump and timing live", func(t *testing.T) {
		setVerifyEach(t, true)
		setDumpAfter(t, "all")
		setPassTiming(t, true)
		var buf bytes.Buffer
		if err := runNop(t, returnSumGraph(), pkg.DefaultInstrumentations(&buf)...); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "// ---- after nop ----") {
			t.Errorf("IRDump did not fire under DumpAfter=all: %q", out)
		}
		if !strings.Contains(out, "pass nop: ") {
			t.Errorf("PassTiming report missing 'pass nop: ' line: %q", out)
		}
	})
}
