package opt_test

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"tcacomp/internal/flags"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// stubPass is the test double behind every PM behavior test: it defaults to
// producing StageConstructed (the only Phase-1 window row) over whatever
// graph the test hands to Run.
type stubPass struct {
	id        string
	requires  []string
	produces  opt.Stage
	preserved []string
	run       func(*opt.Context, *ir.Graph) error
}

func (s *stubPass) ID() string          { return s.id }
func (s *stubPass) Requires() []string  { return s.requires }
func (s *stubPass) Produces() opt.Stage { return s.produces }
func (s *stubPass) Preserved() []string { return s.preserved }
func (s *stubPass) Run(ctx *opt.Context, g *ir.Graph) error {
	if s.run != nil {
		return s.run(ctx, g)
	}
	return nil
}

// recording is an Instrumentation that appends its callback events to a
// shared log, named so assertions can check dispatch order.
type recording struct {
	name   string
	events *[]string
}

func (r *recording) BeforePass(p opt.Pass, _ *ir.Graph) {
	*r.events = append(*r.events, r.name+":before:"+p.ID())
}

func (r *recording) AfterPass(p opt.Pass, _ *ir.Graph, _ error) {
	*r.events = append(*r.events, r.name+":after:"+p.ID())
}

// failingEnd implements PipelineEnd with a fixed error.
type failingEnd struct{ err error }

func (f *failingEnd) BeforePass(opt.Pass, *ir.Graph)       {}
func (f *failingEnd) AfterPass(opt.Pass, *ir.Graph, error) {}
func (f *failingEnd) AfterPipeline(*ir.Graph) error        { return f.err }

// setFlag restores a flags.F field after the test, keeping tests
// order-independent.
func setVerifyEach(t *testing.T, v bool) {
	t.Helper()
	old := flags.F.VerifyEach
	flags.F.VerifyEach = v
	t.Cleanup(func() { flags.F.VerifyEach = old })
}

func TestPreconditionRejectsOrder(t *testing.T) {
	iterLower := &stubPass{id: "iter-lower"}
	reduceLower := &stubPass{id: "reduce-lower", requires: []string{"iter-lower"}}

	outOfOrder := opt.NewManager([]opt.Pass{reduceLower, iterLower})
	err := outOfOrder.Run(ir.NewGraph("t"))
	if err == nil {
		t.Fatal("out-of-order pipeline: expected error, got nil")
	}
	for _, want := range []string{"pipeline order", "reduce-lower", "iter-lower"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("out-of-order error %q missing %q", err.Error(), want)
		}
	}

	inOrder := opt.NewManager([]opt.Pass{iterLower, reduceLower})
	if err := inOrder.Run(ir.NewGraph("t")); err != nil {
		t.Fatalf("in-order pipeline: expected success, got %v", err)
	}
}

func TestPipelineEndErrorSurfaces(t *testing.T) {
	// A PipelineEnd error aborts Run even with an EMPTY pass list.
	pm := opt.NewManager(nil)
	pm.AddInstrumentation(&failingEnd{err: errors.New("boom")})
	err := pm.Run(ir.NewGraph("t"))
	if err == nil {
		t.Fatal("expected pipeline-end error, got nil")
	}
	for _, want := range []string{"pipeline end", "boom"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("pipeline-end error %q missing %q", err.Error(), want)
		}
	}

	// VerifyEach registered on an empty pipeline verifies the graph exactly
	// once at pipeline end: an unwired operand fails, naming the node.
	setVerifyEach(t, true)
	broken := ir.NewGraph("broken")
	bad := broken.NewNode(ir.KindBinOp, "test", nil, nil)
	broken.NewReturn("test", bad) // effect root: keeps the unwired binop reachable

	ve := opt.NewVerifyEach(io.Discard)
	pmVerify := opt.NewManager(nil)
	pmVerify.AddInstrumentation(ve)
	err = pmVerify.Run(broken)
	if err == nil {
		t.Fatal("empty-pipeline verify over unwired operand: expected error, got nil")
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("node %d", bad.BasePtr().ID())) {
		t.Errorf("verify error %q does not name node %d", err.Error(), bad.BasePtr().ID())
	}

	// The clean oracle-style graph verifies clean.
	clean := ir.NewGraph("clean")
	f32 := ir.InternScalar(ir.TF32)
	c1 := clean.NewConst("test", f32, 1)
	c2 := clean.NewConst("test", f32, 2)
	clean.NewReturn("test", clean.NewBinOp("test", ir.BinOpAdd, c1, c2))

	pmClean := opt.NewManager(nil)
	pmClean.AddInstrumentation(opt.NewVerifyEach(io.Discard))
	if err := pmClean.Run(clean); err != nil {
		t.Fatalf("empty-pipeline verify over clean graph: expected nil, got %v", err)
	}
}
