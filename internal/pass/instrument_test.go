package pass_test

import (
	"bytes"
	"strings"
	"testing"

	"tcacomp/internal/flags"
	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
)

func setFlag(t *testing.T, dst *string, v string) {
	t.Helper()
	old := *dst
	*dst = v
	t.Cleanup(func() { *dst = old })
}

func setBoolFlag(t *testing.T, dst *bool, v bool) {
	t.Helper()
	old := *dst
	*dst = v
	t.Cleanup(func() { *dst = old })
}

func TestInstrumentation(t *testing.T) {
	t.Run("callback order", func(t *testing.T) {
		var events []string
		r1 := &recording{name: "r1", events: &events}
		r2 := &recording{name: "r2", events: &events}
		pm := pass.NewManager([]pass.Pass{
			&stubPass{id: "a"},
			&stubPass{id: "b"},
		})
		pm.AddInstrumentation(r1)
		pm.AddInstrumentation(r2)
		if err := pm.Run(ir.NewGraph("t")); err != nil {
			t.Fatalf("Run: %v", err)
		}
		want := []string{
			"r1:before:a", "r2:before:a", "r2:after:a", "r1:after:a",
			"r1:before:b", "r2:before:b", "r2:after:b", "r1:after:b",
		}
		if len(events) != len(want) {
			t.Fatalf("got %d events, want %d: %v", len(events), len(want), events)
		}
		for i, w := range want {
			if events[i] != w {
				t.Errorf("event %d = %q, want %q (all: %v)", i, events[i], w, events)
			}
		}
	})

	t.Run("verify-each gate", func(t *testing.T) {
		broken := ir.NewGraph("broken")
		broken.AddStatement(broken.NewNode(ir.KindBinOp, "test", nil, nil))

		// VerifyEach=false: the instrumentation is inert, even over a graph
		// whose verification would fail.
		setVerifyEach(t, false)
		var buf bytes.Buffer
		pmOff := pass.NewManager(nil)
		pmOff.AddInstrumentation(pass.NewVerifyEach(&buf))
		if err := pmOff.Run(broken); err != nil {
			t.Fatalf("VerifyEach=false: expected nil, got %v", err)
		}
		if buf.Len() != 0 {
			t.Errorf("VerifyEach=false: expected no output, got %q", buf.String())
		}

		// VerifyEach=true: the same graph fails verification at pipeline end.
		flags.F.VerifyEach = true
		var bufOn bytes.Buffer
		pmOn := pass.NewManager(nil)
		pmOn.AddInstrumentation(pass.NewVerifyEach(&bufOn))
		if err := pmOn.Run(broken); err == nil {
			t.Fatal("VerifyEach=true: expected verify error, got nil")
		}
		if !strings.Contains(bufOn.String(), "verify: node") {
			t.Errorf("VerifyEach=true: expected rendered violations, got %q", bufOn.String())
		}
	})

	t.Run("irdump selects pass id", func(t *testing.T) {
		setFlag(t, &flags.F.DumpAfter, "iter-lower")
		var buf bytes.Buffer
		pm := pass.NewManager([]pass.Pass{
			&stubPass{id: "iter-lower"},
			&stubPass{id: "reduce-lower"},
		})
		pm.AddInstrumentation(pass.NewIRDump(&buf))
		if err := pm.Run(ir.NewGraph("g")); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "// ---- after iter-lower ----") {
			t.Errorf("dump missing iter-lower header: %q", out)
		}
		if !strings.Contains(out, "// [ir dump: graph g after iter-lower — no text emitter wired]") {
			t.Errorf("dump missing pinned placeholder: %q", out)
		}
		if strings.Contains(out, "reduce-lower") {
			t.Errorf("dump fired for unselected pass: %q", out)
		}
	})

	t.Run("pass timing reports sorted", func(t *testing.T) {
		setBoolFlag(t, &flags.F.PassTiming, true)
		var buf bytes.Buffer
		pm := pass.NewManager([]pass.Pass{
			&stubPass{id: "zzz"},
			&stubPass{id: "aaa"},
		})
		pm.AddInstrumentation(pass.NewPassTiming(&buf))
		if err := pm.Run(ir.NewGraph("t")); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "pass aaa:") || !strings.Contains(out, "pass zzz:") {
			t.Fatalf("timing report missing passes: %q", out)
		}
		if strings.Index(out, "pass aaa:") > strings.Index(out, "pass zzz:") {
			t.Errorf("timing report not in sorted ID order: %q", out)
		}
	})
}
