package pkg_test

import (
	"io"
	"testing"

	"tcacomp/internal/golden"
	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
	"tcacomp/pkg"
)

// buildMatmulScale constructs the doc/tca.md §5.1 oracle through the public
// builder (FE-01). The rung sequence is exactly the plan's pinned chain; it
// is unrolled into per-rung statements because Go forbids chaining a method
// call off a (rung, error) return. Shared with the determinism test.
func buildMatmulScale(t *testing.T) *ir.Graph {
	t.Helper()
	b := pkg.NewBuilder("matmul_scale")
	m := b.Dim("m", 0)
	k := b.Dim("k", 0)
	n := b.Dim("n", 0)
	lhs := b.Tensor("lhs", ir.TF32, m, n)
	rhs := b.Tensor("rhs", ir.TF32, n, k)
	out := b.Tensor("output", ir.TF32, m, k)
	scale := b.ScalarParam("scale", ir.TF32)

	root, err := b.Stmt().Iter(m, k, n).
		Map([]pkg.Operand{pkg.At(lhs, m, n), pkg.At(rhs, n, k)},
			func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) })
	if err != nil {
		t.Fatalf("root map: %v", err)
	}
	mid := root.As(m, k, n)
	ar, err := mid.Reduce(b.Float32(0.0),
		func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
	if err != nil {
		t.Fatalf("reduce: %v", err)
	}
	am, err := ar.Map(nil, func(v pkg.Expr) pkg.Expr { return b.Mul(v, scale) })
	if err != nil {
		t.Fatalf("piped map: %v", err)
	}
	tail := am.As(m, k)
	if _, err := tail.Store(out, m, k); err != nil {
		t.Fatalf("store: %v", err)
	}

	graph, err := b.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return graph
}

// TestMatmulScaleGolden (FE-01 + TOOL-05) is the phase's flagship gate: the
// programmatically constructed oracle must byte-match the doc-pinned golden
// committed by plan 01-03. The golden is never regenerated (-update is for
// derived goldens only); on mismatch the diff is the spec.
func TestMatmulScaleGolden(t *testing.T) {
	graph := buildMatmulScale(t)
	golden.Assert(t, "testdata/matmul_scale.ir", []byte(ir.Print(graph)))
}

// TestBuildVerifiesThroughPM (D-10) proves Build's verification flows
// exclusively through the pass-manager path and can actually fail:
//
//   - (a) Build() runs the completed oracle through pass.NewManager(nil) —
//     an EMPTY pass list — carrying the VerifyEach instrumentation, and
//     returns nil: verification executed because VerifyEach.AfterPipeline
//     runs a final verification pass even with zero passes and Manager.Run
//     returns its error.
//   - (b) the same PM path over a hand-built graph with an unwired operand
//     (which the ir constructors deliberately allow) returns a non-nil
//     error, so the single authoritative path is non-vacuous.
func TestBuildVerifiesThroughPM(t *testing.T) {
	t.Run("oracleBuildVerifies", func(t *testing.T) {
		b := pkg.NewBuilder("pm-path")
		m := b.Dim("m", 0)
		n := b.Dim("n", 0)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		out := b.Tensor("output", ir.TF32, m) // reduce over the last as axis @n leaves @m live
		root, err := b.Stmt().Iter(m, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(l pkg.Expr) pkg.Expr { return l })
		if err != nil {
			t.Fatalf("setup map: %v", err)
		}
		mid := root.As(m, n)
		ar, err := mid.Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
		if err != nil {
			t.Fatalf("setup reduce: %v", err)
		}
		if _, err := ar.Store(out, m); err != nil {
			t.Fatalf("setup store: %v", err)
		}
		graph, err := b.Build()
		if err != nil {
			t.Fatalf("Build through the empty-pipeline PM path: %v", err)
		}
		if graph == nil {
			t.Fatal("Build returned a nil graph")
		}
	})

	t.Run("unwiredOperandFailsPMVerification", func(t *testing.T) {
		bg := ir.NewGraph("bad")
		bg.AddStatement(bg.NewNode(ir.KindBinOp, "test", nil, nil))
		pm := pass.NewManager(nil)
		pm.AddInstrumentation(pass.NewVerifyEach(io.Discard))
		if err := pm.Run(bg); err == nil {
			t.Fatal("unwired operand through the PM path: nil error, want verification failure")
		}
	})
}
