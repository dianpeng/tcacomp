package pkg_test

import (
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/pkg"
)

// oracleDims builds the three dynamic dims shared by the eager-check tests.
func oracleDims(b *pkg.Builder) (m, k, n pkg.Dim) {
	return b.Dim("m", 0), b.Dim("k", 0), b.Dim("n", 0)
}

// TestBuilderEagerChecks (D-10) pins the cheap invariant checks that fire at
// the builder call sites, long before the full verifier sees the graph.
func TestBuilderEagerChecks(t *testing.T) {
	t.Run("wrongMapArity", func(t *testing.T) {
		b := pkg.NewBuilder("arity")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		rhs := b.Tensor("rhs", ir.TF32, n, k)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n), pkg.At(rhs, n, k)},
				func(l pkg.Expr) pkg.Expr { return l }) // 1 param for 2 operands
		if err == nil || !strings.Contains(err.Error(), "arity") {
			t.Fatalf("wrong-arity root map: err = %v, want arity error", err)
		}
	})

	t.Run("wrongPipedMapArity", func(t *testing.T) {
		b := pkg.NewBuilder("arity-piped")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(v pkg.Expr) pkg.Expr { return v }).
			As(m, k, n).
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
			Map(nil, func(a, b pkg.Expr) pkg.Expr { return a }) // 2 params for pipe only
		if err == nil || !strings.Contains(err.Error(), "arity") {
			t.Fatalf("wrong-arity piped map: err = %v, want arity error", err)
		}
	})

	t.Run("duplicateAxisInAs", func(t *testing.T) {
		b := pkg.NewBuilder("dup-as")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		rhs := b.Tensor("rhs", ir.TF32, n, k)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n), pkg.At(rhs, n, k)},
				func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) }).
			As(m, k, k). // @k listed twice, @n omitted
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
		if err == nil || !strings.Contains(err.Error(), "@n") {
			t.Fatalf("duplicate axis in as: err = %v, want live-axis coverage error", err)
		}
	})

	t.Run("omitLiveAxisInAs", func(t *testing.T) {
		b := pkg.NewBuilder("omit-as")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(l pkg.Expr) pkg.Expr { return l }).
			As(m). // @k and @n omitted
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
		if err == nil {
			t.Fatal("omitted live axis in as: nil error")
		}
	})

	t.Run("heterogeneousReducer", func(t *testing.T) {
		b := pkg.NewBuilder("hetero-reduce")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(l pkg.Expr) pkg.Expr { return l }).
			As(m, k, n).
			Reduce(b.Int32(0), // i32 seed over an f32 pipe: homogeneous-reducer violation
				func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
		if err == nil || !strings.Contains(err.Error(), "homogeneous") {
			t.Fatalf("heterogeneous reducer: err = %v, want homogeneous-reducer error", err)
		}
	})

	t.Run("wrongSubscriptCount", func(t *testing.T) {
		b := pkg.NewBuilder("subscripts")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		rhs := b.Tensor("rhs", ir.TF32, n, k)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m), pkg.At(rhs, n, k)}, // lhs rank 2, one axis
				func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) })
		if err == nil || !strings.Contains(err.Error(), "rank") {
			t.Fatalf("wrong subscript count: err = %v, want rank error", err)
		}
	})

	t.Run("undeclaredDim", func(t *testing.T) {
		b := pkg.NewBuilder("undeclared")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		rhs := b.Tensor("rhs", ir.TF32, n, k)
		foreign := pkg.NewBuilder("other").Dim("z", 0) // dim from another builder
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, foreign), pkg.At(rhs, n, k)},
				func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) })
		if err == nil || !strings.Contains(err.Error(), "declared") {
			t.Fatalf("undeclared dim: err = %v, want declared-dim error", err)
		}
	})

	t.Run("unknownBuiltin", func(t *testing.T) {
		b := pkg.NewBuilder("builtin")
		if _, err := pkg.Call("threadIdx_x"); err == nil ||
			!strings.Contains(err.Error(), "not a registered pure scalar builtin") {
			t.Fatalf("unknown builtin: err = %v, want registry error", err)
		}
	})

	t.Run("badOperandAxis", func(t *testing.T) {
		b := pkg.NewBuilder("operand-axis")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		// @k is not a dim of lhs; structural indexing binds subscript i to dim i.
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, k)},
				func(l pkg.Expr) pkg.Expr { return l })
		if err == nil || !strings.Contains(err.Error(), "dim") {
			t.Fatalf("operand axis mismatch: err = %v, want dim error", err)
		}
	})
}

// TestBuilderTypestate (D-07, compile-time half) exercises every legal rung
// sequence; the chain TYPES enforce the pipe order — after Reduce only
// Reduce/Map/Store exist, after Store the value is terminal — so the illegal
// orders cannot even be written down (they fail `go build`, which the oracle
// test in Task 2 proves by compiling).
func TestBuilderTypestate(t *testing.T) {
	t.Run("reduceThenStore", func(t *testing.T) {
		b := pkg.NewBuilder("rs")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		out := b.Tensor("output", ir.TF32, m, k)
		stored, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(l pkg.Expr) pkg.Expr { return l }).
			As(m, k, n).
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
			Store(out, m, k)
		if err != nil {
			t.Fatalf("reduce->store chain: %v", err)
		}
		if stored == nil {
			t.Fatal("Store returned nil *Stored")
		}
	})

	t.Run("chainedReductions", func(t *testing.T) {
		b := pkg.NewBuilder("chain-red")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		out := b.Tensor("output", ir.TF32, m)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n)},
				func(l pkg.Expr) pkg.Expr { return l }).
			As(m, k, n).
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
			Store(out, m)
		if err != nil {
			t.Fatalf("chained reductions: %v", err)
		}
	})

	t.Run("fullPipe", func(t *testing.T) {
		b := pkg.NewBuilder("full-pipe")
		m, k, n := oracleDims(b)
		lhs := b.Tensor("lhs", ir.TF32, m, n)
		rhs := b.Tensor("rhs", ir.TF32, n, k)
		out := b.Tensor("output", ir.TF32, m, k)
		scale := b.ScalarParam("scale", ir.TF32)
		_, err := b.Stmt().Iter(m, k, n).
			Map([]pkg.Operand{pkg.At(lhs, m, n), pkg.At(rhs, n, k)},
				func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) }).
			As(m, k, n).
			Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
			Map(nil, func(v pkg.Expr) pkg.Expr { return b.Mul(v, scale) }).
			As(m, k).
			Store(out, m, k)
		if err != nil {
			t.Fatalf("full pipe: %v", err)
		}
	})
}
