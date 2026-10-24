package pkg_test

import (
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/pkg"
)

// TestBuilderLocal (syntax-design.md §6 Task 4) pins the minimal pkg surface
// the frontend's `let`-local lowering needs: a KindMemNew-backed tensor
// handle that a chain can store into and a later chain can map over. The
// local is conceptually GMEM (doc/syntax.md); v0 codegen admits no live
// GMEM locals, so this test proves the graph-level capability only — Build
// must verify the graph clean.
func TestBuilderLocal(t *testing.T) {
	b := pkg.NewBuilder("local")
	m := b.Dim("m", 0)
	n := b.Dim("n", 0)
	lhs := b.Tensor("lhs", ir.TF32, m, n)
	out := b.Tensor("out", ir.TF32, m)

	// tmp[m] is a local tensor with a fully static shape (v0: dynamic local
	// extents are rejected at declaration).
	tmp := b.Local("tmp", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	// Chain 1: row sums of lhs land in the local: tmp[m] = Σ_n lhs[m][n].
	root, err := b.Stmt().Iter(m, n).
		Map([]pkg.Operand{pkg.At(lhs, m, n)}, func(v pkg.Expr) pkg.Expr { return v })
	if err != nil {
		t.Fatalf("root map: %v", err)
	}
	mid := root.As(m, n)
	ar, err := mid.Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
	if err != nil {
		t.Fatalf("reduce: %v", err)
	}
	if _, err := ar.Store(tmp, m); err != nil {
		t.Fatalf("store into local: %v", err)
	}

	// Chain 2 reads the local back as a map operand: out[m] = tmp[m] * 2.
	root2, err := b.Stmt().Iter(m, n).
		Map([]pkg.Operand{pkg.At(tmp, m)}, func(v pkg.Expr) pkg.Expr { return v })
	if err != nil {
		t.Fatalf("map over local: %v", err)
	}
	mid2 := root2.As(m, n)
	ar2, err := mid2.Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
	if err != nil {
		t.Fatalf("second reduce: %v", err)
	}
	am, err := ar2.Map(nil, func(v pkg.Expr) pkg.Expr { return b.Mul(v, b.Float32(2)) })
	if err != nil {
		t.Fatalf("piped map: %v", err)
	}
	tail := am.As(m)
	if _, err := tail.Store(out, m); err != nil {
		t.Fatalf("store out: %v", err)
	}

	// Build runs BuildingDCE + Materialize + the VerifyEach gate: the
	// MemNew-backed local (store target, operand paramRef) must verify clean.
	if _, err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
}

// TestBuilderLocalRejectsDynamicExtents: a local whose shape carries a
// dynamic dim has no constexpr GMEM extent in v0 — declaration records the
// violation, surfaced by the next error-returning call or Build.
func TestBuilderLocalRejectsDynamicExtents(t *testing.T) {
	b := pkg.NewBuilder("local-dyn")
	m := b.Dim("m", 0)
	n := b.Dim("n", 0)
	lhs := b.Tensor("lhs", ir.TF32, m, n)
	tmp := b.Local("tmp", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}, {Name: "n"}}))
	root, err := b.Stmt().Iter(m, n).
		Map([]pkg.Operand{pkg.At(lhs, m, n)}, func(v pkg.Expr) pkg.Expr { return v })
	if err != nil {
		t.Fatalf("root map: %v", err)
	}
	mid := root.As(m, n)
	if _, err := mid.Reduce(b.Float32(0), func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }); err == nil ||
		!strings.Contains(err.Error(), "dynamic") {
		t.Fatalf("dynamic-extent local: err = %v, want dynamic error", err)
	}
	_ = tmp
}
