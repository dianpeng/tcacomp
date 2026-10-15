package ir_test

import (
	"testing"

	"tcacomp/internal/ir"
)

// TestConstructSurface is the cross-package contract test: everything waves
// 2-3 (pkg/ builder, pass tests, visualization tests) codes against must be
// reachable through the exported construction surface only.
func TestConstructSurface(t *testing.T) {
	build := func() (*ir.Graph, []uint64) {
		g := ir.NewGraph("surface")
		f32 := ir.InternScalar(ir.TF32)
		m := g.AddDimParam("m")
		lhs := g.AddTensorParam("lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
		scale := g.AddScalarParam("scale", f32)
		if m.Name() != "m" || lhs.Name() != "lhs" || scale.Name() != "scale" {
			t.Fatal("param names must round-trip through the exported surface")
		}
		if lhs.Type() != ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}) {
			t.Fatal("interned tensor types must be pointer-identical (D-18)")
		}

		c1 := g.NewConst("ctor", f32, 1.0)
		c2 := g.NewNode(ir.KindConst, "ctor") // raw low-level constructor: any kind
		if c2.KindOf() != ir.KindConst {
			t.Fatalf("NewNode kind = %v, want const", c2.KindOf())
		}
		sum := g.NewBinOp("ctor", ir.BinOpAdd, c1, c2)
		stmt := g.AddStatement(sum)

		ids := []uint64{c1.BasePtr().ID(), c2.BasePtr().ID(), sum.BasePtr().ID(), stmt.ID()}
		return g, ids
	}

	g1, ids1 := build()
	_, ids2 := build()

	want := []uint64{1, 2, 3, 4}
	for i := range want {
		if ids1[i] != want[i] || ids2[i] != want[i] {
			t.Fatalf("node IDs = %v / %v, want %v per graph", ids1, ids2, want)
		}
	}

	stmts := g1.Block().Statements()
	if len(stmts) != 1 {
		t.Fatalf("surface graph has %d statements, want 1", len(stmts))
	}
	if stmts[0].ID() != ids1[3] {
		t.Fatal("statement ID must come from the same per-graph monotonic counter")
	}
	if stmts[0].RHS().BasePtr().ID() != ids1[2] {
		t.Fatal("statement RHS must be the constructed binop")
	}
	if g1.Block() != g1.Block() {
		t.Fatal("graph must expose exactly one basic block (IR-02)")
	}
}
