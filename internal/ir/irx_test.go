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
		m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
		lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
		scale := g.NewInputParam("ctor", ir.RoleScalar, "scale", f32)
		if len(g.Params()) != 3 {
			t.Fatal("kernel params must be graph-owned InputParam nodes")
		}

		c1 := g.NewConst("ctor", f32, 1.0)
		c2 := g.NewNode(ir.KindConst, "ctor") // raw low-level constructor: any kind
		if c2.KindOf() != ir.KindConst {
			t.Fatalf("NewNode kind = %v, want const", c2.KindOf())
		}
		sum := g.NewBinOp("ctor", ir.BinOpAdd, c1, c2)
		stmt := g.AddStatement(sum)

		// InputParams are real nodes with real IDs, allocated from the same
		// monotonic counter (IR-05).
		ids := []uint64{m.BasePtr().ID(), lhs.BasePtr().ID(), scale.BasePtr().ID(),
			c1.BasePtr().ID(), c2.BasePtr().ID(), sum.BasePtr().ID(), stmt.ID()}
		return g, ids
	}

	g1, ids1 := build()
	_, ids2 := build()

	want := []uint64{1, 2, 3, 4, 5, 6, 7}
	for i := range want {
		if ids1[i] != want[i] || ids2[i] != want[i] {
			t.Fatalf("node IDs = %v / %v, want %v per graph", ids1, ids2, want)
		}
	}

	if g1.Block().First() != g1.Block().Last() {
		t.Fatal("surface graph must hold exactly one statement")
	}
	if g1.Block().First().ID() != ids1[6] {
		t.Fatal("statement ID must come from the same per-graph monotonic counter")
	}
	if g1.Block().First().RHS().BasePtr().ID() != ids1[5] {
		t.Fatal("statement RHS must be the constructed binop")
	}
	if g1.Block() != g1.Block() {
		t.Fatal("graph must expose exactly one basic block (IR-02)")
	}
}
