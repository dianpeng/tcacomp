package ir

import (
	"testing"
)

// TestMemoryVerifyShapes pins the fail-closed shape checks of the HIR GMEM
// family (D-3): memload's tensor operand and memstore's target tensor bind a
// RoleTensor InputParam node.
func TestMemoryVerifyShapes(t *testing.T) {
	f32 := InternScalar(TF32)

	t.Run("memload rejects non-tensor operand", func(t *testing.T) {
		g := NewGraph("m1")
		ld := g.NewMemLoad("ctor", g.NewConst("ctor", f32, 0), nil)
		g.NewReturn("ctor", ld)
		assertViolation(t, g, "tensor operand must be a RoleTensor InputParam node")
	})

	t.Run("memstore rejects non-tensor target", func(t *testing.T) {
		g := NewGraph("m2")
		st := g.NewMemStore("ctor", g.NewConst("ctor", f32, 0), nil, g.NewConst("ctor", f32, 1))
		g.NewReturn("ctor", st)
		assertViolation(t, g, "tensor operand must be a RoleTensor InputParam node")
	})
}

// TestMemLoadChained pins the skeleton memory.go read-chaining rule: loads
// participate in the per-tensor GMEM effect chains exactly like stores — a
// chained load ImpactBy's the current chain head and becomes the new head,
// so a later store to the same tensor chains after the read.
func TestMemLoadChained(t *testing.T) {
	g := NewGraph("chain")
	f32 := InternScalar(TF32)
	dm := g.NewInputParam("ctor", RoleDim, "m", nil)
	tn := g.NewInputParam("ctor", RoleTensor, "t", InternTensor(TF32, []DimRef{{Name: "m"}}))
	env := NewEnv()

	st1 := g.NewMemStoreChained("ctor", env, tn, []Node{dm}, g.NewConst("ctor", f32, 1))
	ld := g.NewMemLoadChained("ctor", env, tn, []Node{dm})
	st2 := g.NewMemStoreChained("ctor", env, tn, []Node{dm}, g.NewConst("ctor", f32, 2))

	imp := ImpactBy(ld)
	if len(imp) != 1 || imp[0].Def() != st1 {
		t.Fatalf("chained load must ImpactBy the prior store head, got %v", imp)
	}
	if head := env.ChainHead(tn); head != st2 {
		t.Fatal("the second store must become the chain head")
	}
	imp2 := ImpactBy(st2)
	if len(imp2) != 1 || imp2[0].Def() != ld {
		t.Fatalf("the store after the load must ImpactBy the load, got %v", imp2)
	}

	// Chain isolation: a load to a distinct tensor stays un-chained.
	other := g.NewInputParam("ctor", RoleTensor, "u", InternTensor(TF32, []DimRef{{Name: "m"}}))
	ld2 := g.NewMemLoadChained("ctor", env, other, []Node{dm})
	if got := ImpactBy(ld2); len(got) != 0 {
		t.Fatalf("load to an untouched tensor must be un-chained, got %v", got)
	}

	// ImpactBy records ride the payload enumeration, so Verify covers them.
	g.NewReturn("ctor", st2)
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chained memory graph must verify clean: %v", vs)
	}
}
