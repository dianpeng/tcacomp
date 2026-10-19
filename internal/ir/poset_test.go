package ir

import "testing"

// posetGraph builds the canonical R1/R2 graph these tests pin: two tensors
// with env-chained stores and ONE end node carrying the last store as its
// returned value and both tensors' final chain states via ImpactBy.
//
//	nodes (allocation order): m=1, out1=2, out2=3, a=4, b=5, sum=6,
//	st1=7, st2=8, ret=9
func posetGraph(t *testing.T) (g *Graph, env *Env, m, out1, out2, sum, st1, st2, ret Node) {
	t.Helper()
	g = NewGraph("poset")
	f32 := InternScalar(TF32)
	m = g.NewInputParam("ctor", RoleDim, "m", nil)
	out1 = g.NewInputParam("ctor", RoleTensor, "out1", InternTensor(TF32, []DimRef{{Name: "m"}}))
	out2 = g.NewInputParam("ctor", RoleTensor, "out2", InternTensor(TF32, []DimRef{{Name: "m"}}))
	a := g.NewConst("ctor", f32, 1)
	b := g.NewConst("ctor", f32, 2)
	sum = g.NewBinOp("ctor", BinOpAdd, a, b)
	env = NewEnv()
	st1 = g.NewMemStoreChained("ctor", env, out1, []Node{m}, sum)
	st2 = g.NewMemStoreChained("ctor", env, out2, []Node{m}, sum)
	ret = g.NewReturnChained("ctor", env, []Node{out1, out2}, st2)
	return g, env, m, out1, out2, sum, st1, st2, ret
}

// TestDataDepsPinsOperandOrder: data dependencies are the LIVE operand-slot
// defs in slot order — the poset edge set the scheduler's topo sort and
// LICM treat as "must come after" (R1). A store's target access rides its
// operand slots (Input/ImpactBy-only convention), so it IS a data dep;
// ImpactBy edges stay on their own face.
func TestDataDepsPinsOperandOrder(t *testing.T) {
	_, _, _, _, _, sum, st1, st2, ret := posetGraph(t)

	if got := DataDeps(sum); len(got) != 2 || got[0].BasePtr().ID() != 4 || got[1].BasePtr().ID() != 5 {
		t.Fatalf("DataDeps(sum) = %v, want the two consts in slot order", got)
	}
	// A store's data deps are [value, tensor, axes...]: the stored value,
	// then the target-access components in slot order.
	if got := DataDeps(st1); len(got) != 3 || got[0] != sum {
		t.Fatalf("DataDeps(st1) = %v, want [sum, tensor, axis] (value then target access)", got)
	}
	if got := DataDeps(ret); len(got) != 1 || got[0] != st2 {
		t.Fatalf("DataDeps(end) = %v, want [st2] (the returned value; ImpactBy is separate)", got)
	}
}

// TestEffectDepsPinsImpactByEdges: effect dependencies are the LIVE
// ImpactBy payload defs — the explicit sea-of-nodes ordering edges.
func TestEffectDepsPinsImpactByEdges(t *testing.T) {
	_, _, _, _, _, _, st1, st2, ret := posetGraph(t)

	if got := EffectDeps(st1); len(got) != 0 {
		t.Fatalf("first store is chain-head-free: EffectDeps = %v, want empty", got)
	}
	if got := EffectDeps(ret); len(got) != 2 || got[0] != st1 || got[1] != st2 {
		t.Fatalf("EffectDeps(end) = %v, want [st1 st2] (final chain states in tensor order)", got)
	}
	// Effect-free nodes carry no effect deps.
	if got := EffectDeps(st2); len(got) != 0 {
		t.Fatalf("st2 heads a fresh chain: EffectDeps = %v, want empty", got)
	}
}

// TestPosetDepsDataThenEffect: the combined dependency list is data edges
// (slot order) followed by effect edges — one flat "must come after" list.
func TestPosetDepsDataThenEffect(t *testing.T) {
	_, _, _, _, _, _, st1, st2, ret := posetGraph(t)

	got := PosetDeps(ret)
	if len(got) != 3 || got[0] != st2 || got[1] != st1 || got[2] != st2 {
		t.Fatalf("PosetDeps(end) = %v, want [st2(data) st1 st2(effect)]", got)
	}
}

// TestSeaRootsTwoRootSet pins R3's root set: the RoleTensor params (graph
// param declaration order — dims and scalars excluded) followed by the end
// node. Stores are NEVER roots; the end node's ImpactBy path is what keeps
// them alive.
func TestSeaRootsTwoRootSet(t *testing.T) {
	g, _, _, out1, out2, _, _, _, ret := posetGraph(t)

	roots := g.SeaRoots()
	if len(roots) != 3 || roots[0] != out1 || roots[1] != out2 || roots[2] != ret {
		t.Fatalf("SeaRoots() = %v, want [out1 out2 end] (RoleTensor params, then the end node)", roots)
	}

	// A graph with no end node roots at its tensor params only.
	empty := NewGraph("no-end")
	empty.NewInputParam("ctor", RoleScalar, "s", InternScalar(TF32))
	tn := empty.NewInputParam("ctor", RoleTensor, "t", InternTensor(TF32, []DimRef{{Name: "m"}}))
	if roots := empty.SeaRoots(); len(roots) != 1 || roots[0] != tn {
		t.Fatalf("SeaRoots() without an end node = %v, want [t] (scalar params are not roots)", roots)
	}
}

// TestPosetDepsKilledRecordsSkipped: killed use records contribute no
// dependency edges (the handle model's corruption class is checkable, not
// traversable).
func TestPosetDepsKilledRecordsSkipped(t *testing.T) {
	_, _, _, _, _, sum, _, _, _ := posetGraph(t)
	sum.(*binOpNode).ops[0].rec.killed = true // white-box corruption
	if got := DataDeps(sum); len(got) != 1 {
		t.Fatalf("killed records must contribute no edges, got %v", got)
	}
}
