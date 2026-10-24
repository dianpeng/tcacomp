package ir

// loopspec_test.go pins the FamilyLoop category spec (quick-261022-f5o,
// D-01): loop-header membership (KindIter and KindForLoop only —
// IterCluster/DimVar/BoundedDimVar are never loops), and the iter
// dialect's ParentEdge (first loop-kind operand in DataDeps slot order).
// ForLoop has no body operand (sea-of-nodes ruling, fix1.md §0); LIR
// nesting is derived from IV data-dep references (fix1.md D-b).

import "testing"

// specFixture builds a graph with one dim, one tensor, the u32/f32 types,
// and a normalized dimvar pair — the minimal environment for both dialects.
func specFixture(name string) (g *Graph, dim, out, dv, biv Node) {
	g = NewGraph(name)
	u32 := InternScalar(TU32)
	dim = g.NewInputParam("ctor", RoleDim, "d0", nil)
	out = g.NewInputParam("ctor", RoleTensor, "out",
		InternTensor(TF32, []DimRef{{Name: "d0"}}))
	dv = g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	biv = g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, 8))
	return g, dim, out, dv, biv
}

// TestLoopSpecIsLoop pins membership: KindIter and KindForLoop pass;
// IterCluster, DimVar, BoundedDimVar, consts, and params never do.
func TestLoopSpecIsLoop(t *testing.T) {
	g, dim, out, dv, biv := specFixture("isloop")
	iter := g.NewIter("ctor", biv, nil)
	loop := g.NewForLoop("ctor", dv, biv)
	cluster := g.NewIterCluster("ctor", []Node{dim}, g.NewConst("ctor", InternScalar(TF32), 0))
	c := g.NewConst("ctor", InternScalar(TF32), 1)

	for name, n := range map[string]Node{
		"iter": iter, "forloop": loop,
	} {
		if !LoopNesting.IsLoop(n) {
			t.Fatalf("%s must be a loop per the spec", name)
		}
	}
	for name, n := range map[string]Node{
		"itercluster": cluster, "dimvar": dv, "boundeddimvar": biv,
		"const": c, "dim-param": dim, "tensor-param": out,
	} {
		if LoopNesting.IsLoop(n) {
			t.Fatalf("%s must NOT be a loop per the spec", name)
		}
	}
	if LoopNesting.IsLoop(nil) {
		t.Fatal("nil must not be a loop (nil-safe read face)")
	}
}

// TestLoopSpecParentEdge pins the iter dialect's explicit nesting operand:
// a nested iter carries its parent iter in DataDeps, and the FIRST
// loop-kind dep wins; a forloop answers nil (containment dialect); a root
// iter (no loop operand) answers nil.
func TestLoopSpecParentEdge(t *testing.T) {
	g, _, _, dv0, biv0 := specFixture("parent")
	outer := g.NewIter("ctor", biv0, nil)
	inner := g.NewIter("ctor", biv0, outer) // ops = [biv, outer]
	if p := LoopNesting.ParentEdge(inner); p == nil || p.BasePtr() != outer.BasePtr() {
		t.Fatalf("nested iter ParentEdge = %v, want the parent iter", p)
	}
	if p := LoopNesting.ParentEdge(outer); p != nil {
		t.Fatalf("root iter ParentEdge = %v, want nil", p)
	}
	loop := g.NewForLoop("ctor", dv0, biv0)
	if p := LoopNesting.ParentEdge(loop); p != nil {
		t.Fatalf("forloop ParentEdge = %v, want nil (nesting via IV data-dep derivation, fix1.md D-b)", p)
	}
	if p := LoopNesting.ParentEdge(nil); p != nil {
		t.Fatal("nil ParentEdge must answer nil")
	}
}

// TestLoopSpecBodyRoot pins the forloop dialect's body operand: ops[2] of
// a KindForLoop; nil for iters (no body operand), nil for a corrupted
// short-operand forloop, nil for non-loop nodes.
//
// DEPRECATED: this test and BodyRoot will be removed in Step 2 of the
// fix1.md body-operand removal (the method only exists for Step 1 →
// Step 2 transition greenness).
func TestLoopSpecBodyRoot(t *testing.T) {
	g, _, _, dv, biv := specFixture("bodyroot")
	body := g.NewConst("ctor", InternScalar(TF32), 0)
	// Build a forloop manually with 3+ operands so BodyRoot still works
	// during the Step 1 transition (NewForLoop no longer takes a body).
	loop := g.allocNode(KindForLoop, "ctor", 3).(*forLoopNode)
	wireOperands(loop, dv, biv, body)
	if b := LoopNesting.BodyRoot(loop); b == nil || b.BasePtr() != body.BasePtr() {
		t.Fatalf("forloop BodyRoot = %v, want ops[2] (the body subtree root)", b)
	}
	iter := g.NewIter("ctor", biv, nil)
	if b := LoopNesting.BodyRoot(iter); b != nil {
		t.Fatalf("iter BodyRoot = %v, want nil (no body operand)", b)
	}
	// a corrupted short-operand forloop (fewer than 3 slots) via allocNode
	two := g.allocNode(KindForLoop, "ctor", 2).(*forLoopNode)
	wireOperands(two, dv, biv)
	if b := LoopNesting.BodyRoot(two); b != nil {
		t.Fatal("a forloop with fewer than 3 operands must answer nil, not panic")
	}
	if b := LoopNesting.BodyRoot(nil); b != nil {
		t.Fatal("nil BodyRoot must answer nil")
	}
}
