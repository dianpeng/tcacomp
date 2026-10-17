package ir

import "testing"

// buildWalkGraph builds a fixed graph with a shared def, a map whose lambda
// body references an earlier def, and TWO effect roots (returns anchoring the
// sum and the map — the sea has no statements, so effect roots are what keep
// both subtrees reachable):
//
//	a = 1.0, b = 2.0
//	return sum          (sum = a + b)
//	return map(b, lambda(v): v * a) as (@m)
//
// Pre-order DFS at Level 1 must visit: param m first, then the effect roots
// in registration order (sum's subtree, then the map), the map's operand b,
// the lambda param v, the body and its paramRef (operand-slot order, then
// payload uses, then lambda bodies; a and m are already seen by then).
func buildWalkGraph() *Graph {
	g := NewGraph("walk")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.NewReturn("ctor", sum)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	v := g.NewLambdaInputParam("ctor", "v", f32)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), a)
	rootMap := g.NewMap("ctor", []Node{b}, NewLambda([]Node{v}, body), []Node{m})
	g.NewReturn("ctor", rootMap)
	return g
}

func walkIDs(order []Node) []uint64 {
	ids := make([]uint64, len(order))
	for i, n := range order {
		ids[i] = n.BasePtr().ID()
	}
	return ids
}

func equalIDs(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWalkPreOrderDeterministic pins the D-5 contract: pre-order DFS, every
// reachable node exactly once, params first then effect-root order at Level
// 1, operand-slot order, lambda bodies last — and two walks over the same
// graph yield identical order. After Materialize the Level-2 roots (params,
// then statement order) must yield the SAME order here: the statements are
// the effect roots in the same order.
func TestWalkPreOrderDeterministic(t *testing.T) {
	g := buildWalkGraph()
	collect := func() []uint64 {
		var order []Node
		Walk(g, VisitorFunc(func(n Node) VisitResult {
			order = append(order, n)
			return Descend
		}))
		return walkIDs(order)
	}
	first := collect()
	second := collect()
	if !equalIDs(first, second) {
		t.Fatalf("walk order not deterministic: %v vs %v", first, second)
	}
	want := []uint64{5, 4, 3, 1, 2, 10, 9, 6, 8, 7} // m, retSum, sum, a, b, retMap, map, lambda param v, body, paramRef
	if !equalIDs(first, want) {
		t.Fatalf("walk order = %v, want pre-order %v", first, want)
	}

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if after := collect(); !equalIDs(after, want) {
		t.Fatalf("Level-2 walk order = %v, want the same pre-order %v", after, want)
	}
}

// TestWalkStopBailsEarly: Stop terminates the walk immediately — the
// remaining nodes are never visited.
func TestWalkStopBailsEarly(t *testing.T) {
	g := buildWalkGraph()
	var order []Node
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		order = append(order, n)
		if n.KindOf() == KindMap {
			return Stop
		}
		return Descend
	}))
	want := []uint64{5, 4, 3, 1, 2, 10, 9} // param m, sum's subtree, retMap, then Stop at the map
	if got := walkIDs(order); !equalIDs(got, want) {
		t.Fatalf("Stop must terminate at the map, visited %v, want %v", got, want)
	}

	// Stop on the very first node visits exactly one node.
	order = nil
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		order = append(order, n)
		return Stop
	}))
	if got := walkIDs(order); !equalIDs(got, []uint64{5}) {
		t.Fatalf("immediate Stop must visit exactly the first node, got %v", got)
	}
}

// TestWalkSkipChildrenPrunes: SkipChildren visits the node but none of its
// reachable descendants; siblings are unaffected.
func TestWalkSkipChildrenPrunes(t *testing.T) {
	g := buildWalkGraph()
	var order []Node
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		order = append(order, n)
		if n.KindOf() == KindBinOp && n.BasePtr().ID() == 8 {
			return SkipChildren // prune the lambda body's subtree
		}
		return Descend
	}))
	// The body is pruned, and paramRef — reachable ONLY through the body — is
	// never visited; a was already visited as sum's operand, so SkipChildren
	// prunes exactly the skipped edge, not the nodes globally.
	want := []uint64{5, 4, 3, 1, 2, 10, 9, 6, 8} // m, retSum, sum, a, b, retMap, map, v, body (paramRef pruned)
	if got := walkIDs(order); !equalIDs(got, want) {
		t.Fatalf("SkipChildren must prune the subtree, visited %v, want %v", got, want)
	}
}

// TestWalkCyclicSafe: a hand-built cyclic operand graph (white-box record
// rewiring) must walk without stack overflow and terminate, visiting each
// node exactly once (T-1.1-02).
func TestWalkCyclicSafe(t *testing.T) {
	g := NewGraph("cyclic")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewBinOp("ctor", BinOpAdd, a, a)
	c := g.NewBinOp("ctor", BinOpMul, b, a)
	g.NewReturn("ctor", c)

	// White-box cycle: b's first operand now points back at c.
	b.(*binOpNode).ops[0].rec.def = c.BasePtr().ssa

	count := 0
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		count++
		return Descend
	}))
	if count != 4 {
		t.Fatalf("cyclic walk visited %d nodes, want exactly 4 (return, c, b, a — each once)", count)
	}
}
