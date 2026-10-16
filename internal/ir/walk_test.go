package ir

import "testing"

// buildWalkGraph builds a fixed two-statement graph with a shared def and a
// map whose lambda body references an earlier def:
//
//	a = 1.0, b = 2.0
//	statement #4:  sum = a + b
//	statement #8:  map(b, lambda(v): v * a) as (@m)
//
// Pre-order DFS from the block must visit: sum, a, b, map, body, paramRef
// (statement order roots; operand-slot order; lambda bodies last).
func buildWalkGraph() *Graph {
	g := NewGraph("walk")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)
	m := g.AddDimParam("m")
	v := NewLambdaParam("v", f32)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), a)
	rootMap := g.NewMap("ctor", []Node{b}, Lambda{Params: []Param{v}, Body: body}, []Param{m})
	g.AddStatement(rootMap)
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
// reachable node exactly once, roots in BB statement order, operand-slot
// order, lambda bodies last — and two walks over the same graph yield
// identical order.
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
	want := []uint64{3, 1, 2, 7, 6, 5} // sum, a, b, map, body, paramRef
	if !equalIDs(first, want) {
		t.Fatalf("walk order = %v, want pre-order %v", first, want)
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
	want := []uint64{3, 1, 2, 7} // everything up to and including the map
	if got := walkIDs(order); !equalIDs(got, want) {
		t.Fatalf("Stop must terminate at the map, visited %v, want %v", got, want)
	}

	// Stop on the very first node visits exactly one node.
	order = nil
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		order = append(order, n)
		return Stop
	}))
	if got := walkIDs(order); !equalIDs(got, []uint64{3}) {
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
		if n.KindOf() == KindBinOp && n.BasePtr().ID() == 3 {
			return SkipChildren // prune sum's a/b subtree
		}
		return Descend
	}))
	want := []uint64{3, 7, 6, 5} // sum, map, body, paramRef — a and b unreachable now
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
	g.AddStatement(b)
	c := g.NewBinOp("ctor", BinOpMul, b, a)
	g.AddStatement(c)

	// White-box cycle: b's first operand now points back at c.
	b.(*binOpNode).ops[0].rec.def = c.BasePtr().ssa

	count := 0
	Walk(g, VisitorFunc(func(n Node) VisitResult {
		count++
		return Descend
	}))
	if count != 3 {
		t.Fatalf("cyclic walk visited %d nodes, want exactly 3 (each once)", count)
	}
}
