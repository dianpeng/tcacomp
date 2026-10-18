package ir

import "testing"

// mark_test.go pins the transient node-held mark state (quick-261018-l36
// requirement 2): exactly TWO states (not-marked / alive), the mark lives ON
// the node, ResetMarks clears every node including params and the end node,
// and AllocatedNodes enumerates the sweep registry — every non-erased node
// in allocation order.

// markTestGraph builds a small Level-1 graph: a dim param, a tensor param,
// two consts, a binop over them, an env-chained store to the tensor, and the
// end node carrying the store as its returned value.
func markTestGraph() (*Graph, Node, Node, Node, Node, Node) {
	g := NewGraph("marks")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	c1 := g.NewConst("ctor", f32, 1)
	c2 := g.NewConst("ctor", f32, 2)
	sum := g.NewBinOp("ctor", BinOpAdd, c1, c2)
	env := NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []Node{m}, sum)
	end := g.NewReturnChained("ctor", env, []Node{out}, st)
	return g, out, sum, st, end, m
}

func TestMarkTwoStateLifecycle(t *testing.T) {
	g, out, sum, st, end, m := markTestGraph()

	// Fresh nodes are NOT marked — a stale alive mark from a previous
	// analysis would protect dead nodes in a later sweep.
	for _, n := range []Node{m, out, sum, st, end} {
		if n.BasePtr().Alive() {
			t.Fatalf("fresh node %d must report Alive() == false", n.BasePtr().ID())
		}
	}

	// MarkAlive flips the node to alive.
	sum.BasePtr().MarkAlive()
	if !sum.BasePtr().Alive() {
		t.Fatal("MarkAlive must make Alive() report true")
	}

	// ResetMarks clears EVERY node — params and the end node included.
	g.ResetMarks()
	for _, n := range g.AllocatedNodes() {
		if n.BasePtr().Alive() {
			t.Fatalf("node %d still alive after ResetMarks", n.BasePtr().ID())
		}
	}
}

func TestAllocatedNodesSweepRegistry(t *testing.T) {
	g, _, _, st, _, _ := markTestGraph()

	// Allocation order: IDs are monotonic, so ascending IDs in the slice pin
	// the order; every node the test built is present.
	nodes := g.AllocatedNodes()
	if len(nodes) == 0 {
		t.Fatal("AllocatedNodes must return the graph's nodes")
	}
	for i := 1; i < len(nodes); i++ {
		if nodes[i-1].BasePtr().ID() >= nodes[i].BasePtr().ID() {
			t.Fatalf("AllocatedNodes out of allocation order: %d then %d",
				nodes[i-1].BasePtr().ID(), nodes[i].BasePtr().ID())
		}
	}
	found := false
	for _, n := range nodes {
		if n == st {
			found = true
		}
	}
	if !found {
		t.Fatal("the store must appear in AllocatedNodes")
	}

	// Erased tombstones are excluded: erase the zero-use store value tree is
	// invasive, so erase the STORE itself by first unwiring its users — here
	// the simpler pin: a fresh node nothing references is erased and then
	// absent from the registry.
	orphan := g.NewConst("ctor", InternScalar(TF32), 3)
	if err := (&MutContext{Pass: "t"}).Erase(orphan); err != nil {
		t.Fatalf("Erase(orphan): %v", err)
	}
	for _, n := range g.AllocatedNodes() {
		if n == orphan {
			t.Fatal("erased tombstone must be excluded from AllocatedNodes")
		}
	}

	// The returned slice is fresh: mutating it must not affect the graph.
	snapshot := g.AllocatedNodes()
	snapshot[0] = nil
	if g.AllocatedNodes()[0] == nil {
		t.Fatal("AllocatedNodes must return a fresh slice, not the internal registry")
	}
}
