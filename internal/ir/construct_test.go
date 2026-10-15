package ir

import "testing"

// newTestGraph builds the canonical small graph used across the white-box
// tests: c1 = 1.0, c2 = 2.0, sum = c1 + c2, one statement binding sum.
func newTestGraph(t *testing.T) (*Graph, Node, Node, Node, *Statement) {
	t.Helper()
	g := NewGraph("test")
	f32 := InternScalar(TF32)
	c1 := g.NewConst("ctor", f32, 1.0)
	c2 := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, c1, c2)
	stmt := g.AddStatement(sum)
	return g, c1, c2, sum, stmt
}

func TestSingleBlock(t *testing.T) {
	g := NewGraph("single-block")

	if g.Block() == nil {
		t.Fatal("new graph has no basic block")
	}
	if g.Block() != g.Block() {
		t.Fatal("graph must hold exactly one stable Block")
	}

	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	add := g.NewBinOp("ctor", BinOpAdd, a, b)
	s1 := g.AddStatement(add)

	c := g.NewConst("ctor", f32, 3.0)
	mul := g.NewBinOp("ctor", BinOpMul, add, c)
	s2 := g.AddStatement(mul)

	stmts := g.Block().Statements()
	if len(stmts) != 2 {
		t.Fatalf("expected 2 statements, got %d", len(stmts))
	}
	// Insertion order is iteration order (IR-02).
	if stmts[0] != s1 || stmts[1] != s2 {
		t.Fatal("statements do not iterate in insertion order")
	}
	// Each statement's RHS is its value-producing node.
	if stmts[0].RHS() != Node(add) {
		t.Fatalf("s1 RHS = %v, want the add node", stmts[0].RHS())
	}
	if stmts[1].RHS() != Node(mul) {
		t.Fatalf("s2 RHS = %v, want the mul node", stmts[1].RHS())
	}
}

func TestNodeIdentity(t *testing.T) {
	// Two structurally identical graphs built through identical call
	// sequences must have identical per-graph monotonic ID sequences
	// (IR-05): IDs are per-graph, start at 1, and are never reused.
	build := func() []uint64 {
		g := NewGraph("identity")
		f32 := InternScalar(TF32)
		c1 := g.NewConst("ctor", f32, 1.0)
		c2 := g.NewConst("ctor", f32, 2.0)
		sum := g.NewBinOp("ctor", BinOpAdd, c1, c2)
		stmt := g.AddStatement(sum)
		return []uint64{c1.ID(), c2.ID(), sum.ID(), stmt.ID()}
	}

	ids1, ids2 := build(), build()
	if len(ids1) != 4 {
		t.Fatalf("expected 4 IDs, got %d", len(ids1))
	}
	want := []uint64{1, 2, 3, 4}
	for i := range want {
		if ids1[i] != want[i] {
			t.Fatalf("ID sequence = %v, want %v", ids1, want)
		}
		if ids1[i] != ids2[i] {
			t.Fatalf("identical builds diverged: %v vs %v", ids1, ids2)
		}
	}
}

func TestUseListBasics(t *testing.T) {
	g, c1, c2, sum, _ := newTestGraph(t)

	uses := UsesOf(c1)
	if len(uses) != 1 {
		t.Fatalf("c1 has %d uses, want 1", len(uses))
	}
	if DefOf(uses[0]) != Node(c1) {
		t.Fatal("DefOf must return the def a use points at")
	}
	if uses[0].User != Node(sum) {
		t.Fatal("use back-pointer must name the owning node")
	}

	// Two operand slots on one user: both link into c-def lists, most
	// recently linked first (LLVM prepend discipline).
	bin := sum.(*binOpNode)
	if len(bin.ops) != 2 {
		t.Fatalf("binop has %d operand slots, want 2", len(bin.ops))
	}
	if u := UsesOf(c1); u[0] != &bin.ops[0] {
		t.Fatal("c1's use-list head must be the binop's first operand slot")
	}
	if u := UsesOf(c2); u[0] != &bin.ops[1] {
		t.Fatal("c2's use-list head must be the binop's second operand slot")
	}

	// Snapshot safety: mutating the returned slice must not touch the chain.
	snap := UsesOf(c1)
	snap[0] = nil
	if UsesOf(c1)[0] == nil {
		t.Fatal("UsesOf must return a snapshot, not live list links")
	}
}
