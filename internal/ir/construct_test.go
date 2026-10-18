package ir

import "testing"

// newTestGraph builds the canonical small graph used across the white-box
// tests: c1 = 1.0, c2 = 2.0, sum = c1 + c2, anchored by `return sum` so the
// whole tree is reachable from the Level-1 effect roots.
func newTestGraph(t *testing.T) (*Graph, Node, Node, Node) {
	t.Helper()
	g := NewGraph("test")
	f32 := InternScalar(TF32)
	c1 := g.NewConst("ctor", f32, 1.0)
	c2 := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, c1, c2)
	g.NewReturn("ctor", sum)
	return g, c1, c2, sum
}

func TestSingleBlock(t *testing.T) {
	g := NewGraph("single-block")

	// Level 1: no block, no CFG (T-quick-02 nil-safety).
	if g.Block() != nil || g.CFG() != nil {
		t.Fatal("a fresh graph must have no block and no CFG before Materialize")
	}

	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	add := g.NewBinOp("ctor", BinOpAdd, a, b)
	c := g.NewConst("ctor", f32, 3.0)
	mul := g.NewBinOp("ctor", BinOpMul, add, c)
	ret := g.NewReturn("ctor", mul)

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g.Block() == nil || g.Block() != g.Block() {
		t.Fatal("after Materialize the graph must hold exactly one stable Block")
	}

	// The block stores statements in an intrusive list: one statement (the
	// return — the only effect root), and mul/add reachable in its tree.
	if g.Block().First() != g.Block().Last() {
		t.Fatal("one effect root must materialize to exactly one statement")
	}
	if g.Block().First().Prev() != nil || g.Block().First().Next() != nil {
		t.Fatal("list boundaries must read nil")
	}
	if g.Block().First().RHS() != Node(ret) {
		t.Fatalf("statement RHS = %v, want the return node", g.Block().First().RHS())
	}
	seen := map[Node]bool{}
	for _, n := range StmtNodes(g.Block().First()) {
		seen[n] = true
	}
	if !seen[add] || !seen[mul] || !seen[a] || !seen[b] || !seen[c] {
		t.Fatal("the whole operand tree must render inside the return's statement")
	}
}

func TestNodeIdentity(t *testing.T) {
	// Two structurally identical graphs built through identical call
	// sequences must have identical per-graph monotonic ID sequences
	// (IR-05): IDs are per-graph, start at 1, and are never reused. The
	// statement ID comes from the same counter at Materialize time (D-04).
	build := func() []uint64 {
		g := NewGraph("identity")
		f32 := InternScalar(TF32)
		c1 := g.NewConst("ctor", f32, 1.0)
		c2 := g.NewConst("ctor", f32, 2.0)
		sum := g.NewBinOp("ctor", BinOpAdd, c1, c2)
		ret := g.NewReturn("ctor", sum)
		if err := g.Materialize(); err != nil {
			t.Fatalf("Materialize: %v", err)
		}
		return []uint64{c1.BasePtr().ID(), c2.BasePtr().ID(), sum.BasePtr().ID(), ret.BasePtr().ID(), g.Block().First().ID()}
	}

	ids1, ids2 := build(), build()
	if len(ids1) != 5 {
		t.Fatalf("expected 5 IDs, got %d", len(ids1))
	}
	want := []uint64{1, 2, 3, 4, 5}
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
	_, c1, c2, sum := newTestGraph(t)

	uses := UsesOf(c1)
	if len(uses) != 1 {
		t.Fatalf("c1 has %d uses, want 1", len(uses))
	}
	if DefOf(uses[0]) != Node(c1) {
		t.Fatal("DefOf must return the def a use points at")
	}
	// Two operand slots on one user: both link into the def vars' ref lists
	// in append (link) order.
	bin := sum.(*binOpNode)
	if len(bin.ops) != 2 {
		t.Fatalf("binop has %d operand slots, want 2", len(bin.ops))
	}
	if u := UsesOf(c1); u[0] != bin.ops[0] {
		t.Fatal("c1's ref list head must be the binop's first operand slot")
	}
	if u := UsesOf(c2); u[0] != bin.ops[1] {
		t.Fatal("c2's ref list head must be the binop's second operand slot")
	}
	// The record's user back-pointer must name the owning node.
	if UsesOf(c1)[0].rec.user != Node(sum) {
		t.Fatal("use back-pointer must name the owning node")
	}

	// Snapshot safety: mutating the returned slice must not touch the chain.
	snap := UsesOf(c1)
	snap[0] = SSAUse{}
	if UsesOf(c1)[0] == (SSAUse{}) {
		t.Fatal("UsesOf must return a snapshot, not live list links")
	}
}

// TestStatementValueNone pins the Value interface semantics (directive 2):
// the return root is the none Value — a Value, but nothing — while every
// other node kind (store included, doc §2.5) produces a real value. Statement
// roots stay Node-typed: a plain constructor result is accepted without a
// cast, and Value() is the single classification accessor.
func TestStatementValueNone(t *testing.T) {
	g := NewGraph("value-none")
	f32 := InternScalar(TF32)
	v := g.NewConst("ctor", f32, 1.0)

	// A store root produces the stored value (doc §2.5): NOT a none Value.
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	st := g.NewMemStore("ctor", out, []Node{m}, v)
	ret := g.NewReturn("ctor", st)

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	sStore := g.Block().First()
	if sStore.Value().IsNone() {
		t.Fatal("store root must produce a real Value (it returns the stored value, doc §2.5)")
	}
	if Node(sStore.Value()) != sStore.RHS() {
		t.Fatal("Value() viewed as Node must equal RHS()")
	}

	// The return root is the none Value: it satisfies Value so it can sit in
	// a statement RHS slot, and IsNone() reports the none semantics.
	sRet := g.Block().Last()
	if !sRet.Value().IsNone() {
		t.Fatal("return root must report IsNone() == true (the none Value)")
	}
	if Node(sRet.Value()) != sRet.RHS() || sRet.RHS() != ret {
		t.Fatal("Value() viewed as Node must equal RHS()")
	}

	// A plain constructor Node result classifies as a real value.
	if classifyStmt(g.NewBinOp("ctor", BinOpAdd, v, v)) != StmtAssign {
		t.Fatal("binop root must produce a real Value")
	}
}
