package ir

import (
	"strings"
	"testing"
)

func TestUseChain(t *testing.T) {
	g := NewGraph("chain")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)
	c := g.NewConst("ctor", f32, 3.0)

	ctx := &MutContext{Pass: "test-rauw"}

	// RAUW on the middle def: every use of b must move to c (IR-01).
	if err := ctx.ReplaceAllUsesWith(b, c); err != nil {
		t.Fatalf("RAUW: %v", err)
	}
	if got := UsesOf(b); len(got) != 0 {
		t.Fatalf("old def still has %d uses after RAUW", len(got))
	}
	uses := UsesOf(c)
	if len(uses) != 1 {
		t.Fatalf("new def has %d uses, want 1", len(uses))
	}
	if uses[0].Val != c || uses[0].User != sum {
		t.Fatal("RAUW must retarget the use records, not drop them")
	}
	if got := sum.(*binOpNode).ops[1].Val; got != c {
		t.Fatal("the user's operand slot must see the new def")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chains inconsistent after RAUW: %v", vs)
	}

	// Erase the old def: uses gone, ID tombstoned.
	if err := ctx.Erase(b); err != nil {
		t.Fatalf("Erase: %v", err)
	}
	if got := UsesOf(b); len(got) != 0 {
		t.Fatalf("erased def still has %d uses", len(got))
	}
	if !b.BasePtr().erased {
		t.Fatal("Erase must tombstone the node's ID")
	}
	if next := g.NewID(); next <= b.BasePtr().ID() {
		t.Fatal("IDs must never be reused after Erase (IR-05)")
	}

	// A lingering Use hand-pointed at the tombstoned node is flagged on
	// re-verify — the exact corruption class Erase protects against.
	d := g.NewConst("ctor", f32, 4.0)
	y := g.NewBinOp("ctor", BinOpMul, a, d)
	g.AddStatement(y)
	yn := y.(*binOpNode)
	yn.ops[1].Val = b // white-box: no relink, as if a pass leaked a pointer

	vs := Verify(g)
	found := false
	for _, v := range vs {
		if v.NodeID == y.BasePtr().ID() && strings.Contains(v.Msg, "erased") {
			found = true
		}
	}
	if !found {
		t.Fatalf("re-verify must flag the lingering use of erased node %d, got %v", b.BasePtr().ID(), vs)
	}
}

// TestAddUseKeepsChainsConsistent exercises AddUse on a node whose ops slice
// is exactly at capacity after construction (len 2, cap 2): the append inside
// addSlot reallocates, and every *Use handed out earlier — embedded in the
// defs' use-lists and held by callers — must stay valid (CR-01).
func TestAddUseKeepsChainsConsistent(t *testing.T) {
	g := NewGraph("adduse")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)
	c := g.NewConst("ctor", f32, 3.0)

	ctx := &MutContext{Pass: "test-adduse"}
	u := ctx.AddUse(sum, c)
	if u == nil {
		t.Fatal("AddUse must return the new slot")
	}
	if u.Val != c || u.User != sum {
		t.Fatal("the new slot must be wired to its def and user")
	}
	uses := UsesOf(c)
	if len(uses) != 1 || uses[0] != u {
		t.Fatalf("c's use-list must hold exactly the new use, got %v", uses)
	}
	if got := UsesOf(a); len(got) != 1 || got[0].User != sum {
		t.Fatalf("a's use-list corrupted by the reallocating append: %v", got)
	}
	if got := UsesOf(b); len(got) != 1 || got[0].User != sum {
		t.Fatalf("b's use-list corrupted by the reallocating append: %v", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chains inconsistent after AddUse past capacity: %v", vs)
	}

	// Same hazard on a variadic node grown repeatedly past capacity.
	call := g.NewCall("ctor", "f", a, b)
	g.AddStatement(call)
	for i := 0; i < 8; i++ {
		ctx.AddUse(call, c)
	}
	if got, want := len(call.(*callNode).ops), 10; got != want {
		t.Fatalf("call has %d operand slots, want %d", got, want)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chains inconsistent after growing call past capacity: %v", vs)
	}
}

func TestRauwGuards(t *testing.T) {
	g := NewGraph("guards")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)

	ctx := &MutContext{Pass: "test-guard"}

	// Self-RAUW is a no-op.
	before := UsesOf(a)
	if err := ctx.ReplaceAllUsesWith(a, a); err != nil {
		t.Fatalf("self-RAUW: %v", err)
	}
	after := UsesOf(a)
	if len(before) != len(after) || before[0] != after[0] {
		t.Fatal("self-RAUW must change nothing")
	}

	// Erase on a def with live uses refuses loudly — never silent dangling
	// pointers (RESEARCH Pattern 2 guard).
	if err := ctx.Erase(a); err == nil {
		t.Fatal("Erase with live uses must return an error")
	}
	if got := UsesOf(a); len(got) != 1 || got[0] != before[0] {
		t.Fatal("refused Erase must leave the use-list untouched")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("guards corrupted the chains: %v", vs)
	}

	// Nil operands are rejected, not panics (T-1-01-02).
	if err := ctx.ReplaceAllUsesWith(nil, b); err == nil {
		t.Fatal("RAUW from nil must return an error")
	}
	if err := ctx.ReplaceAllUsesWith(a, nil); err == nil {
		t.Fatal("RAUW to nil must return an error")
	}
}
