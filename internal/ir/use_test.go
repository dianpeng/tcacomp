package ir

import (
	"strings"
	"testing"
)

// The tests in this file are written against the handle model (directive 2):
// SSAUse/SSADef are value-type handles over a per-node SSAVar pointee, and
// ref-record indirection makes mutations through ANY copy of a handle visible
// from every other copy. See ssa.go for the data structures.

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
	if uses[0].Def() != c {
		t.Fatal("RAUW must retarget the use handles, not drop them")
	}
	if got := sum.(*binOpNode).ops[1].Def(); got != c {
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

	// A lingering use hand-pointed at the tombstoned node is flagged on
	// re-verify — the exact corruption class Erase protects against. The
	// white-box rewire retargets the record's def pointer without moving it
	// between ref lists, exactly as a leaked pointer would.
	d := g.NewConst("ctor", f32, 4.0)
	y := g.NewBinOp("ctor", BinOpMul, a, d)
	g.AddStatement(y)
	yn := y.(*binOpNode)
	yn.ops[1].rec.def = b.BasePtr().ssa

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

// TestSetDefPropagatesThroughCopies pins the ref-record indirection: SSAUse
// is a value-type handle, so a copy made BEFORE the mutation must observe the
// mutation — through the operand slot AND through UsesOf.
func TestSetDefPropagatesThroughCopies(t *testing.T) {
	g := NewGraph("copies")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)
	c := g.NewConst("ctor", f32, 3.0)

	ctx := &MutContext{Pass: "test-setdef"}
	slot := sum.(*binOpNode).ops[1]
	copied := slot // value-type handle copy, taken before the mutation
	if err := ctx.SetDef(copied, c); err != nil {
		t.Fatalf("SetDef: %v", err)
	}
	if slot.Def() != c {
		t.Fatal("mutation through a handle copy must propagate to every copy (record indirection)")
	}
	if got := UsesOf(c); len(got) != 1 || got[0].Def() != c {
		t.Fatalf("c must own the physically moved record, got %v", got)
	}
	if got := UsesOf(b); len(got) != 0 {
		t.Fatalf("b must have lost the record, got %d uses", len(got))
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chains inconsistent after SetDef: %v", vs)
	}

	// A zero handle is rejected, not a panic (T-1.1-03).
	if err := ctx.SetDef(SSAUse{}, c); err == nil {
		t.Fatal("SetDef on a zero handle must return an error")
	}
}

// TestZeroHandleNilSafety pins T-1.1-03: every accessor on a zero/nil handle
// is nil-safe — no panics anywhere.
func TestZeroHandleNilSafety(t *testing.T) {
	var d SSADef
	if !d.IsInvalid() || d.SSA() != nil || d.Define() != nil {
		t.Fatal("zero SSADef must read as invalid with nil-safe accessors")
	}
	var u SSAUse
	if !u.IsKilled() || u.SSA() != nil || u.Def() != nil {
		t.Fatal("zero SSAUse must read as killed with nil-safe accessors")
	}
	var v *SSAVar
	if v.ID() != 0 || v.Node() != nil || v.Define() != nil || v.RefSize() != 0 {
		t.Fatal("nil *SSAVar accessors must be nil-safe")
	}
}

// TestRemoveUseEachSlotPosition removes each operand slot of a fully-wired
// three-operand select, one graph per position. Slot removal must drop only
// the targeted record: surviving operands keep their wiring, defs' ref lists
// lose exactly the removed record, and Verify stays clean (CR-02 — the
// middle-slot case is the one that used to nil out the WRONG record).
func TestRemoveUseEachSlotPosition(t *testing.T) {
	defs := func(ops []SSAUse) []Node {
		out := make([]Node, len(ops))
		for i, u := range ops {
			out[i] = u.Def()
		}
		return out
	}

	for pos := 0; pos < 3; pos++ {
		g := NewGraph("removeuse")
		f32 := InternScalar(TF32)
		a := g.NewConst("ctor", f32, 1.0)
		b := g.NewConst("ctor", f32, 2.0)
		c := g.NewConst("ctor", f32, 3.0)
		sel := g.NewSelect("ctor", a, b, c)
		g.AddStatement(sel)

		ctx := &MutContext{Pass: "test-removeuse"}
		sn := sel.(*selectNode)
		removed := sn.ops[pos]
		wantSurvivors := defs(sn.ops)

		ctx.RemoveUse(removed)

		if got := len(sn.ops); got != 2 {
			t.Fatalf("pos %d: select has %d slots after RemoveUse, want 2", pos, got)
		}
		if !removed.IsKilled() {
			t.Fatalf("pos %d: removed use must read as killed", pos)
		}
		// The surviving slots in order must be the original minus pos.
		wantSurvivors = append(wantSurvivors[:pos], wantSurvivors[pos+1:]...)
		gotSurvivors := defs(sn.ops)
		for i := range wantSurvivors {
			if gotSurvivors[i] != wantSurvivors[i] {
				t.Fatalf("pos %d: surviving slot %d points at the wrong def", pos, i)
			}
		}
		// Every surviving def's ref list still holds exactly its select use.
		for i, d := range gotSurvivors {
			uses := UsesOf(d)
			if len(uses) != 1 || uses[0] != sn.ops[i] {
				t.Fatalf("pos %d: def %d has uses %v, want exactly its surviving slot", pos, i, uses)
			}
		}
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("pos %d: chains inconsistent after RemoveUse: %v", pos, vs)
		}
	}
}

// TestAddUseKeepsChainsConsistent exercises AddUse on a node whose ops slice
// is exactly at capacity after construction (len 2, cap 2): the append inside
// addSlot reallocates, and every SSAUse handed out earlier — held in operand
// slots and by callers — must still resolve, because handles point at ref
// records, never at slice elements (CR-01).
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
	if u.IsKilled() {
		t.Fatal("AddUse must return a live handle")
	}
	if u.Def() != c {
		t.Fatal("the new slot must be wired to its def")
	}
	uses := UsesOf(c)
	if len(uses) != 1 || uses[0] != u {
		t.Fatalf("c's ref list must hold exactly the new use, got %v", uses)
	}
	if got := UsesOf(a); len(got) != 1 || got[0].Def() != a {
		t.Fatalf("a's ref list corrupted by the reallocating append: %v", got)
	}
	if got := UsesOf(b); len(got) != 1 || got[0].Def() != b {
		t.Fatalf("b's ref list corrupted by the reallocating append: %v", got)
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

// TestRauwPropagatesThroughCopies pins the ref-record indirection across a
// whole-chain move: a handle snapshot taken BEFORE RAUW must resolve to the
// new def afterwards, because RAUW rewires the records the handles point at.
func TestRauwPropagatesThroughCopies(t *testing.T) {
	g := NewGraph("rauw-copies")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.AddStatement(sum)
	c := g.NewConst("ctor", f32, 3.0)

	ctx := &MutContext{Pass: "test-rauw-copies"}
	before := UsesOf(b) // snapshot of live handles, taken before the mutation
	if len(before) != 1 {
		t.Fatalf("b has %d uses, want 1", len(before))
	}
	if err := ctx.ReplaceAllUsesWith(b, c); err != nil {
		t.Fatalf("RAUW: %v", err)
	}
	if before[0].Def() != c {
		t.Fatal("a handle copy taken before RAUW must resolve to the new def (record indirection)")
	}
	if got := UsesOf(c); len(got) != 1 || got[0].Def() != c {
		t.Fatalf("c must own the relocated record, got %v", got)
	}
	if got := sum.(*binOpNode).ops[1].Def(); got != c {
		t.Fatal("the operand slot must see the new def after RAUW")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("chains inconsistent after RAUW: %v", vs)
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
		t.Fatal("refused Erase must leave the ref list untouched")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("guards corrupted the chains: %v", vs)
	}

	// Nil operands are rejected, not panics (T-1.1-03).
	if err := ctx.ReplaceAllUsesWith(nil, b); err == nil {
		t.Fatal("RAUW from nil must return an error")
	}
	if err := ctx.ReplaceAllUsesWith(a, nil); err == nil {
		t.Fatal("RAUW to nil must return an error")
	}
}
