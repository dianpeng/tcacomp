package ir

import "testing"

// typecell_test.go pins the computed-type cell (quick-261019-tu3 Task 1):
// every node carries a Base.ty cell that starts nil (uninferred), is written
// ONLY through MutContext.SetType (idempotent: an Equal type neither writes
// nor re-stamps), and is read through the TypeOf/IsErased package faces.

// TestTypeCellFreshNil pins that freshly built nodes report an uninferred
// type and a live (non-erased) cell across representative kinds.
func TestTypeCellFreshNil(t *testing.T) {
	g := NewGraph("cell")
	f := InternScalar(TF32)
	nodes := []Node{
		g.NewConst("test", f, 1),
		g.NewInputParam("test", RoleScalar, "s", f),
		g.NewBinOp("test", BinOpAdd, nil, nil),
		g.NewMemNew("test", nil),
		g.NewReturn("test"),
	}
	for _, n := range nodes {
		if got := TypeOf(n); got != nil {
			t.Fatalf("fresh node %d (%s): TypeOf = %v, want nil", n.BasePtr().ID(), n.KindOf(), got)
		}
		if IsErased(n) {
			t.Fatalf("fresh node %d (%s): IsErased = true", n.BasePtr().ID(), n.KindOf())
		}
	}
	// Nil-safety of the read faces.
	if TypeOf(nil) != nil {
		t.Fatal("TypeOf(nil) must be nil")
	}
	if IsErased(nil) {
		t.Fatal("IsErased(nil) must be false")
	}
}

// TestSetTypeStamps pins the write path: SetType records the type and stamps
// LastTouchedBy with the mutating pass.
func TestSetTypeStamps(t *testing.T) {
	g := NewGraph("stamp")
	n := g.NewConst("test", InternScalar(TF32), 1)
	ctx := &MutContext{Pass: "typeinfer"}
	u32 := InternScalar(TU32)
	if err := ctx.SetType(n, u32); err != nil {
		t.Fatalf("SetType: %v", err)
	}
	if got := TypeOf(n); got == nil || !got.Equal(u32) {
		t.Fatalf("TypeOf = %v, want scalar u32", got)
	}
	if got := n.BasePtr().LastTouchedBy(); got != "typeinfer" {
		t.Fatalf("LastTouchedBy = %q, want %q", got, "typeinfer")
	}
}

// TestSetTypeIdempotent pins the idempotency primitive: a second SetType
// with an Equal type neither changes TypeOf nor re-stamps LastTouchedBy;
// a second SetType with a DIFFERENT type does update both.
func TestSetTypeIdempotent(t *testing.T) {
	g := NewGraph("idem")
	n := g.NewConst("test", InternScalar(TF32), 1)
	ctx := &MutContext{Pass: "typeinfer"}
	u32 := InternScalar(TU32)
	f32 := InternScalar(TF32)
	if err := ctx.SetType(n, u32); err != nil {
		t.Fatalf("first SetType: %v", err)
	}

	// Move the stamp out of the way: a no-op second SetType must not
	// re-stamp it (package-internal write, test-only).
	n.BasePtr().lastTouchedBy = "someone-else"
	if err := ctx.SetType(n, InternScalar(TU32)); err != nil {
		t.Fatalf("idempotent SetType: %v", err)
	}
	if got := TypeOf(n); got == nil || !got.Equal(u32) {
		t.Fatalf("idempotent SetType changed the type: %v", got)
	}
	if got := n.BasePtr().LastTouchedBy(); got != "someone-else" {
		t.Fatalf("idempotent SetType re-stamped: LastTouchedBy = %q", got)
	}

	// A DIFFERENT type does write and re-stamp.
	if err := ctx.SetType(n, f32); err != nil {
		t.Fatalf("changing SetType: %v", err)
	}
	if got := TypeOf(n); got == nil || !got.Equal(f32) {
		t.Fatalf("changing SetType did not update the type: %v", got)
	}
	if got := n.BasePtr().LastTouchedBy(); got != "typeinfer" {
		t.Fatalf("changing SetType did not re-stamp: LastTouchedBy = %q", got)
	}
}

// TestSetTypeNilNode pins the choke-point argument check: a nil node errors,
// never panics (mirrors SetDef/Erase).
func TestSetTypeNilNode(t *testing.T) {
	ctx := &MutContext{Pass: "typeinfer"}
	if err := ctx.SetType(nil, InternScalar(TU32)); err == nil {
		t.Fatal("SetType(nil, ...) must return an error")
	}
}

// TestIsErasedAfterErase pins that IsErased flips exactly on MutContext.Erase
// (AllocatedNodes keeps tombstones; the sweep must be able to see them).
func TestIsErasedAfterErase(t *testing.T) {
	g := NewGraph("erase")
	p := g.NewInputParam("test", RoleScalar, "s", InternScalar(TF32))
	c := g.NewConst("test", InternScalar(TF32), 1)
	if IsErased(c) || IsErased(p) {
		t.Fatal("fresh nodes must not report erased")
	}
	ctx := &MutContext{Pass: "dce"}
	if err := ctx.Erase(c); err != nil {
		t.Fatalf("Erase(use-free const): %v", err)
	}
	if !IsErased(c) {
		t.Fatal("the erased const must report IsErased")
	}
	if IsErased(p) {
		t.Fatal("the live param must not report erased")
	}
	// The tombstone leaves AllocatedNodes (live nodes only) and shows up in
	// Tombstones — IsErased is the per-node face of that same state.
	for _, n := range g.AllocatedNodes() {
		if n.BasePtr() == c.BasePtr() {
			t.Fatal("the tombstoned const must not appear in AllocatedNodes")
		}
	}
	found := false
	for _, n := range g.Tombstones() {
		if n.BasePtr() == c.BasePtr() {
			found = true
		}
	}
	if !found {
		t.Fatal("the tombstoned const must appear in Tombstones")
	}
}
