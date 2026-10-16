package ir

import "testing"

// dceTestGraph builds the review's P1R-08 scenario: a live store/return
// pair, one directly-unreferenced side-effect-free statement, and a
// two-statement dead chain (dead def feeding a dead def).
//
//	statement:  shared = 1.0 + 2.0          (live: feeds the store)
//	statement:  tmp = shared * shared        (dead: nothing uses tmp)
//	statement:  dead2 = tmp + tmp            (dead: only dead tmp uses it)
//	statement:  store(out[@m], shared)       (live: side effect)
//	statement:  return store                 (live: side effect)
func dceTestGraph() (*Graph, *Statement, *Statement) {
	g := NewGraph("dce")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	shared := g.NewBinOp("ctor", BinOpAdd, g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 2))
	g.AddStatement(shared)

	tmp := g.NewBinOp("ctor", BinOpMul, shared, shared)
	sTmp := g.AddStatement(tmp)
	dead2 := g.NewBinOp("ctor", BinOpAdd, tmp, tmp)
	sDead2 := g.AddStatement(dead2)

	st := g.NewStore("ctor", out, []Node{m}, shared)
	g.AddStatement(st)
	g.AddStatement(g.NewReturn("ctor", st))

	return g, sTmp, sDead2
}

// TestBuildingDCEWipesUnreferenced pins review step 1: an unreferenced
// side-effect-free statement is wiped, live statements untouched.
func TestBuildingDCEWipesUnreferenced(t *testing.T) {
	g, sTmp, sDead2 := dceTestGraph()
	sShared := g.Block().First()
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("precondition: fresh graph must verify clean: %v", vs)
	}

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}

	// Both dead statements are unlinked from the intrusive list.
	for s := g.Block().First(); s != nil; s = s.Next() {
		if s == sTmp || s == sDead2 {
			t.Fatalf("dead statement %d survived DCE", s.ID())
		}
	}
	if g.Block().First() != sShared {
		t.Fatal("the live shared-def statement must head the block after DCE")
	}
	if g.Block().First().Next() == nil || g.Block().First().Next().Next() == nil || g.Block().First().Next().Next().Next() != nil {
		t.Fatal("after DCE the block must hold exactly shared-def, store, and return")
	}

	// The wipe is structural: the dead roots are tombstoned and their
	// operand records killed (tmp/dead2 no longer reference shared).
	if !sTmp.RHS().BasePtr().erased || !sDead2.RHS().BasePtr().erased {
		t.Fatal("dead roots must be tombstoned by Erase")
	}
	if got := len(UsesOf(sTmp.RHS())); got != 0 {
		t.Fatalf("erased tmp keeps %d use records, want 0 (dead2 died first, killing them)", got)
	}

	// The surviving graph is verifier-clean: shared lost its dead-side
	// records and keeps exactly the store's use.
	if got := len(UsesOf(g.Block().First().RHS())); got != 1 {
		t.Fatalf("shared def has %d uses after DCE, want 1 (the store)", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE graph must verify clean: %v", vs)
	}
}

// TestBuildingDCEFixpointChainKill pins the review's fixpoint requirement: a
// def whose EVERY use is dead becomes dead itself — the kill propagates
// backward through the dead chain (dead2 dies, tmp loses its last use, tmp
// dies) without a live statement ever referencing either.
func TestBuildingDCEFixpointChainKill(t *testing.T) {
	g, _, _ := dceTestGraph()
	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	// Both chain links are gone from the block; the three survivors are the
	// shared-def statement (Assign — the store consumes it) and the store and
	// return (StmtEffect, StmtEffect).
	var kinds []StmtKind
	for s := g.Block().First(); s != nil; s = s.Next() {
		kinds = append(kinds, s.Kind())
	}
	if len(kinds) != 3 || kinds[0] != StmtAssign || kinds[1] != StmtEffect || kinds[2] != StmtEffect {
		t.Fatalf("survivors = %v, want [StmtAssign StmtEffect StmtEffect] (shared-def, store, return)", kinds)
	}
}

// TestBuildingDCEKeepsSideEffects pins review step 1's other half:
// side-effecting statements are live roots even when nothing consumes their
// value — a store and its return survive with no users at all.
func TestBuildingDCEKeepsSideEffects(t *testing.T) {
	g := NewGraph("effects")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	v := g.NewConst("ctor", f32, 1)
	st := g.NewStore("ctor", out, []Node{m}, v)
	g.AddStatement(st)
	g.AddStatement(g.NewReturn("ctor", st))

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if g.Block().First() == nil || g.Block().First().Next() == nil || g.Block().First().Next().Next() != nil {
		t.Fatal("effect statements must survive DCE even with zero users")
	}
}

// TestBuildingDCESharedDefSurvives pins the wipe's shared-node rule: a def
// used by BOTH a dead statement and a live store keeps exactly its live-side
// record; the dead consumer is wiped.
func TestBuildingDCESharedDefSurvives(t *testing.T) {
	g, sTmp, _ := dceTestGraph()
	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	// sTmp (tmp = shared * shared) is dead and wiped; shared survives
	// because the store uses it. The killed side left no dangling records.
	shared := g.Block().First().Next().RHS().(*storeNode).ops[0].SSA().Node()
	if shared.BasePtr().erased {
		t.Fatal("shared def must survive: the live store references it")
	}
	if got := len(UsesOf(shared)); got != 1 {
		t.Fatalf("shared def has %d uses after DCE, want 1 (tmp's records died with it)", got)
	}
	if !sTmp.RHS().BasePtr().erased {
		t.Fatal("the dead consumer of shared must be wiped")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE graph must verify clean: %v", vs)
	}
}

// TestBuildingDCENoDeadIsNoOp: a clean graph passes through untouched —
// statement identities and list order are preserved.
func TestBuildingDCENoDeadIsNoOp(t *testing.T) {
	g := NewGraph("clean")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	v := g.NewConst("ctor", f32, 1)
	st := g.NewStore("ctor", out, []Node{m}, v)
	s1 := g.AddStatement(st)
	s2 := g.AddStatement(g.NewReturn("ctor", st))

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if g.Block().First() != s1 || g.Block().First().Next() != s2 || g.Block().Last() != s2 {
		t.Fatal("no-op DCE must not touch statement identities or order")
	}
}
