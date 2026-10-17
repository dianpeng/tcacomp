package ir

import "testing"

// dceTestGraph builds the review's P1R-08 scenario in sea form: a live
// store/return pair, a shared def feeding the store, and a two-link dead
// chain (dead def feeding a dead def) with no statement structure — at
// Level 1 reachability from the params and effect roots decides liveness.
//
//	effect roots: store(out[@m], shared), return store
//	shared = 1.0 + 2.0   (live: feeds the store)
//	tmp = shared * shared    (dead: nothing uses tmp)
//	dead2 = tmp + tmp        (dead: only dead tmp uses it)
func dceTestGraph() (*Graph, Node, Node) {
	g := NewGraph("dce")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	shared := g.NewBinOp("ctor", BinOpAdd, g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 2))
	tmp := g.NewBinOp("ctor", BinOpMul, shared, shared)
	dead2 := g.NewBinOp("ctor", BinOpAdd, tmp, tmp)

	st := g.NewStore("ctor", out, []Node{m}, shared)
	g.NewReturn("ctor", st)

	return g, tmp, dead2
}

// TestBuildingDCEWipesUnreferenced pins review step 1 at Level 1: an
// unreferenced side-effect-free chain is wiped, live nodes untouched, and the
// materialized statement list holds exactly the store and the return.
func TestBuildingDCEWipesUnreferenced(t *testing.T) {
	g, tmp, dead2 := dceTestGraph()
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("precondition: fresh graph must verify clean at Level 1: %v", vs)
	}

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}

	if !tmp.BasePtr().erased || !dead2.BasePtr().erased {
		t.Fatal("dead chain roots must be tombstoned by Erase")
	}
	if got := len(UsesOf(tmp)); got != 0 {
		t.Fatalf("erased tmp keeps %d use records, want 0 (dead2 died first, killing them)", got)
	}

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First() == nil || bb.First().Next() == nil || bb.First().Next().Next() != nil {
		t.Fatal("after DCE the block must hold exactly the store and the return")
	}
	if bb.First().Kind() != StmtEffect || bb.First().Next().Kind() != StmtEffect {
		t.Fatal("the survivors are the store and the return (StmtEffect, StmtEffect)")
	}

	// The surviving graph is verifier-clean at BOTH levels: shared lost its
	// dead-side records and keeps exactly the store's use.
	shared := bb.First().RHS().(*storeNode).ops[0].SSA().Node()
	if got := len(UsesOf(shared)); got != 1 {
		t.Fatalf("shared def has %d uses after DCE, want 1 (the store)", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE materialized graph must verify clean: %v", vs)
	}
}

// TestBuildingDCEFixpointChainKill pins the fixpoint requirement: a def whose
// EVERY use is dead becomes dead itself — the kill propagates backward
// through the dead chain (dead2 dies, tmp loses its last use, tmp dies)
// without a live root ever referencing either.
func TestBuildingDCEFixpointChainKill(t *testing.T) {
	g, tmp, dead2 := dceTestGraph()
	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if !tmp.BasePtr().erased || !dead2.BasePtr().erased {
		t.Fatal("both dead chain links must be wiped")
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	count := 0
	for s := g.Block().First(); s != nil; s = s.Next() {
		if s.Kind() != StmtEffect {
			t.Fatalf("survivor statement %d must be StmtEffect, got %v", s.ID(), s.Kind())
		}
		count++
	}
	if count != 2 {
		t.Fatalf("exactly the store and the return survive, got %d statements", count)
	}
}

// TestBuildingDCEKeepsSideEffects pins review step 1's other half: effect
// roots are live by construction — a store and its return survive with no
// users at all.
func TestBuildingDCEKeepsSideEffects(t *testing.T) {
	g := NewGraph("effects")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	v := g.NewConst("ctor", f32, 1)
	st := g.NewStore("ctor", out, []Node{m}, v)
	g.NewReturn("ctor", st)

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if st.BasePtr().erased {
		t.Fatal("the store must survive DCE even with zero users")
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g.Block().First() == nil || g.Block().First().Next() == nil || g.Block().First().Next().Next() != nil {
		t.Fatal("effect statements must survive DCE even with zero users")
	}
}

// TestBuildingDCESharedDefSurvives pins the sweep's shared-node rule: a def
// used by BOTH a dead chain and a live store keeps exactly its live-side
// record; the dead consumer is wiped and its records die with it.
func TestBuildingDCESharedDefSurvives(t *testing.T) {
	g, tmp, _ := dceTestGraph()
	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	shared := g.Block().First().RHS().(*storeNode).ops[0].SSA().Node()
	if shared.BasePtr().erased {
		t.Fatal("shared def must survive: the live store references it")
	}
	if got := len(UsesOf(shared)); got != 1 {
		t.Fatalf("shared def has %d uses after DCE, want 1 (tmp's records died with it)", got)
	}
	if !tmp.BasePtr().erased {
		t.Fatal("the dead consumer of shared must be wiped")
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE graph must verify clean: %v", vs)
	}
}

// TestBuildingDCENoDeadIsNoOp: a clean graph passes through untouched — the
// materialized statement order and identities are the effect roots in
// registration order.
func TestBuildingDCENoDeadIsNoOp(t *testing.T) {
	g := NewGraph("clean")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	v := g.NewConst("ctor", f32, 1)
	st := g.NewStore("ctor", out, []Node{m}, v)
	ret := g.NewReturn("ctor", st)

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g.Block().First().RHS() != st || g.Block().First().Next().RHS() != ret || g.Block().Last().RHS() != ret {
		t.Fatal("no-op DCE must not touch effect roots or their statement order")
	}
}

// TestBuildingDCESeaWipesAbandonedChain is the regression test for the sea
// sweep's payload discipline: an abandoned map/iteration-cluster chain (the
// builder's "started a new chain" debris) carries PAYLOAD uses — iteration
// dims, as-orders, lambda params — and erasing it must detach those records
// from the kernel params' ref lists. Before the payload-aware Erase, the
// def-use invariant flagged the dim param for records no visited node owned.
func TestBuildingDCESeaWipesAbandonedChain(t *testing.T) {
	g := NewGraph("abandoned")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	// The abandoned chain: map + iteration cluster + lambda, never stored.
	v := g.NewLambdaInputParam("ctor", "v", f32)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), g.NewConst("ctor", f32, 2))
	abandoned := g.NewMap("ctor", nil, NewLambda([]Node{v}, body), []Node{m})
	abandoned = g.NewIterCluster("ctor", []Node{m}, abandoned)

	// The real chain: a store + return over an unrelated value.
	st := g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))
	g.NewReturn("ctor", st)

	if err := BuildingDCE(g); err != nil {
		t.Fatalf("BuildingDCE: %v", err)
	}
	if !abandoned.BasePtr().erased {
		t.Fatal("the abandoned chain's root must be wiped")
	}
	// Level 1: the dim param's ref list must not retain payload records from
	// the erased chain — this is exactly what the payload-aware Erase fixes.
	// @m was used by the abandoned map's as-order, the cluster's dims, and
	// the live store's axis: after the wipe only the store's axis remains.
	if got := len(UsesOf(m)); got != 1 {
		t.Fatalf("dim @m has %d uses after DCE, want 1 (the live store's axis; the abandoned chain's payload records died with it)", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE sea must verify clean (payload records detached): %v", vs)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("post-DCE materialized graph must verify clean: %v", vs)
	}
	if g.Block().First().RHS() != st || g.Block().First().Next().Next() != nil {
		t.Fatal("only the live store and return schedule statements")
	}
}
