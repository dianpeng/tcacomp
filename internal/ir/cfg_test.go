package ir

import "testing"

// TestIntrusiveListRemoval pins the review's P1R-02 requirement: a
// BasicBlock stores statements in an intrusive doubly-linked list, so
// removing a statement given its pointer is O(1) — no position search, no
// slice reindex, and no other statement's identity changes. The statements
// arrive via Materialize over three chained stores (the sea's effect roots).
func TestIntrusiveListRemoval(t *testing.T) {
	g := NewGraph("intrusive")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	env := NewEnv()
	g.NewStoreChained("ctor", env, out, []Node{m}, g.NewConst("ctor", f32, 1))
	g.NewStoreChained("ctor", env, out, []Node{m}, g.NewConst("ctor", f32, 2))
	g.NewStoreChained("ctor", env, out, []Node{m}, g.NewConst("ctor", f32, 3))

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	s1, s2, s3 := bb.First(), bb.First().Next(), bb.Last()
	if s1 == nil || s1.Next() != s2 || s2.Next() != s3 || s3.Prev() != s2 || s2.Prev() != s1 {
		t.Fatal("materialized statement order must be the effect-root (construction) order")
	}

	// Unlink the middle statement: O(1) via prev/next, order preserved.
	if !bb.Remove(s2) {
		t.Fatal("Remove must report true for a linked statement")
	}
	if bb.First() != s1 || bb.Last() != s3 {
		t.Fatal("removing the middle statement must not touch head/tail")
	}
	if s1.Next() != s3 || s3.Prev() != s1 {
		t.Fatal("surrounding statements must relink across the removed one")
	}
	if s2.Prev() != nil || s2.Next() != nil {
		t.Fatal("the removed statement must be fully detached")
	}
	if s2.RHS().BasePtr().stmt != nil {
		t.Fatal("the removed statement's root must lose its stmt back-pointer")
	}
	if bb.Remove(s2) {
		t.Fatal("double removal must report false, not corrupt the list")
	}

	// Remove the head and the tail.
	if !bb.Remove(s1) || bb.First() != s3 || bb.Last() != s3 {
		t.Fatal("head removal must promote the next statement")
	}
	if !bb.Remove(s3) || bb.First() != nil || bb.Last() != nil {
		t.Fatal("removing the last statement must empty the list")
	}
}

// TestIntrusiveListInsert pins the O(1) insertion primitives that later
// passes need for statement motion. s1 is a materialized store statement;
// s2/s3/s4 are white-box statements hand-linked around it (list surgery is
// independent of how statements were created).
func TestIntrusiveListInsert(t *testing.T) {
	g := NewGraph("insert")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	s1 := g.Block().First()
	s3 := &Statement{id: g.NewID(), kind: StmtAssign, rhs: g.NewConst("ctor", f32, 3)}
	s3.rhs.BasePtr().stmt = s3
	s3.def = SSADef{ssa: s3.rhs.BasePtr().ssa}
	g.Block().Append(s3)

	s2 := &Statement{id: g.NewID(), kind: StmtAssign, rhs: g.NewConst("ctor", f32, 2)}
	s2.rhs.BasePtr().stmt = s2
	s2.def = SSADef{ssa: s2.rhs.BasePtr().ssa}
	g.Block().InsertBefore(s2, s3)
	if s1.Next() != s2 || s2.Next() != s3 || s2.Prev() != s1 {
		t.Fatal("InsertBefore must link between the surrounding statements")
	}

	s4 := &Statement{id: g.NewID(), kind: StmtAssign, rhs: g.NewConst("ctor", f32, 4)}
	s4.rhs.BasePtr().stmt = s4
	s4.def = SSADef{ssa: s4.rhs.BasePtr().ssa}
	g.Block().InsertAfter(s4, s3)
	if s3.Next() != s4 || g.Block().Last() != s4 {
		t.Fatal("InsertAfter must extend the tail")
	}
	want := []uint64{s1.ID(), s2.ID(), s3.ID(), s4.ID()}
	i := 0
	for s := g.Block().First(); s != nil; s = s.Next() {
		if s.ID() != want[i] {
			t.Fatalf("position %d has statement %d, want %d", i, s.ID(), want[i])
		}
		i++
	}
}

// TestStatementKindClassification pins the assign/effect split (review's
// building-phase model): stores and returns are pure side-effect statements,
// everything else binds a new local. Materialize emits only effect roots, so
// the effect kinds are observed on real statements and the assign kind via
// the same classifier.
func TestStatementKindClassification(t *testing.T) {
	g := NewGraph("kinds")
	f32 := InternScalar(TF32)
	v := g.NewConst("ctor", f32, 1.0)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	st := g.NewStore("ctor", out, []Node{m}, v)
	ret := g.NewReturn("ctor", st)
	if classifyStmt(st) != StmtEffect || classifyStmt(ret) != StmtEffect {
		t.Fatal("store and return roots must classify as StmtEffect")
	}
	if classifyStmt(g.NewBinOp("ctor", BinOpAdd, v, v)) != StmtAssign {
		t.Fatal("a value-producing root must classify as StmtAssign")
	}

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().Kind() != StmtEffect {
		t.Fatal("store-rooted statement must be StmtEffect")
	}
	if !bb.First().Def().IsInvalid() {
		t.Fatal("effect statement must not bind a local def")
	}
	if bb.Last().Kind() != StmtEffect {
		t.Fatal("return-rooted statement must be StmtEffect (binds no local)")
	}
	if bb.First().RHS() != st || bb.Last().RHS() != ret {
		t.Fatal("materialized statements must root at the effect roots in construction order")
	}
}

// TestInputParamReferences pins the review's P1R-01 def resolution: every
// reference to a parameter is an SSAUse whose def IS the graph-owned
// InputParam node — here exercised through a store's target access, an
// iteration cluster's dims, and a lambda parameter. Verification is pinned at
// BOTH levels: the Level-1 sea (effect-root walk) and the Level-2 scheduled
// graph must both report clean.
func TestInputParamReferences(t *testing.T) {
	g := NewGraph("params")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	k := g.NewInputParam("ctor", RoleDim, "k", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}, {Name: "k"}}))
	if len(g.Params()) != 3 {
		t.Fatalf("graph owns %d params, want 3", len(g.Params()))
	}

	// Lambda parameter: an InputParam node referenced through Lambda.params
	// payload uses AND through the body's paramRef.
	v := g.NewLambdaInputParam("ctor", "v", f32)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), g.NewConst("ctor", f32, 2))
	rootMap := g.NewMap("ctor", nil, NewLambda([]Node{v}, body), []Node{m, k})
	ic := g.NewIterCluster("ctor", []Node{m, k}, rootMap)
	st := g.NewStore("ctor", out, []Node{m, k}, ic)

	// The store's payload uses resolve to the parameter nodes.
	sn := st.(*storeNode)
	if sn.target.tensorUse.SSA().Node() != out {
		t.Fatal("store target tensor use must resolve to the tensor InputParam node")
	}
	for i, ax := range sn.target.axes {
		want := []Node{m, k}[i]
		if ax.SSA().Node() != want {
			t.Fatalf("store axis use %d must resolve to its dim InputParam node", i)
		}
	}

	// The iteration cluster's dims and the lambda's param resolve the same way.
	icn := ic.(*iterClusterNode)
	if icn.dims[0].SSA().Node() != m || icn.dims[1].SSA().Node() != k {
		t.Fatal("iteration cluster dim uses must resolve to the dim InputParam nodes")
	}
	mn := rootMap.(*mapNode)
	if mn.lam.params[0].SSA().Node() != v {
		t.Fatal("lambda param use must resolve to the lambda InputParam node")
	}
	if mn.as[0].SSA().Node() != m || mn.as[1].SSA().Node() != k {
		t.Fatal("map as-order uses must resolve to the dim InputParam nodes")
	}

	// Fully wired at construction: the verifier reports nothing at Level 1
	// (params + effect roots) and again after Materialize at Level 2 — the
	// def-use direction ties every parameter ref list to exactly the records
	// the graph's payload uses own.
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("freshly built param graph must verify clean at Level 1: %v", vs)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("materialized param graph must verify clean at Level 2: %v", vs)
	}
	// The lambda InputParam is NOT a graph param.
	if len(g.Params()) != 3 {
		t.Fatalf("lambda param leaked into the graph param list: %d params", len(g.Params()))
	}
}

// TestEraseIsO1 pins the reshape requirement that MutContext.Erase unlinks a
// statement through the intrusive list (via the root's stmt back-pointer),
// not by scanning the block. Erase runs at either level; here the graph is
// materialized first so the unlink path is exercised.
func TestEraseIsO1(t *testing.T) {
	g := NewGraph("erase-o1")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	st := g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1.0))
	last := g.NewReturn("ctor", st)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	ctx := &MutContext{Pass: "test-erase"}
	if err := ctx.Erase(last); err != nil {
		t.Fatalf("Erase: %v", err)
	}
	if g.Block().First().Next() != nil || g.Block().Last() != g.Block().First() {
		t.Fatal("Erase must unlink the statement from the intrusive list")
	}
	if last.BasePtr().stmt != nil {
		t.Fatal("erased root must lose its stmt back-pointer")
	}
	if !last.BasePtr().erased {
		t.Fatal("Erase must tombstone the node's ID")
	}
	if st.BasePtr().stmt == nil || st.BasePtr().stmt.Next() != nil {
		t.Fatal("sibling statement must be untouched")
	}
}
