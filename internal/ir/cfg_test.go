package ir

import "testing"

// TestIntrusiveListRemoval pins the review's P1R-02 requirement: a
// BasicBlock stores statements in an intrusive doubly-linked list, so
// removing a statement given its pointer is O(1) — no position search, no
// slice reindex, and no other statement's identity changes.
func TestIntrusiveListRemoval(t *testing.T) {
	g := NewGraph("intrusive")
	f32 := InternScalar(TF32)
	mk := func(v float64) *Statement {
		c := g.NewConst("ctor", f32, v)
		return g.AddStatement(c)
	}
	s1, s2, s3 := mk(1), mk(2), mk(3)

	bb := g.Block()
	if bb.First() != s1 || bb.Last() != s3 {
		t.Fatal("append order must define list order")
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
// passes need for statement motion.
func TestIntrusiveListInsert(t *testing.T) {
	g := NewGraph("insert")
	f32 := InternScalar(TF32)
	mk := func(v float64) *Statement {
		return g.AddStatement(g.NewConst("ctor", f32, v))
	}
	s1, s3 := mk(1), mk(3)
	bb := g.Block()

	s2 := &Statement{id: g.NewID(), kind: StmtAssign, rhs: g.NewConst("ctor", f32, 2)}
	s2.rhs.BasePtr().stmt = s2
	s2.def = SSADef{ssa: s2.rhs.BasePtr().ssa}
	bb.InsertBefore(s2, s3)
	if s1.Next() != s2 || s2.Next() != s3 || s2.Prev() != s1 {
		t.Fatal("InsertBefore must link between the surrounding statements")
	}

	s4 := &Statement{id: g.NewID(), kind: StmtAssign, rhs: g.NewConst("ctor", f32, 4)}
	s4.rhs.BasePtr().stmt = s4
	s4.def = SSADef{ssa: s4.rhs.BasePtr().ssa}
	bb.InsertAfter(s4, s3)
	if s3.Next() != s4 || bb.Last() != s4 {
		t.Fatal("InsertAfter must extend the tail")
	}
	want := []uint64{s1.ID(), s2.ID(), s3.ID(), s4.ID()}
	i := 0
	for s := bb.First(); s != nil; s = s.Next() {
		if s.ID() != want[i] {
			t.Fatalf("position %d has statement %d, want %d", i, s.ID(), want[i])
		}
		i++
	}
}

// TestStatementKindClassification pins the assign/effect split (review's
// building-phase model): stores and returns are pure side-effect statements,
// everything else binds a new local.
func TestStatementKindClassification(t *testing.T) {
	g := NewGraph("kinds")
	f32 := InternScalar(TF32)
	v := g.NewConst("ctor", f32, 1.0)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	bin := g.NewBinOp("ctor", BinOpAdd, v, v)
	sAssign := g.AddStatement(bin)
	if sAssign.Kind() != StmtAssign {
		t.Fatal("binop-rooted statement must be StmtAssign")
	}
	if sAssign.Def().IsInvalid() {
		t.Fatal("assign statement's def must be wired to the RHS value")
	}

	st := g.NewStore("ctor", out, []Node{m}, bin)
	sStore := g.AddStatement(st)
	if sStore.Kind() != StmtEffect {
		t.Fatal("store-rooted statement must be StmtEffect")
	}
	if !sStore.Def().IsInvalid() {
		t.Fatal("effect statement must not bind a local def")
	}

	ret := g.NewReturn("ctor", st)
	sRet := g.AddStatement(ret)
	if sRet.Kind() != StmtEffect {
		t.Fatal("return-rooted statement must be StmtEffect (binds no local)")
	}
}

// TestInputParamReferences pins the review's P1R-01 def resolution: every
// reference to a parameter is an SSAUse whose def IS the graph-owned
// InputParam node — here exercised through a store's target access, an
// iteration cluster's dims, and a lambda parameter.
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
	g.AddStatement(st)

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

	// Fully wired at construction: the verifier reports nothing, and the
	// def-use direction ties every parameter ref list to exactly the records
	// the graph's payload uses own.
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("freshly built param graph must verify clean: %v", vs)
	}
	// The lambda InputParam is NOT a graph param.
	if len(g.Params()) != 3 {
		t.Fatalf("lambda param leaked into the graph param list: %d params", len(g.Params()))
	}
}

// TestEraseIsO1 pins the reshape requirement that MutContext.Erase unlinks a
// statement through the intrusive list (via the root's stmt back-pointer),
// not by scanning the block.
func TestEraseIsO1(t *testing.T) {
	g := NewGraph("erase-o1")
	f32 := InternScalar(TF32)
	first := g.NewConst("ctor", f32, 1.0)
	g.AddStatement(first)
	last := g.NewConst("ctor", f32, 9.0)
	g.AddStatement(last)

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
	if first.BasePtr().stmt == nil || first.BasePtr().stmt.Next() != nil {
		t.Fatal("sibling statement must be untouched")
	}
}
