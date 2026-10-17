package ir

import (
	"strings"
	"testing"
)

// TestMaterializeGuards pins the Level-1 nil-safety contract (T-quick-02)
// and Materialize's error discipline: a fresh graph has no CFG and no block,
// and Materialize refuses double scheduling.
func TestMaterializeGuards(t *testing.T) {
	g := NewGraph("guards")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))

	if g.Block() != nil || g.CFG() != nil {
		t.Fatal("a fresh Level-1 graph must have no block and no CFG")
	}
	if g.Materialized() {
		t.Fatal("a fresh graph must not read as materialized")
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if !g.Materialized() || g.Block() == nil || g.CFG() == nil || g.Block() != g.CFG().Entry() {
		t.Fatal("after Materialize the graph must expose one stable block through Block and CFG")
	}
	if err := g.Materialize(); err == nil {
		t.Fatal("double Materialize must return an error, not reschedule")
	}
	var nilGraph *Graph
	if err := nilGraph.Materialize(); err == nil {
		t.Fatal("Materialize on a nil graph must return an error")
	}
}

// TestMaterializeEffectRootOrder pins the scheduler's only ordering rule:
// statements are the LIVE effect roots in registration (construction) order,
// and shared pure values reachable from a root do not become statements of
// their own.
func TestMaterializeEffectRootOrder(t *testing.T) {
	g := NewGraph("order")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out1 := g.NewInputParam("ctor", RoleTensor, "out1", InternTensor(TF32, []DimRef{{Name: "m"}}))
	out2 := g.NewInputParam("ctor", RoleTensor, "out2", InternTensor(TF32, []DimRef{{Name: "m"}}))

	shared := g.NewBinOp("ctor", BinOpAdd, g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 2))
	st1 := g.NewStore("ctor", out1, []Node{m}, shared)
	st2 := g.NewStore("ctor", out2, []Node{m}, shared)
	ret := g.NewReturn("ctor", st1)

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().RHS() != st1 || bb.First().Next().RHS() != st2 || bb.Last().RHS() != ret {
		t.Fatal("statement order must be effect-root registration order: store1, store2, return")
	}
	if bb.First().Next().Next().Next() != nil {
		t.Fatal("exactly three statements: one per effect root; shared defs are not statements")
	}
	for s := bb.First(); s != nil; s = s.Next() {
		if s.Kind() != StmtEffect {
			t.Fatalf("effect-root statement %d must be StmtEffect", s.ID())
		}
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("materialized graph must verify clean: %v", vs)
	}
}

// TestMaterializePinnedIDs pins the builder-reservation contract (Scheme X of
// the two-level adaptation): a PinStatementID-reserved ID lands on exactly
// that root's statement; unpinned roots draw from the monotonic counter in
// emission order. This is what keeps the committed canonical goldens
// byte-identical across the scheduler change.
func TestMaterializePinnedIDs(t *testing.T) {
	g := NewGraph("pinned")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	g.PinStatementID(nil, 99) // nil root: ignored, never a panic
	st := g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))
	g.PinStatementID(st, 40)
	g.NewReturn("ctor", st) // unpinned

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().ID() != 40 {
		t.Fatalf("pinned store statement ID = %d, want 40", bb.First().ID())
	}
	if bb.Last().ID() != 6 {
		t.Fatalf("unpinned return statement ID = %d, want 6 (the next monotonic ID: nodes took 1-5, the pin did not consume counter IDs)",
			bb.Last().ID())
	}
}

// TestMaterializeSkipsErasedRoots pins the DCE interplay: effect roots
// tombstoned before Materialize schedule nothing; the remaining roots keep
// their relative order.
func TestMaterializeSkipsErasedRoots(t *testing.T) {
	g := NewGraph("erased-roots")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	dead := g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))
	st := g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 2))
	ret := g.NewReturn("ctor", st)

	ctx := &MutContext{Pass: "test"}
	if err := ctx.Erase(ret); err != nil { // the return uses st, so it dies first
		t.Fatalf("Erase(ret): %v", err)
	}
	if err := ctx.Erase(dead); err != nil {
		t.Fatalf("Erase(dead): %v", err)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().RHS() != st || bb.First().Next() != nil {
		t.Fatal("erased effect roots must schedule nothing; only the live store remains")
	}
}

// TestPrintRequiresMaterialize pins the codegen-surface contract: the
// canonical printer and the SSA numbering are Level-2 operations — an
// unmaterialized graph yields a deterministic marker, never partial content.
func TestPrintRequiresMaterialize(t *testing.T) {
	g := NewGraph("print-guard")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	g.NewStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))

	if got := Print(g); got != "ir: graph not materialized\n" {
		t.Fatalf("Print on an unmaterialized graph = %q, want the deterministic marker", got)
	}
	if n := ComputeNumbering(g); n == nil {
		t.Fatal("ComputeNumbering on an unmaterialized graph must not return nil")
	} else if _, ok := n.Num(out); ok {
		t.Fatal("ComputeNumbering on an unmaterialized graph must yield an empty numbering")
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if got := Print(g); !strings.Contains(got, "statement #") {
		t.Fatalf("Print after Materialize must render statements, got %q", got)
	}
}
