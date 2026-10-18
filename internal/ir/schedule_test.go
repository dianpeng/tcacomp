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
	g.NewMemStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))

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

// TestMaterializeEffectfulOrder pins Materialize's only ordering rule (the
// ir tier's deliberately minimal default scheduling, D-7): statements are
// the LIVE effectful nodes reachable from the sea roots (R2/R3: the end
// node's operand + ImpactBy paths), in allocation (construction) order —
// and shared pure values reachable from them do not become statements of
// their own.
func TestMaterializeEffectfulOrder(t *testing.T) {
	g := NewGraph("order")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out1 := g.NewInputParam("ctor", RoleTensor, "out1", InternTensor(TF32, []DimRef{{Name: "m"}}))
	out2 := g.NewInputParam("ctor", RoleTensor, "out2", InternTensor(TF32, []DimRef{{Name: "m"}}))

	shared := g.NewBinOp("ctor", BinOpAdd, g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 2))
	env := NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, shared)
	st2 := g.NewMemStoreChained("ctor", env, out2, []Node{m}, shared)
	ret := g.NewReturnChained("ctor", env, []Node{out1, out2}, st1)

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().RHS() != st1 || bb.First().Next().RHS() != st2 || bb.Last().RHS() != ret {
		t.Fatal("statement order must be the effectful nodes' allocation order: store1, store2, end node")
	}
	if bb.First().Next().Next().Next() != nil {
		t.Fatal("exactly three statements: one per live effectful node; shared defs are not statements")
	}
	for s := bb.First(); s != nil; s = s.Next() {
		if s.Kind() != StmtEffect {
			t.Fatalf("effectful-node statement %d must be StmtEffect", s.ID())
		}
	}
	// R2: the end node carries BOTH tensors' final chain states via ImpactBy.
	imp := ImpactBy(ret)
	if len(imp) != 2 || imp[0].Def() != st1 || imp[1].Def() != st2 {
		t.Fatalf("end node ImpactBy = %v, want [st1 st2] (the final chain states)", imp)
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
	env := NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []Node{m}, g.NewConst("ctor", f32, 1))
	g.PinStatementID(st, 40)
	g.NewReturnChained("ctor", env, []Node{out}, st) // unpinned

	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	if bb.First().ID() != 40 {
		t.Fatalf("pinned store statement ID = %d, want 40", bb.First().ID())
	}
	if bb.Last().ID() != 6 {
		t.Fatalf("unpinned end-node statement ID = %d, want 6 (the next monotonic ID: nodes took 1-5, the pin did not consume counter IDs)",
			bb.Last().ID())
	}
}

// TestMaterializeSkipsErasedAndUnreachable pins the R2/R3 liveness interplay:
// an un-chained store that no end node references is unreachable from the
// sea roots and schedules nothing even while un-erased, and once the end
// node is tombstoned the whole chain schedules nothing.
func TestMaterializeSkipsErasedAndUnreachable(t *testing.T) {
	// Unreachable: no end node references the store (neither as a returned
	// value nor through an ImpactBy edge) — it is not a root (R3).
	g := NewGraph("unreachable")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	g.NewMemStore("ctor", out, []Node{m}, g.NewConst("ctor", f32, 1))
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g.Block().First() != nil {
		t.Fatal("a store no end node references must schedule nothing (stores are not roots)")
	}

	// Erased: the end node dies, and its chain stores die with it.
	g2 := NewGraph("erased")
	m2 := g2.NewInputParam("ctor", RoleDim, "m", nil)
	out2 := g2.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	env := NewEnv()
	st := g2.NewMemStoreChained("ctor", env, out2, []Node{m2}, g2.NewConst("ctor", f32, 2))
	ret := g2.NewReturnChained("ctor", env, []Node{out2}, st)

	ctx := &MutContext{Pass: "test"}
	if err := ctx.Erase(ret); err != nil {
		t.Fatalf("Erase(ret): %v", err)
	}
	if err := g2.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g2.Block().First() != nil {
		t.Fatal("with the end node erased, its chain stores schedule nothing")
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
	env := NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []Node{m}, g.NewConst("ctor", f32, 1))
	g.NewReturnChained("ctor", env, []Node{out}, st)

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
