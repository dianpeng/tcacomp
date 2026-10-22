package ir

// astcfg_test.go pins the construction and accessor contracts of the
// AST-style CFG (quick-261021-ta3 task 1): block-kind classification with
// the soc KeyedAttribute (including the deliberate fail-soft no-metadata
// default), the fail-closed soc error path, AttachAST's refusal set, and
// the persistent per-node scheduling state on Base (D-03). It touches
// nothing of the legacy CFG, Materialize, the printer, or the goldens.

import (
	"strings"
	"testing"
)

// forLoopWithAttrs builds a minimal KindForLoop node with the given soc /
// dosoc / loc KeyedAttribute annotations over a hand-built dimvar header.
func forLoopWithAttrs(t *testing.T, g *Graph, soc, dosoc, loc float64) Node {
	t.Helper()
	u32 := InternScalar(TU32)
	dim := g.NewInputParam("ctor", RoleDim, "d", nil)
	iv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	bound := g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 8))
	socA := g.NewKeyedAttribute("ctor", "soc", g.NewConst("ctor", u32, soc))
	dosocA := g.NewKeyedAttribute("ctor", "dosoc", g.NewConst("ctor", u32, dosoc))
	locA := g.NewKeyedAttribute("ctor", "loc", g.NewConst("ctor", u32, loc))
	return g.NewForLoop("ctor", iv, bound, g.NewConst("ctor", u32, 0), socA, dosocA, locA)
}

func TestASTGroupKind(t *testing.T) {
	blk := NewASTGroup()
	if blk.Kind() != BlockGroup {
		t.Fatalf("NewASTGroup kind = %s, want Group", blk.Kind())
	}
	if blk.Loop() != nil || blk.IV() != nil || blk.Bound() != nil {
		t.Fatal("a Group must carry no loop header info")
	}
	if blk.DoSoC() != -1 || blk.LoC() != 1 {
		t.Fatalf("a Group's DoSoC/LoC = %d/%d, want -1/1", blk.DoSoC(), blk.LoC())
	}
	if got := blk.Items(); len(got) != 0 {
		t.Fatalf("a fresh Group must be empty, got %d items", len(got))
	}
	it := ASTItem{Node: blk.Loop()}
	_ = it // ASTItem zero value: both fields nil is only reachable pre-use
}

func TestASTLoopClassification(t *testing.T) {
	g := NewGraph("cls")
	loop := forLoopWithAttrs(t, g, 1, 2, 64)
	blk, err := NewASTLoop(loop)
	if err != nil {
		t.Fatalf("NewASTLoop = %v", err)
	}
	if blk.Kind() != BlockCTALoop {
		t.Fatalf("soc=1 kind = %s, want CTALoop", blk.Kind())
	}
	ops := OperandsOf(loop)
	if blk.IV().BasePtr() != ops[0].BasePtr() || blk.Bound().BasePtr() != ops[1].BasePtr() {
		t.Fatal("NewASTLoop must expose the loop's ops[0]/ops[1] as IV/Bound")
	}
	if blk.Loop().BasePtr() != loop.BasePtr() {
		t.Fatal("NewASTLoop must remember its ForLoop node")
	}
	if blk.DoSoC() != 2 || blk.LoC() != 64 {
		t.Fatalf("DoSoC/LoC = %d/%d, want 2/64", blk.DoSoC(), blk.LoC())
	}
	if blk.Kind().String() != "CTALoop" {
		t.Fatalf("Kind().String() = %q, want CTALoop", blk.Kind().String())
	}
}

func TestASTLoopKindTable(t *testing.T) {
	cases := []struct {
		soc  float64
		want BlockKind
	}{
		{0, BlockSeqLoop},
		{1, BlockCTALoop},
		{2, BlockThrLoop},
	}
	for _, c := range cases {
		g := NewGraph("k")
		blk, err := NewASTLoop(forLoopWithAttrs(t, g, c.soc, 3, 16))
		if err != nil {
			t.Fatalf("soc=%v: NewASTLoop = %v", c.soc, err)
		}
		if blk.Kind() != c.want {
			t.Fatalf("soc=%v kind = %s, want %s", c.soc, blk.Kind(), c.want)
		}
		if blk.DoSoC() != 3 || blk.LoC() != 16 {
			t.Fatalf("soc=%v: DoSoC/LoC = %d/%d, want 3/16", c.soc, blk.DoSoC(), blk.LoC())
		}
	}
}

// TestASTLoopNoAttrsDefault pins the deliberate fail-soft default: a
// ForLoop built with NO KeyedAttribute attrs schedules as a BlockSeqLoop
// with dosoc -1 and loc 1 — the no-LNT-metadata edge (the same zero-value
// convention tiling.SoCType uses).
func TestASTLoopNoAttrsDefault(t *testing.T) {
	g := NewGraph("plain")
	u32 := InternScalar(TU32)
	dim := g.NewInputParam("ctor", RoleDim, "d", nil)
	iv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	loop := g.NewForLoop("ctor", iv, g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 8)),
		g.NewConst("ctor", u32, 0))
	blk, err := NewASTLoop(loop)
	if err != nil {
		t.Fatalf("NewASTLoop on an attrs-less loop = %v", err)
	}
	if blk.Kind() != BlockSeqLoop {
		t.Fatalf("attrs-less loop kind = %s, want SeqLoop", blk.Kind())
	}
	if blk.DoSoC() != -1 || blk.LoC() != 1 {
		t.Fatalf("attrs-less loop DoSoC/LoC = %d/%d, want -1/1", blk.DoSoC(), blk.LoC())
	}
}

func TestASTLoopBadSocFailsClosed(t *testing.T) {
	g := NewGraph("badsoc")
	loop := forLoopWithAttrs(t, g, 7, 0, 1)
	if _, err := NewASTLoop(loop); err == nil {
		t.Fatal("soc=7 must refuse with an error, got nil")
	}
}

// forLoopWithNonConstAttr builds a ForLoop whose soc is a valid const but
// whose key attribute (dosoc or loc) carries a NON-CONST value node — the
// ME-02 corruption shape (quick-261022-b2m).
func forLoopWithNonConstAttr(t *testing.T, g *Graph, key string) Node {
	t.Helper()
	u32 := InternScalar(TU32)
	dim := g.NewInputParam("ctor", RoleDim, "d", nil)
	iv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	bound := g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 8))
	socA := g.NewKeyedAttribute("ctor", "soc", g.NewConst("ctor", u32, 1))
	nc := g.NewBinOp("ctor", BinOpAdd, g.NewConst("ctor", u32, 1), g.NewConst("ctor", u32, 2))
	return g.NewForLoop("ctor", iv, bound, g.NewConst("ctor", u32, 0), socA,
		g.NewKeyedAttribute("ctor", key, nc))
}

// TestASTLoopNonConstAttrsFailsClosed pins ME-02 (quick-261022-b2m): ALL
// THREE annotation keys fail closed on a non-const value — a corrupted
// dosoc or loc must error exactly like a corrupted soc, never be silently
// masked into the documented default and propagated into ASTLoopMeta.
func TestASTLoopNonConstAttrsFailsClosed(t *testing.T) {
	for _, key := range []string{"dosoc", "loc"} {
		g := NewGraph("nc-" + key)
		loop := forLoopWithNonConstAttr(t, g, key)
		_, err := NewASTLoop(loop)
		if err == nil {
			t.Fatalf("non-const %s must fail closed, got nil", key)
		}
		if want := "non-const " + key; !strings.Contains(err.Error(), want) {
			t.Fatalf("non-const %s error = %q, want it to contain %q", key, err.Error(), want)
		}
	}
}

// TestASTGroupNestAccessor pins ruling 7 (quick-261022-b2m): a plain
// NewASTGroup carries NO owning nest, while NewASTGroupInNest carries the
// given LNT record — same pointer, O(1) identity for final_lower's
// epilogue-to-nest mapping.
func TestASTGroupNestAccessor(t *testing.T) {
	if got := NewASTGroup().Nest(); got != nil {
		t.Fatalf("plain group Nest() = %v, want nil", got)
	}
	lnt := &ASTLNT{}
	blk := NewASTGroupInNest(lnt)
	if blk.Kind() != BlockGroup {
		t.Fatalf("in-nest group kind = %s, want Group", blk.Kind())
	}
	if got := blk.Nest(); got != lnt {
		t.Fatalf("in-nest group Nest() = %p, want the same LNT pointer %p", got, lnt)
	}
}

func TestASTLoopShapeGuards(t *testing.T) {
	g := NewGraph("guards")
	if _, err := NewASTLoop(nil); err == nil {
		t.Fatal("nil loop must refuse")
	}
	c := g.NewConst("ctor", InternScalar(TU32), 0)
	if _, err := NewASTLoop(c); err == nil {
		t.Fatal("a non-ForLoop node must refuse")
	}
}

func TestAttachASTRefusals(t *testing.T) {
	g := NewGraph("attach")
	if err := g.AttachAST(nil); err == nil {
		t.Fatal("nil kernel must refuse")
	}
	if err := g.AttachAST(&ASTKernel{}); err == nil {
		t.Fatal("nil Body must refuse")
	}
	loop := forLoopWithAttrs(t, g, 2, 0, 32)
	loopBlk, err := NewASTLoop(loop)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AttachAST(&ASTKernel{Body: loopBlk}); err == nil {
		t.Fatal("a non-Group Body must refuse")
	}
	k := &ASTKernel{Body: NewASTGroup()}
	if err := g.AttachAST(k); err != nil {
		t.Fatalf("first attach = %v", err)
	}
	if g.AST() == nil {
		t.Fatal("AST() must return the attached kernel")
	}
	if err := g.AttachAST(&ASTKernel{Body: NewASTGroup()}); err == nil {
		t.Fatal("double attach must refuse")
	}
	var nilg *Graph
	if nilg.AST() != nil {
		t.Fatal("nil graph AST() must be nil-safe")
	}
}

func TestSetSchedRoundTrip(t *testing.T) {
	g := NewGraph("sched")
	c := g.NewConst("ctor", InternScalar(TU32), 0)
	if c.BasePtr().SchedSN() != 0 || c.BasePtr().SchedBlock() != nil {
		t.Fatal("a fresh node must read SN 0 / block nil")
	}
	blk := NewASTGroup()
	c.BasePtr().SetSched(0, blk)
	if c.BasePtr().SchedSN() != 0 || c.BasePtr().SchedBlock() != blk {
		t.Fatal("SetSched(0, blk) must round-trip: SN 0 is the legitimate prologue rank")
	}
	loopBlk, err := NewASTLoop(forLoopWithAttrs(t, g, 2, 1, 32))
	if err != nil {
		t.Fatal(err)
	}
	c.BasePtr().SetSched(2, loopBlk)
	if c.BasePtr().SchedSN() != 2 || c.BasePtr().SchedBlock() != loopBlk {
		t.Fatal("SetSched(2, loopBlk) must round-trip")
	}
}
