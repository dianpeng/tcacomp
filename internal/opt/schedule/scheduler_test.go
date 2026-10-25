package schedule_test

// scheduler_test.go pins the rewritten schedule pass's behavior
// (quick-261021-ta3 task 3): the spec's edge cases of
// doc/optana/schedule.md — no-loop graphs, LICM by construction, nested
// nests with single-placement shared nodes, effect-only ordering, multiple
// LNTs, the no-metadata fail-soft default, kernel metadata, and the
// fail-closed gates. Graphs are built DIRECTLY at the IR level (the
// gvn_test fixture pattern) and the pass is invoked directly —
// Scheduler{}.Run(nil, g) — so the tests exercise the pass's behavior
// without the tile-lower precondition the pipeline-level Requires pins.
// All assertions are deterministic: items are walked in placement order,
// never map order.

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling" // Preserved pin: the provenance record's cache name
)

// fx bundles a fixture graph's common kernel params.
type fx struct {
	g   *ir.Graph
	f32 ir.Type
	u32 ir.Type
	m   ir.Node // dim param
	out ir.Node // tensor param
	env *ir.Env
}

func newFx(name string) *fx {
	g := ir.NewGraph(name)
	return &fx{
		g:   g,
		f32: ir.InternScalar(ir.TF32),
		u32: ir.InternScalar(ir.TU32),
		m:   g.NewInputParam("ctor", ir.RoleDim, "m", nil),
		out: g.NewInputParam("ctor", ir.RoleTensor, "out",
			ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}})),
		env: ir.NewEnv(),
	}
}

func (f *fx) tensor(name string) ir.Node {
	return f.g.NewInputParam("ctor", ir.RoleTensor, name,
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
}

// loopAttrs builds the soc/dosoc/loc KeyedAttribute annotation triple.
func (f *fx) loopAttrs(soc, dosoc, loc float64) []ir.Node {
	attr := func(key string, v float64) ir.Node {
		return f.g.NewKeyedAttribute("ctor", key, f.g.NewConst("ctor", f.u32, v))
	}
	return []ir.Node{attr("soc", soc), attr("dosoc", dosoc), attr("loc", loc)}
}

// dimVar builds a normalized (0,1) dimvar + bounded header over the dim.
func (f *fx) dimVar() (iv, bound ir.Node) {
	iv = f.g.NewDimVar("ctor", f.m, f.g.NewConst("ctor", f.u32, 0), f.g.NewConst("ctor", f.u32, 1))
	return iv, f.g.NewBoundedDimVar("ctor", iv, f.g.NewConst("ctor", f.u32, 8))
}

// run schedules g and fails the test on any error.
func run(t *testing.T, g *ir.Graph) *ir.ASTKernel {
	t.Helper()
	if err := (schedule.Scheduler{}).Run(nil, g); err != nil {
		t.Fatalf("schedule run = %v", err)
	}
	k := g.AST()
	if k == nil {
		t.Fatal("g.AST() must be attached after a successful run")
	}
	return k
}

// itemNodes flattens one block's scheduled-node items in placement order.
func itemNodes(blk *ir.ASTBlock) []ir.Node {
	var out []ir.Node
	for _, it := range blk.Items() {
		if it.Node != nil {
			out = append(out, it.Node)
		}
	}
	return out
}

// indexOf returns n's position in list (-1 when absent).
func indexOf(list []ir.Node, n ir.Node) int {
	for i, x := range list {
		if x.BasePtr() == n.BasePtr() {
			return i
		}
	}
	return -1
}

// countOccurrences counts how many block item lists across the whole AST
// contain n — the single-placement assertion walks every Group and Loop.
func countOccurrences(blk *ir.ASTBlock, n ir.Node) int {
	c := 0
	for _, it := range blk.Items() {
		switch {
		case it.Node != nil && it.Node.BasePtr() == n.BasePtr():
			c++
		case it.Block != nil:
			c += countOccurrences(it.Block, n)
		}
	}
	return c
}

// bodyBlocks returns the kernel body's top-level block items — since
// quick-261023-lex, ONE lexical Group per LNT (head items + the embedded
// LBBN root + after-nest items). Node items are
// SKIPPED: since final lowering joined the default pipeline
// (quick-261022-l1o) the kernel Body legitimately carries statement items —
// the RuntimeKernelInfo prepended as its FIRST item (LR-11) — beside the
// cluster blocks.
func bodyBlocks(t *testing.T, k *ir.ASTKernel) []*ir.ASTBlock {
	t.Helper()
	var out []*ir.ASTBlock
	for _, it := range k.Body.Items() {
		if it.Block == nil {
			continue // a kernel-level statement item (the KernelInfo prepend)
		}
		out = append(out, it.Block)
	}
	return out
}

// rootBlockOf returns lex's embedded root-loop block — the LNT's nest is a
// statement INSIDE its lexical Group (quick-261023-lex). v0 nests are
// chains: the first (and only) Block item.
func rootBlockOf(t *testing.T, lex *ir.ASTBlock) *ir.ASTBlock {
	t.Helper()
	for _, it := range lex.Items() {
		if it.Block != nil {
			return it.Block
		}
	}
	t.Fatal("the lexical Group carries no embedded loop nest")
	return nil
}

// afterRoot reports whether n is an item of lex positioned AFTER the
// embedded root block — the old epilogue region, now just an item position
// (quick-261023-lex).
func afterRoot(t *testing.T, lex *ir.ASTBlock, n ir.Node) bool {
	t.Helper()
	rootIdx := -1
	for i, it := range lex.Items() {
		if it.Block != nil {
			rootIdx = i
			break
		}
	}
	if rootIdx < 0 {
		t.Fatal("the lexical Group carries no embedded loop nest")
	}
	for i, it := range lex.Items() {
		if it.Node != nil && it.Node.BasePtr() == n.BasePtr() {
			return i > rootIdx
		}
	}
	return false
}

// headNodes returns lex's HEAD-region node items — those before the
// embedded root block (the old prologue region).
func headNodes(t *testing.T, lex *ir.ASTBlock) []ir.Node {
	t.Helper()
	var out []ir.Node
	for _, it := range lex.Items() {
		if it.Block != nil {
			break // the head ends at the embedded nest
		}
		if it.Node != nil {
			out = append(out, it.Node)
		}
	}
	return out
}

// TestNoLoops pins spec step 2's degenerate shape: a store-only graph with
// no ForLoop yields a kernel body that is a SINGLE Group holding every live
// non-placeholder node in dependency order, all at SN 0, every such node's
// SchedBlock pointing at that group. The two placeholders (the dim param m
// and the tensor param out) anchor at SN -1 into the Prototype block
// (quick-261022-e2u, D-1) — rendered first, not inside the body.
func TestNoLoops(t *testing.T) {
	f := newFx("noloop")
	a := f.g.NewConst("ctor", f.f32, 3)
	b := f.g.NewConst("ctor", f.f32, 4)
	prod := f.g.NewBinOp("ctor", ir.BinOpMul, a, b)
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, prod)
	end := f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)

	k := run(t, f.g)
	if k.Body.Kind() != ir.BlockGroup {
		t.Fatalf("no-loop body kind = %s, want Group", k.Body.Kind())
	}
	if k.Prototype.Kind() != ir.BlockPrototype {
		t.Fatalf("no-loop prototype kind = %s, want Prototype", k.Prototype.Kind())
	}
	items := itemNodes(k.Body)
	if len(items) != 5 { // a, b, prod, st, end — m/out are prototype placeholders (D-1)
		t.Fatalf("body holds %d nodes, want the 5 non-placeholder live nodes", len(items))
	}
	if indexOf(items, a) > indexOf(items, prod) || indexOf(items, b) > indexOf(items, prod) {
		t.Fatal("producers must precede their consumer (dependency order)")
	}
	if indexOf(items, prod) > indexOf(items, st) {
		t.Fatal("the store must follow its value")
	}
	for _, n := range []ir.Node{a, b, prod, st, end} {
		if n.BasePtr().SchedSN() != 0 {
			t.Fatalf("node %d SN = %d, want 0", n.BasePtr().ID(), n.BasePtr().SchedSN())
		}
		if n.BasePtr().SchedBlock() != k.Body {
			t.Fatalf("node %d must land in the kernel body group", n.BasePtr().ID())
		}
	}
	for _, n := range []ir.Node{f.m, f.out} {
		if n.BasePtr().SchedSN() != ir.PrototypeSN || n.BasePtr().SchedBlock() != k.Prototype {
			t.Fatalf("placeholder %d must anchor at SN -1 in the Prototype block, got %d/%p",
				n.BasePtr().ID(), n.BasePtr().SchedSN(), n.BasePtr().SchedBlock())
		}
	}
	protoItems := itemNodes(k.Prototype)
	if indexOf(protoItems, f.m) < 0 || indexOf(protoItems, f.out) < 0 {
		t.Fatal("the Prototype block's items must be exactly the live params")
	}
}

// TestSingleNestLICM pins scheduling-number LICM: an invariant expression
// lands in the LNT prologue at SN 0 while the loop-varying node (depends
// on the loop iv) lands in the ThrLoop block at SN 1; the ForLoop node's
// own linkage points at its Loop block.
func TestSingleNestLICM(t *testing.T) {
	f := newFx("licm")
	iv, bound := f.dimVar()
	inv := f.g.NewBinOp("ctor", ir.BinOpMul, f.g.NewConst("ctor", f.f32, 3), f.g.NewConst("ctor", f.f32, 4))
	varying := f.g.NewBinOp("ctor", ir.BinOpAdd, iv, inv)
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, varying)
	loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
	end := f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 1 {
		t.Fatalf("body items = %d blocks, want ONE lexical Group per LNT", len(blocks))
	}
	prologue, thr := blocks[0], rootBlockOf(t, blocks[0])
	epilogue := prologue // the after-nest region is an item position in the SAME Group
	if prologue.Kind() != ir.BlockGroup || thr.Kind() != ir.BlockThrLoop {
		t.Fatalf("cluster kinds = [%s %s], want [Group(lexical) ThrLoop]",
			prologue.Kind(), thr.Kind())
	}
	if thr.Loop().BasePtr() != loop.BasePtr() {
		t.Fatal("the ThrLoop block must materialize the ForLoop node")
	}
	if indexOf(itemNodes(prologue), inv) < 0 {
		t.Fatal("the invariant must land in the LNT prologue Group (SN 0)")
	}
	if inv.BasePtr().SchedSN() != 0 || inv.BasePtr().SchedBlock() != prologue {
		t.Fatal("the invariant must read SN 0 with the prologue linkage")
	}
	thrItems := itemNodes(thr)
	if indexOf(thrItems, varying) < 0 || indexOf(thrItems, st) < 0 {
		t.Fatal("the varying node and its store must land in the ThrLoop block")
	}
	if indexOf(thrItems, varying) > indexOf(thrItems, st) {
		t.Fatal("the varying node must precede its store")
	}
	if varying.BasePtr().SchedSN() != 1 || varying.BasePtr().SchedBlock() != thr {
		t.Fatalf("varying node SN/block = %d/%p, want 1/ThrLoop", varying.BasePtr().SchedSN(), varying.BasePtr().SchedBlock())
	}
	if loop.BasePtr().SchedSN() != 1 || loop.BasePtr().SchedBlock() != thr {
		t.Fatal("the ForLoop node must carry SN 1 and its own Loop block linkage")
	}
	if iv.BasePtr().SchedBlock() != thr || bound.BasePtr().SchedBlock() != thr {
		t.Fatal("the loop header (iv/bound) must land in the Loop block")
	}
	// CR-02 pin (quick-261022-b2m): the end node's deps live in the loop
	// (the root loop itself at depth 1, the chained store), so it lands in
	// the epilogue Group — AFTER the nest — keeping SN 1: the propagated
	// DEPTH REQUIREMENT, not an after-nest rank (ruling 6). The old
	// assertion clamped it to the prologue and encoded the CR-02 inversion
	// (a block executing BEFORE its own dependencies) as spec.
	if end.BasePtr().SchedBlock() != epilogue {
		t.Fatal("the end node must land in the epilogue Group (after its producing nest)")
	}
	if end.BasePtr().SchedSN() != 1 {
		t.Fatalf("end node SN = %d, want 1 (the propagated depth requirement, ruling 6)", end.BasePtr().SchedSN())
	}
	if end.BasePtr().SchedBlock().Nest() == nil || end.BasePtr().SchedBlock().Nest().Root.BasePtr() != loop.BasePtr() {
		t.Fatal("the epilogue Group must expose its owning nest via Nest() (ruling 7)")
	}
	if countOccurrences(k.Body, varying) != 1 || countOccurrences(k.Body, st) != 1 {
		t.Fatal("every node must be placed exactly once")
	}
}

// TestNestedNestSinglePlacement pins the shared-node rule over a nested
// nest: a node varying with the INNER loop lands ONCE at the deepest
// necessary block (SN 2, the inner ThrLoop) — no duplication across the
// whole AST.
func TestNestedNestSinglePlacement(t *testing.T) {
	f := newFx("nested")
	outerIV := f.g.NewDimVar("ctor", f.m, f.g.NewConst("ctor", f.u32, 0), f.g.NewConst("ctor", f.u32, 4))
	outerBound := f.g.NewBoundedDimVar("ctor", outerIV, f.g.NewConst("ctor", f.u32, 64))
	innerIV := f.g.NewDimVar("ctor", f.m, f.g.NewConst("ctor", f.u32, 0), f.g.NewConst("ctor", f.u32, 1))
	// fix1.md D-b nesting: the inner loop's bound references the outer IV in
	// its data-dep closure (the loopforest forOverNested idiom) — the old
	// body-operand link is gone.
	innerBound := f.g.NewBoundedDimVar("ctor", innerIV,
		f.g.NewBinOp("ctor", ir.BinOpAdd, outerIV, f.g.NewConst("ctor", f.u32, 8)))
	shared := f.g.NewBinOp("ctor", ir.BinOpAdd, innerIV, f.g.NewConst("ctor", f.f32, 1))
	use := f.g.NewBinOp("ctor", ir.BinOpMul, shared, outerIV)
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, use)
	inner := f.g.NewForLoop("ctor", innerIV, innerBound, f.loopAttrs(2, 1, 32)...)
	outer := f.g.NewForLoop("ctor", outerIV, outerBound, f.loopAttrs(1, 0, 64)...)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, outer)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 1 {
		t.Fatalf("body items = %d blocks, want ONE lexical Group per LNT", len(blocks))
	}
	cta := rootBlockOf(t, blocks[0])
	if cta.Kind() != ir.BlockCTALoop {
		t.Fatalf("outer block kind = %s, want CTALoop", cta.Kind())
	}
	var thr *ir.ASTBlock
	for _, it := range cta.Items() {
		if it.Block != nil {
			thr = it.Block
		}
	}
	if thr == nil || thr.Kind() != ir.BlockThrLoop {
		t.Fatalf("nested block = %v, want the inner ThrLoop", thr)
	}
	if got := countOccurrences(k.Body, shared); got != 1 {
		t.Fatalf("shared node appears %d times across the AST, want exactly 1", got)
	}
	if shared.BasePtr().SchedSN() != 2 || shared.BasePtr().SchedBlock() != thr {
		t.Fatalf("shared node SN/block = %d/%p, want 2/inner ThrLoop (deepest necessary block)",
			shared.BasePtr().SchedSN(), shared.BasePtr().SchedBlock())
	}
	if indexOf(itemNodes(thr), st) < 0 {
		t.Fatal("the store must land in the inner ThrLoop block")
	}
	if outer.BasePtr().SchedBlock() != cta || inner.BasePtr().SchedBlock() != thr {
		t.Fatal("each loop node must land in its own Loop block")
	}
}

// TestEffectOnlyChain pins same-rank effect ordering: two stores chained
// by ImpactBy (no data edge between them) keep the chain order inside the
// single block.
func TestEffectOnlyChain(t *testing.T) {
	f := newFx("effects")
	st1 := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 1))
	st2 := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 2))
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st2)

	k := run(t, f.g)
	items := itemNodes(k.Body)
	i1, i2 := indexOf(items, st1), indexOf(items, st2)
	if i1 < 0 || i2 < 0 {
		t.Fatal("both stores must land in the kernel body group")
	}
	if i1 > i2 {
		t.Fatal("the ImpactBy-predecessor store must be scheduled first (effect edge, no data edge)")
	}
}

// TestMultipleLNTs pins cluster concatenation in forest-root allocation
// order and the shared rank-0 invariant landing in the FIRST LNT's
// prologue only (which dominates every later cluster).
func TestMultipleLNTs(t *testing.T) {
	f := newFx("multi")
	out2 := f.tensor("out2")
	inv := f.g.NewBinOp("ctor", ir.BinOpMul, f.g.NewConst("ctor", f.f32, 3), f.g.NewConst("ctor", f.f32, 5))

	ivA, boundA := f.dimVar()
	stA := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewBinOp("ctor", ir.BinOpAdd, ivA, inv))
	loopA := f.g.NewForLoop("ctor", ivA, boundA, f.loopAttrs(2, 0, 32)...)

	ivB, boundB := f.dimVar()
	stB := f.g.NewMemStoreChained("ctor", f.env, out2, []ir.Node{f.m}, f.g.NewBinOp("ctor", ir.BinOpAdd, ivB, inv))
	loopB := f.g.NewForLoop("ctor", ivB, boundB, f.loopAttrs(1, 0, 64)...)

	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out, out2}, loopA, loopB)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 2 {
		t.Fatalf("body items = %d blocks, want ONE lexical Group per LNT [A, B]", len(blocks))
	}
	aPro, aRoot := blocks[0], rootBlockOf(t, blocks[0])
	bPro, bRoot := blocks[1], rootBlockOf(t, blocks[1])
	aEpi, bEpi := aPro, bPro // after-nest: item positions in the same Groups
	if aRoot.Kind() != ir.BlockThrLoop || bRoot.Kind() != ir.BlockCTALoop {
		t.Fatalf("LBBN roots = [%s %s], want [ThrLoop CTALoop] in allocation order", aRoot.Kind(), bRoot.Kind())
	}
	if aRoot.Loop().BasePtr() != loopA.BasePtr() || bRoot.Loop().BasePtr() != loopB.BasePtr() {
		t.Fatal("the clusters must follow the roots' allocation order (A before B)")
	}
	if len(itemNodes(aPro)) == 0 {
		t.Fatal("A's prologue must hold the shared rank-0 invariant")
	}
	if indexOf(itemNodes(aPro), inv) < 0 {
		t.Fatal("the shared invariant must land in A's prologue")
	}
	if countOccurrences(k.Body, inv) != 1 {
		t.Fatal("the shared invariant must be placed exactly once (A's prologue only)")
	}
	if len(headNodes(t, bPro)) != 0 {
		t.Fatal("B's head region must stay empty (rank-0 nodes are claimed by the first LNT)")
	}
	// CR-02 pin (quick-261022-b2m): the end node's deps span BOTH nests
	// (both roots and both in-loop stores), so it lands in the LAST
	// contributing root's epilogue — B's — keeping SN 1 (its propagated
	// depth requirement; both roots sit at depth 1); A's epilogue stays
	// empty (D-04 landing pads plus the pass-placed after-nest consumers).
	if indexOf(itemNodes(aEpi), f.g.End()) >= 0 {
		t.Fatal("A's cluster must not hold the end node (its home is the LAST contributing root)")
	}
	if indexOf(itemNodes(bEpi), f.g.End()) < 0 {
		t.Fatal("the end node must land in B's (the last root's) after-nest region")
	}
	if !afterRoot(t, bEpi, f.g.End()) {
		t.Fatal("the end node must sit AFTER B's embedded nest (the old epilogue position)")
	}
	if f.g.End().BasePtr().SchedSN() != 1 {
		t.Fatalf("end node SN = %d, want 1 (the propagated depth requirement, ruling 6)", f.g.End().BasePtr().SchedSN())
	}
	if nst := f.g.End().BasePtr().SchedBlock().Nest(); nst == nil || nst.Root.BasePtr() != loopB.BasePtr() {
		t.Fatal("the end node's epilogue must identify B's LNT via Nest() (ruling 7)")
	}
	if indexOf(itemNodes(aRoot), stA) < 0 || indexOf(itemNodes(bRoot), stB) < 0 {
		t.Fatal("each nest's store must land in its own LBBN root block")
	}
}

// TestNoLNTMetadata pins the fail-soft default: a ForLoop built with NO
// KeyedAttribute attrs schedules as a BlockSeqLoop with loc 1 and the pass
// does not error.
func TestNoLNTMetadata(t *testing.T) {
	f := newFx("plain")
	iv, bound := f.dimVar()
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewBinOp("ctor", ir.BinOpAdd, iv, f.g.NewConst("ctor", f.f32, 1)))
	loop := f.g.NewForLoop("ctor", iv, bound)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	seq := rootBlockOf(t, bodyBlocks(t, k)[0])
	if seq.Kind() != ir.BlockSeqLoop {
		t.Fatalf("attrs-less loop block = %s, want SeqLoop", seq.Kind())
	}
	if seq.LoC() != 1 || seq.DoSoC() != -1 {
		t.Fatalf("defaults LoC/DoSoC = %d/%d, want 1/-1", seq.LoC(), seq.DoSoC())
	}
	if indexOf(itemNodes(seq), st) < 0 {
		t.Fatal("the store must land in the SeqLoop block")
	}
	_ = loop
}

// TestKernelMetadata pins the spec-4 kernel metadata: Params pass through,
// Gmem lists every live GMEM allocation in allocation order with its
// extent, and LNTs carries the per-loop SoC info.
func TestKernelMetadata(t *testing.T) {
	f := newFx("meta")
	extent := f.g.NewConst("ctor", f.u32, 128)
	tmp := f.g.NewMemNew("ctor", extent)
	iv, bound := f.dimVar()
	load := f.g.NewMemLoad("ctor", tmp, []ir.Node{iv})
	f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, load)
	loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 1, 32)...)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	if len(k.Params) != 2 || k.Params[0].BasePtr() != f.m.BasePtr() || k.Params[1].BasePtr() != f.out.BasePtr() {
		t.Fatalf("Params = %d entries, want the kernel params in declaration order", len(k.Params))
	}
	if len(k.Gmem) != 1 {
		t.Fatalf("Gmem = %d allocations, want 1", len(k.Gmem))
	}
	if k.Gmem[0].New.BasePtr() != tmp.BasePtr() || k.Gmem[0].Extent.BasePtr() != extent.BasePtr() {
		t.Fatal("Gmem must list the MemNew with its extent (op[0])")
	}
	// quick-261022-e2u (D-1/D-2): the canonical schedule view — the GMEM
	// MemNew and both params anchor at SN -1 as items of the Prototype
	// block (rendered first); the extent const stays scheduled at SN 0.
	if k.Prototype.Kind() != ir.BlockPrototype {
		t.Fatalf("prototype kind = %s, want Prototype", k.Prototype.Kind())
	}
	if tmp.BasePtr().SchedSN() != ir.PrototypeSN || tmp.BasePtr().SchedBlock() != k.Prototype {
		t.Fatal("the GMEM MemNew must anchor at SN -1 in the Prototype block")
	}
	if extent.BasePtr().SchedSN() != 0 || extent.BasePtr().SchedBlock() == k.Prototype {
		t.Fatal("the extent const is a value: scheduled at SN 0, NOT in the Prototype")
	}
	for _, p := range []ir.Node{f.m, f.out} {
		if p.BasePtr().SchedSN() != ir.PrototypeSN || p.BasePtr().SchedBlock() != k.Prototype {
			t.Fatalf("param %d must anchor at SN -1 in the Prototype block", p.BasePtr().ID())
		}
	}
	protoItems := itemNodes(k.Prototype)
	if len(protoItems) != 3 { // m, out, tmp — exactly the live placeholders
		t.Fatalf("prototype items = %d, want exactly the 3 live placeholders (m, out, tmp)", len(protoItems))
	}
	if len(k.LNTs) != 1 {
		t.Fatalf("LNTs = %d entries, want 1", len(k.LNTs))
	}
	lnt := k.LNTs[0]
	if lnt.Root.BasePtr() != loop.BasePtr() {
		t.Fatal("the LNT root must be the nest root loop")
	}
	if len(lnt.Loops) != 1 {
		t.Fatalf("LNT loops = %d, want 1", len(lnt.Loops))
	}
	if lnt.Loops[0].Soc != 2 || lnt.Loops[0].Dosoc != 1 || lnt.Loops[0].Loc != 32 {
		t.Fatalf("loop SoC metadata = %d/%d/%d, want 2/1/32",
			lnt.Loops[0].Soc, lnt.Loops[0].Dosoc, lnt.Loops[0].Loc)
	}
}

// TestReductionShapeEpilogue pins THE ruling-4/ruling-6 shape
// (quick-261022-b2m, review probe 2 / CR-02): a kernel-level consumer of an
// in-loop value — final_lower.md §"Reducer Lower Integration"'s reduction
// shape, R = n * 2 consuming in-loop n — lands in its producing nest's
// EPILOGUE Group (never in the prologue before its dependency), keeping
// SN == 1 == the reduced loop's depth (the propagated DEPTH REQUIREMENT,
// not an after-nest rank). The nest is recoverable through BOTH routes
// final_lower documents: R's own SN/block-Nest() AND R's operand's
// SchedBlock() == the ThrLoop block.
func TestReductionShapeEpilogue(t *testing.T) {
	f := newFx("redshape")
	iv, bound := f.dimVar()
	n := f.g.NewBinOp("ctor", ir.BinOpAdd, iv, f.g.NewConst("ctor", f.f32, 1))
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, n)
	loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
	R := f.g.NewBinOp("ctor", ir.BinOpMul, n, f.g.NewConst("ctor", f.f32, 2)) // kernel-level consumer of in-loop n
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop, R)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	prologue, thr := blocks[0], rootBlockOf(t, blocks[0])
	epilogue := prologue // after-nest: an item position in the same Group
	if thr.Kind() != ir.BlockThrLoop {
		t.Fatalf("root block kind = %s, want ThrLoop", thr.Kind())
	}
	// R: epilogue Group, SN 1 (== the reduced loop's depth), nest identity.
	if R.BasePtr().SchedBlock() != epilogue {
		t.Fatal("the reduction shape must land in its nest's epilogue Group (after the loop)")
	}
	if R.BasePtr().SchedSN() != 1 {
		t.Fatalf("reduction shape SN = %d, want 1 (the reduced loop's depth — depth requirement, ruling 6)", R.BasePtr().SchedSN())
	}
	if nst := R.BasePtr().SchedBlock().Nest(); nst == nil || nst.Root.BasePtr() != loop.BasePtr() {
		t.Fatal("the epilogue Group must identify the producing nest via Nest() (ruling 7)")
	}
	// DUAL ROUTE: R's operand's SchedBlock IS the loop block — either route
	// identifies the loop (final_lower's "reducer's own SN OR its operand").
	if n.BasePtr().SchedBlock() != thr {
		t.Fatal("the in-loop operand's SchedBlock must be the ThrLoop block (the dual route)")
	}
	// n itself stays in the loop.
	if indexOf(itemNodes(thr), n) < 0 || indexOf(itemNodes(thr), st) < 0 {
		t.Fatal("the in-loop value and its store must stay in the ThrLoop block")
	}
	if !afterRoot(t, prologue, R) {
		t.Fatal("the reduction shape must sit AFTER the nest (never hoisted before its dependency, CR-02)")
	}
}

// TestLambdaParamsExcludedFromPrototype pins P-9 (quick-261022-l1o): after
// schedule over a Reduce-carrying graph, no RoleLambda InputParam is an
// item of the Prototype block — lambda formals are extracted-function
// material (the reduce-inline exemption keeps them alive as declared
// formals), NOT kernel-signature placeholders. Un-anchored they stay
// ordinary SN-0 nodes landing in the kernel/prologue Group.
func TestLambdaParamsExcludedFromPrototype(t *testing.T) {
	f := newFx("lambdaformals")
	iv, bound := f.dimVar()
	acc := f.g.NewLambdaInputParam("ctor", "acc", f.f32)
	now := f.g.NewLambdaInputParam("ctor", "now", f.f32)
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", f.f32, 0),
		f.g.NewBinOp("ctor", ir.BinOpAdd, iv, f.g.NewConst("ctor", f.f32, 1)),
		ir.NewLambda([]ir.Node{acc, now}, comb))
	f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, red)
	loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	for _, it := range k.Prototype.Items() {
		if it.Node == nil {
			continue
		}
		if role, ok := ir.RoleOf(it.Node); ok && role == ir.RoleLambda {
			t.Fatalf("lambda formal node %d must NOT be a Prototype item (P-9: extracted-function material, not a signature placeholder)",
				it.Node.BasePtr().ID())
		}
	}
	// The formals stay live, land at SN 0 (ordinary un-anchored pure nodes),
	// and the prototype still holds exactly the kernel params.
	for name, p := range map[string]ir.Node{"acc": acc, "now": now} {
		if p.BasePtr().Erased() {
			t.Fatalf("the lambda formal %s must stay live (the reduce's combiner)", name)
		}
		if got := p.BasePtr().SchedSN(); got != 0 {
			t.Fatalf("the lambda formal %s must stay an SN-0 node, got SN %d", name, got)
		}
	}
	protoParams := 0
	for _, it := range k.Prototype.Items() {
		if it.Node != nil {
			protoParams++
		}
	}
	if protoParams != len(k.Params) {
		t.Fatalf("prototype items = %d, want exactly the %d kernel params (lambda formals excluded)", protoParams, len(k.Params))
	}
}

// TestPostNestStoreEpilogue pins review probe 3 (CR-02's per-tensor effect
// order): a kernel-level store chained (ImpactBy) after an in-loop store on
// the same tensor lands in the nest's epilogue — the "after the whole loop"
// order tile-lower's repointChains establishes — never in the prologue
// before the loop's own store.
func TestPostNestStoreEpilogue(t *testing.T) {
	f := newFx("postnest")
	out2 := f.tensor("out2")
	iv, bound := f.dimVar()
	st1b := f.g.NewMemStoreChained("ctor", f.env, out2, []ir.Node{f.m},
		f.g.NewBinOp("ctor", ir.BinOpAdd, iv, f.g.NewConst("ctor", f.f32, 1)))
	loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
	st2 := f.g.NewMemStoreChained("ctor", f.env, out2, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 7))
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out, out2}, loop, st2)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	thr, epilogue := rootBlockOf(t, blocks[0]), blocks[0] // after-nest: same Group
	if st2.BasePtr().SchedBlock() != epilogue || st2.BasePtr().SchedSN() != 1 {
		t.Fatalf("post-nest store SN/block = %d/%p, want 1/epilogue", st2.BasePtr().SchedSN(), st2.BasePtr().SchedBlock())
	}
	if indexOf(itemNodes(thr), st1b) < 0 {
		t.Fatal("the in-loop store must stay in the ThrLoop block")
	}
	if indexOf(itemNodes(epilogue), st2) < 0 {
		t.Fatal("the post-nest store must land in the epilogue Group item list")
	}
}

// TestEffectfulNoHoist pins HI-02's content under the fix1.md sea model
// (quick-261022-b2m, recast quick-261024-sdk): an EPOCH-VARYING effectful
// access (the fill idiom — its axes reference the loop's IV even though its
// value is invariant) is an IV member whose SN already meets its loop's
// depth by construction, so it stays in-loop; a fully invariant access
// (no IV reference anywhere in its closure) is unordered w.r.t. the loop
// and legally hoists to the prologue; pure invariants hoist, and GMEM
// allocations anchor in the Prototype (D-1).
func TestEffectfulNoHoist(t *testing.T) {
	t.Run("epoch store stays in loop", func(t *testing.T) {
		f := newFx("nohoist-store")
		iv, bound := f.dimVar()
		// The fill idiom: the axes reference the iv (epoch-varying address),
		// the value is invariant. Membership via the IV relation keeps the
		// per-epoch re-execution; there is no separate floor pass post-fix1.
		st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{iv}, f.g.NewConst("ctor", f.f32, 1))
		loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

		k := run(t, f.g)
		thr := rootBlockOf(t, bodyBlocks(t, k)[0])
		if st.BasePtr().SchedSN() != 1 || st.BasePtr().SchedBlock() != thr {
			t.Fatalf("epoch effectful store SN/block = %d/%p, want 1/ThrLoop (IV membership)", st.BasePtr().SchedSN(), st.BasePtr().SchedBlock())
		}
		if indexOf(itemNodes(thr), st) < 0 {
			t.Fatal("the epoch store must be an ITEM of the loop block")
		}
	})
	t.Run("pure hoists, allocation exempt", func(t *testing.T) {
		f := newFx("nohoist-alloc")
		iv, bound := f.dimVar()
		inv := f.g.NewBinOp("ctor", ir.BinOpMul, f.g.NewConst("ctor", f.f32, 3), f.g.NewConst("ctor", f.f32, 4))
		buf := f.g.NewMemNew("ctor", f.g.NewConst("ctor", f.u32, 128)) // allocation: exempt from the floor
		load := f.g.NewMemLoad("ctor", buf, []ir.Node{iv})             // varies with the iv: in-loop
		st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewBinOp("ctor", ir.BinOpMul, load, inv))
		loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

		k := run(t, f.g)
		blocks := bodyBlocks(t, k)
		prologue, thr := blocks[0], rootBlockOf(t, blocks[0])
		if inv.BasePtr().SchedSN() != 0 || inv.BasePtr().SchedBlock() != prologue {
			t.Fatal("the pure invariant must still hoist to the prologue (LICM intact)")
		}
		// quick-261022-e2u (D-1/D-4): a KindMemNew (GMEM) is a PLACEHOLDER —
		// anchored at SN -1 into the Prototype block, not scheduled. (The
		// allocation non-membership likewise governs the tiered MemNew classes,
		// pinned by the pipeline fill test's buffer-at-SN-0 leg.)
		if buf.BasePtr().SchedSN() != ir.PrototypeSN || buf.BasePtr().SchedBlock() != k.Prototype {
			t.Fatalf("GMEM allocation SN/block = %d/%p, want -1/Prototype (placeholder anchoring)", buf.BasePtr().SchedSN(), buf.BasePtr().SchedBlock())
		}
		if load.BasePtr().SchedSN() != 1 || load.BasePtr().SchedBlock() != thr {
			t.Fatalf("the iv-varying load SN/block = %d/%p, want 1/ThrLoop", load.BasePtr().SchedSN(), load.BasePtr().SchedBlock())
		}
		if st.BasePtr().SchedSN() != 1 || st.BasePtr().SchedBlock() != thr {
			t.Fatalf("the store SN/block = %d/%p, want 1/ThrLoop", st.BasePtr().SchedSN(), st.BasePtr().SchedBlock())
		}
	})
}

// TestBranchedNestRefusal pins HI-01 (quick-261022-b2m): two sibling loops
// under one parent (both nested in it by the fix1.md D-b IV data-dep
// relation — each sibling's bound references the outer IV) refuse
// fail-closed — byDepth would silently cross-product the siblings'
// iteration spaces. Mirrors tile-lower's F3 contract.
func TestBranchedNestRefusal(t *testing.T) {
	f := newFx("branched")
	out2 := f.tensor("out2")
	u32 := f.u32
	ivO, boundO := f.dimVar()
	// Sibling 1: its bound references the outer IV (D-b nesting).
	iv1 := f.g.NewDimVar("ctor", f.m, f.g.NewConst("ctor", u32, 0), f.g.NewConst("ctor", u32, 1))
	bound1 := f.g.NewBoundedDimVar("ctor", iv1, f.g.NewBinOp("ctor", ir.BinOpAdd, ivO, f.g.NewConst("ctor", u32, 4)))
	st1 := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{iv1}, f.g.NewConst("ctor", f.f32, 1))
	inner1 := f.g.NewForLoop("ctor", iv1, bound1, f.loopAttrs(0, -1, 1)...)
	// Sibling 2: same shape over a second dimvar (also IV-nested in outer).
	iv2 := f.g.NewDimVar("ctor", f.m, f.g.NewConst("ctor", u32, 0), f.g.NewConst("ctor", u32, 1))
	bound2 := f.g.NewBoundedDimVar("ctor", iv2, f.g.NewBinOp("ctor", ir.BinOpAdd, ivO, f.g.NewConst("ctor", u32, 4)))
	st2 := f.g.NewMemStoreChained("ctor", f.env, out2, []ir.Node{iv2}, f.g.NewConst("ctor", f.f32, 2))
	inner2 := f.g.NewForLoop("ctor", iv2, bound2, f.loopAttrs(0, -1, 1)...)
	outer := f.g.NewForLoop("ctor", ivO, boundO, f.loopAttrs(1, -1, 1)...)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out, out2}, outer)
	_ = inner1
	_ = inner2
	_ = st1
	_ = st2

	err := (schedule.Scheduler{}).Run(nil, f.g)
	if err == nil || !strings.Contains(err.Error(), "branched") {
		t.Fatalf("Run on a branched nest = %v, want the branched-nest refusal", err)
	}
	if f.g.AST() != nil {
		t.Fatal("the refused run must not attach an AST")
	}
	_ = outer
}

// TestIterClusterGate pins ME-03 (quick-261022-b2m): a LIVE KindIterCluster
// (the legacy compound header, kept live through the store's value path) is
// refused exactly like a live raw KindIter — an abstract HIR header must
// never be scheduled as an ordinary AST statement.
func TestIterClusterGate(t *testing.T) {
	f := newFx("clustergate")
	load := f.g.NewMemLoad("ctor", f.out, []ir.Node{f.m})
	cl := f.g.NewIterCluster("ctor", []ir.Node{f.m}, load)
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, cl) // the cluster stays live here
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)

	err := (schedule.Scheduler{}).Run(nil, f.g)
	if err == nil || !strings.Contains(err.Error(), "KindIterCluster") {
		t.Fatalf("Run with a live itercluster = %v, want the raw-HIR-iter guard", err)
	}
	if f.g.AST() != nil {
		t.Fatal("the refused run must not attach an AST")
	}
}

// TestUnreachableForLoopRefusal pins the ME-01 refusal path that was never
// exercised: a non-erased ForLoop that no sea root reaches (building debris
// alongside a valid kernel) is refused fail-closed, not scheduled.
func TestUnreachableForLoopRefusal(t *testing.T) {
	f := newFx("unreach")
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 1))
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)
	// Debris: a plain (unchained) store under a ForLoop nothing roots at.
	iv, bound := f.dimVar()
	f.g.NewMemStore("ctor", f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 2))
	f.g.NewForLoop("ctor", iv, bound)

	err := (schedule.Scheduler{}).Run(nil, f.g)
	if err == nil || !strings.Contains(err.Error(), "not reachable from the sea roots") {
		t.Fatalf("Run with unreachable loop debris = %v, want the unreachable-loop refusal", err)
	}
	if f.g.AST() != nil {
		t.Fatal("the refused run must not attach an AST")
	}
}

// dumpAST renders a scheduled kernel deterministically (block kinds, node
// kinds/IDs/SNs in placement order — no map iteration): the byte-identity
// face for determinism comparisons.
func dumpAST(k *ir.ASTKernel) string {
	var b strings.Builder
	fmt.Fprintf(&b, "lnts=%d gmem=%d params=%d\n", len(k.LNTs), len(k.Gmem), len(k.Params))
	var walk func(blk *ir.ASTBlock, indent string)
	walk = func(blk *ir.ASTBlock, indent string) {
		if blk.Kind() == ir.BlockGroup {
			nestRoot := uint64(0)
			if blk.Nest() != nil {
				nestRoot = blk.Nest().Root.BasePtr().ID()
			}
			fmt.Fprintf(&b, "%sGroup[nestroot=%d]\n", indent, nestRoot)
		} else {
			fmt.Fprintf(&b, "%s%s[loop=%d]\n", indent, blk.Kind(), blk.Loop().BasePtr().ID())
		}
		for _, it := range blk.Items() {
			if it.Node != nil {
				fmt.Fprintf(&b, "%s  %s#%d[sn=%d]\n", indent, it.Node.KindOf(), it.Node.BasePtr().ID(), it.Node.BasePtr().SchedSN())
			} else {
				walk(it.Block, indent+"  ")
			}
		}
	}
	walk(k.Body, "")
	return b.String()
}

// TestSchedStateReset pins LO-01 (quick-261022-b2m): a retried Run never
// inherits poisoned sched state. Pre-poison every node of a fixture with
// SetSched(9, nil), schedule it, and compare the full AST dump against an
// identical clean twin graph's dump — reset + determinism in one pin.
func TestSchedStateReset(t *testing.T) {
	build := func() *fx {
		f := newFx("reset")
		iv, bound := f.dimVar()
		inv := f.g.NewBinOp("ctor", ir.BinOpMul, f.g.NewConst("ctor", f.f32, 3), f.g.NewConst("ctor", f.f32, 4))
		varying := f.g.NewBinOp("ctor", ir.BinOpAdd, iv, inv)
		f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, varying)
		loop := f.g.NewForLoop("ctor", iv, bound, f.loopAttrs(2, 0, 32)...)
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)
		return f
	}
	poisoned, clean := build(), build()
	for _, n := range poisoned.g.AllocatedNodes() {
		n.BasePtr().SetSched(9, nil)
	}
	kp := run(t, poisoned.g)
	kc := run(t, clean.g)
	if dp, dc := dumpAST(kp), dumpAST(kc); dp != dc {
		t.Fatalf("poisoned-retry dump != clean dump:\n--- poisoned ---\n%s\n--- clean ---\n%s", dp, dc)
	}
}

// TestFailClosedGates pins the entry gates: a materialized graph, a graph
// with a live raw KindIter, and a second Run on an already-scheduled graph
// all refuse with deterministic messages.
func TestFailClosedGates(t *testing.T) {
	t.Run("materialized", func(t *testing.T) {
		f := newFx("mat")
		st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 1))
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)
		if err := f.g.Materialize(); err != nil {
			t.Fatal(err)
		}
		err := (schedule.Scheduler{}).Run(nil, f.g)
		if err == nil || !strings.Contains(err.Error(), "materialized") {
			t.Fatalf("Run on a materialized graph = %v, want the materialized guard", err)
		}
		if f.g.AST() != nil {
			t.Fatal("the refused run must not attach an AST")
		}
	})

	t.Run("live iter", func(t *testing.T) {
		f := newFx("iter")
		iv, bound := f.dimVar()
		f.g.NewIter("ctor", bound, nil)
		st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 1))
		_ = iv
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)
		err := (schedule.Scheduler{}).Run(nil, f.g)
		if err == nil || !strings.Contains(err.Error(), "KindIter") {
			t.Fatalf("Run with a live raw iter = %v, want the HIR-iter guard", err)
		}
	})

	t.Run("second run", func(t *testing.T) {
		f := newFx("rerun")
		st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, f.g.NewConst("ctor", f.f32, 1))
		f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, st)
		if err := (schedule.Scheduler{}).Run(nil, f.g); err != nil {
			t.Fatal(err)
		}
		err := (schedule.Scheduler{}).Run(nil, f.g)
		if err == nil || !strings.Contains(err.Error(), "already attached") {
			t.Fatalf("second Run = %v, want the one-schedule-per-graph guard", err)
		}
	})
}

// TestRegistry pins the pipeline contract: "schedule" is registered at
// priority 900 with Requires pinning tile-lower, and the pass's Preserved
// whitelist keeps the tiling-provenance cache entry for final-lower
// (quick-261022-l1o P-5: the old vestigial "tiling-record" entry — a record
// that provably dies at tile-lower — is replaced by the provenance).
func TestRegistry(t *testing.T) {
	spec, ok := opt.Lookup("schedule")
	if !ok {
		t.Fatal("opt.Lookup(\"schedule\") must find the pass")
	}
	if spec.Priority != 900 {
		t.Fatalf("schedule priority = %d, want 900", spec.Priority)
	}
	found := false
	for _, r := range spec.Requires {
		if r == "tile-lower" {
			found = true
		}
	}
	if !found {
		t.Fatalf("schedule Requires = %v, want it to pin tile-lower", spec.Requires)
	}
	p := schedule.Scheduler{}
	preserved := false
	for _, name := range p.Preserved() {
		if name == tiling.ProvenanceName {
			preserved = true
		}
	}
	if !preserved {
		t.Fatalf("schedule Preserved = %v, want it to keep %s", p.Preserved(), tiling.ProvenanceName)
	}
}
