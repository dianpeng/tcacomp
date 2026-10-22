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
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/schedule"
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

// bodyBlocks returns the kernel body's top-level block items (the per-LNT
// [prologue, LBBN root, epilogue] clusters) in order.
func bodyBlocks(t *testing.T, k *ir.ASTKernel) []*ir.ASTBlock {
	t.Helper()
	var out []*ir.ASTBlock
	for _, it := range k.Body.Items() {
		if it.Block == nil {
			t.Fatalf("kernel body item = node %d, want a block cluster", it.Node.BasePtr().ID())
		}
		out = append(out, it.Block)
	}
	return out
}

// TestNoLoops pins spec step 2's degenerate shape: a store-only graph with
// no ForLoop yields a kernel body that is a SINGLE Group holding every
// live node in dependency order, all at SN 0, every node's SchedBlock
// pointing at that group.
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
	items := itemNodes(k.Body)
	if len(items) != 7 { // m, out, a, b, prod, st, end
		t.Fatalf("body holds %d nodes, want every live node (7)", len(items))
	}
	if indexOf(items, a) > indexOf(items, prod) || indexOf(items, b) > indexOf(items, prod) {
		t.Fatal("producers must precede their consumer (dependency order)")
	}
	if indexOf(items, prod) > indexOf(items, st) {
		t.Fatal("the store must follow its value")
	}
	for _, n := range []ir.Node{f.m, f.out, a, b, prod, st, end} {
		if n.BasePtr().SchedSN() != 0 {
			t.Fatalf("node %d SN = %d, want 0", n.BasePtr().ID(), n.BasePtr().SchedSN())
		}
		if n.BasePtr().SchedBlock() != k.Body {
			t.Fatalf("node %d must land in the kernel body group", n.BasePtr().ID())
		}
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
	loop := f.g.NewForLoop("ctor", iv, bound, st, f.loopAttrs(2, 0, 32)...)
	end := f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 3 {
		t.Fatalf("body items = %d blocks, want [prologue, ThrLoop, epilogue]", len(blocks))
	}
	prologue, thr, epilogue := blocks[0], blocks[1], blocks[2]
	if prologue.Kind() != ir.BlockGroup || thr.Kind() != ir.BlockThrLoop || epilogue.Kind() != ir.BlockGroup {
		t.Fatalf("block kinds = [%s %s %s], want [Group ThrLoop Group]",
			prologue.Kind(), thr.Kind(), epilogue.Kind())
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
	innerBound := f.g.NewBoundedDimVar("ctor", innerIV, f.g.NewConst("ctor", f.u32, 8))
	shared := f.g.NewBinOp("ctor", ir.BinOpAdd, innerIV, f.g.NewConst("ctor", f.f32, 1))
	use := f.g.NewBinOp("ctor", ir.BinOpMul, shared, outerIV)
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, use)
	inner := f.g.NewForLoop("ctor", innerIV, innerBound, st, f.loopAttrs(2, 1, 32)...)
	outer := f.g.NewForLoop("ctor", outerIV, outerBound, inner, f.loopAttrs(1, 0, 64)...)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, outer)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 3 {
		t.Fatalf("body items = %d blocks, want [prologue, CTALoop, epilogue]", len(blocks))
	}
	cta := blocks[1]
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
	loopA := f.g.NewForLoop("ctor", ivA, boundA, stA, f.loopAttrs(2, 0, 32)...)

	ivB, boundB := f.dimVar()
	stB := f.g.NewMemStoreChained("ctor", f.env, out2, []ir.Node{f.m}, f.g.NewBinOp("ctor", ir.BinOpAdd, ivB, inv))
	loopB := f.g.NewForLoop("ctor", ivB, boundB, stB, f.loopAttrs(1, 0, 64)...)

	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out, out2}, loopA, loopB)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if len(blocks) != 6 {
		t.Fatalf("body items = %d blocks, want [A-pro, A-LBBN, A-epi, B-pro, B-LBBN, B-epi]", len(blocks))
	}
	aPro, aRoot, aEpi := blocks[0], blocks[1], blocks[2]
	bPro, bRoot, bEpi := blocks[3], blocks[4], blocks[5]
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
	if len(itemNodes(bPro)) != 0 {
		t.Fatal("B's prologue must stay empty (rank-0 nodes are claimed by the first LNT)")
	}
	// CR-02 pin (quick-261022-b2m): the end node's deps span BOTH nests
	// (both roots and both in-loop stores), so it lands in the LAST
	// contributing root's epilogue — B's — keeping SN 1 (its propagated
	// depth requirement; both roots sit at depth 1); A's epilogue stays
	// empty (D-04 landing pads plus the pass-placed after-nest consumers).
	if indexOf(itemNodes(aEpi), f.g.End()) >= 0 {
		t.Fatal("A's epilogue must stay empty (the end node's home is the LAST contributing root)")
	}
	if indexOf(itemNodes(bEpi), f.g.End()) < 0 {
		t.Fatal("the end node must land in B's (the last root's) epilogue Group")
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
	loop := f.g.NewForLoop("ctor", iv, bound, st)
	f.g.NewReturnChained("ctor", f.env, []ir.Node{f.out}, loop)

	k := run(t, f.g)
	blocks := bodyBlocks(t, k)
	if blocks[1].Kind() != ir.BlockSeqLoop {
		t.Fatalf("attrs-less loop block = %s, want SeqLoop", blocks[1].Kind())
	}
	if blocks[1].LoC() != 1 || blocks[1].DoSoC() != -1 {
		t.Fatalf("defaults LoC/DoSoC = %d/%d, want 1/-1", blocks[1].LoC(), blocks[1].DoSoC())
	}
	if indexOf(itemNodes(blocks[1]), st) < 0 {
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
	st := f.g.NewMemStoreChained("ctor", f.env, f.out, []ir.Node{f.m}, load)
	loop := f.g.NewForLoop("ctor", iv, bound, st, f.loopAttrs(2, 1, 32)...)
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
// whitelist keeps the tiling-record cache entry for final_lower.
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
		if name == "tiling-record" {
			preserved = true
		}
	}
	if !preserved {
		t.Fatalf("schedule Preserved = %v, want it to keep tiling-record", p.Preserved())
	}
}
