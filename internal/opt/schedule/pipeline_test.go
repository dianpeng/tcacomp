package schedule_test

// pipeline_test.go is the committed full-pipeline integration test
// (quick-261022-b2m task 4, locked decision 4 + ME-01): the matmul worked
// example through the DEFAULT pipeline (opt.BuildPipeline — tiling →
// tile-lower → schedule, imported via internal/opt/all), converted from the
// 261021-ta3 review's scratch probe. Three pins:
//
//   - TestPipelineMatmulSchedule: the pipeline succeeds on the matmul shape,
//     is DETERMINISTIC (byte-identical dumpAST across twin runs), places
//     every walked node exactly once with correct SchedBlock linkage, and
//     the kernel structure/params/gmem/lnts counts match the observed
//     stable dump.
//   - TestPipelineFillInsideHost: THE CR-01 regression — with K=32768 the
//     SMEM fill store lands INSIDE its K-between host loop (SN 3), never in
//     the prologue (the old clamp hoisted it to SN 0 → stale SMEM windows →
//     wrong numerics once final_lower consumes the AST).
//   - TestPipelineEndNodeEpilogue: the kernel end node lands in the LAST
//     LNT's epilogue Group keeping its propagated SN, with the owning nest
//     recoverable via Nest() (rulings 6-7).
//
// The dumpAST helper lives in scheduler_test.go (same package). All
// assertions walk placement order — never map order.

import (
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	_ "tcacomp/internal/opt/all"
	"tcacomp/internal/opt/tiling" // the preserved provenance record's name + type (sibling opt package, data only)
)

// matmulGraphK builds the review probe's matmul shape with the third chain
// bound parameterized: u32 extents 1024/1024/kExtent over the M→N→K iter
// chain, InductionVariable-wired lhs[m][k] and rhs[n][k] loads, and a
// chained store of lhs*rhs over [m,n,k]. The searched tiling defaults
// promote lhs (SMEM window) and decline rhs (ReasonLowReuse) — exactly one
// fill exists per run.
func matmulGraphK(name string, kExtent float64) *ir.Graph {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "d0", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d1", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d2", nil),
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d2"}}))
	rhs := g.NewInputParam("ctor", ir.RoleTensor, "rhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d1"}, {Name: "d2"}}))
	var parent ir.Node
	iters := make([]ir.Node, 3)
	for i, extent := range []float64{1024, 1024, kExtent} {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extent))
		parent = g.NewIter("ctor", biv, parent)
		iters[i] = parent
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }
	l := g.NewMemLoad("ctor", lhs, []ir.Node{iv(0), iv(2)}) // lhs[m][k]
	r := g.NewMemLoad("ctor", rhs, []ir.Node{iv(1), iv(2)}) // rhs[n][k]
	prod := g.NewBinOp("ctor", ir.BinOpMul, l, r)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1), iv(2)}, prod)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return g
}

// runPipeline runs the full default pipeline over g and fails the test on
// any error, returning the attached AST.
func runPipeline(t *testing.T, g *ir.Graph) *ir.ASTKernel {
	t.Helper()
	if err := opt.NewManager(opt.BuildPipeline()).Run(g); err != nil {
		t.Fatalf("default pipeline run = %v", err)
	}
	k := g.AST()
	if k == nil {
		t.Fatal("the pipeline must attach the schedule pass's AST")
	}
	return k
}

// walkItems walks every statement item of blk (recursing into Block items)
// in placement order, invoking fn(node, containingBlock).
func walkItems(blk *ir.ASTBlock, fn func(n ir.Node, containing *ir.ASTBlock)) {
	for _, it := range blk.Items() {
		if it.Node != nil {
			fn(it.Node, blk)
			continue
		}
		if it.Block != nil {
			walkItems(it.Block, fn)
		}
	}
}

// TestPipelineMatmulSchedule pins the full default pipeline over the matmul
// worked example: success, DETERMINISM (byte-identical AST dumps across
// twin graphs), single placement with correct SchedBlock linkage for every
// walked node, and the observed stable kernel structure (one LNT; the
// six-loop CTA→Seq→Thr→Thr→Seq→Seq chain; 6 params; no GMEM temporary).
func TestPipelineMatmulSchedule(t *testing.T) {
	k1 := runPipeline(t, matmulGraphK("mm", 1024))
	k2 := runPipeline(t, matmulGraphK("mm", 1024))
	if d1, d2 := dumpAST(k1), dumpAST(k2); d1 != d2 {
		t.Fatalf("twin pipeline runs differ:\n--- run 1 ---\n%s\n--- run 2 ---\n%s", d1, d2)
	}

	// Linkage + single placement: every walked node's SchedBlock is exactly
	// its containing block, and no node appears twice anywhere. The walk
	// covers the WHOLE kernel — Prototype (quick-261022-e2u) and Body — and
	// pins the placeholder anchoring: every live InputParam at SN -1 in the
	// Prototype block, whose items are exactly the live params.
	walkItems(k1.Body, func(n ir.Node, containing *ir.ASTBlock) {
		if n.BasePtr().SchedBlock() != containing {
			t.Fatalf("node %d (%s): SchedBlock is not its containing block (linkage broken)",
				n.BasePtr().ID(), n.KindOf())
		}
		if got := countOccurrences(k1.Body, n); got != 1 {
			t.Fatalf("node %d (%s) placed %d times, want exactly 1", n.BasePtr().ID(), n.KindOf(), got)
		}
	})
	if k1.Prototype.Kind() != ir.BlockPrototype {
		t.Fatalf("prototype kind = %s, want Prototype", k1.Prototype.Kind())
	}
	walkItems(k1.Prototype, func(n ir.Node, containing *ir.ASTBlock) {
		if n.KindOf() != ir.KindInputParam {
			t.Fatalf("prototype item %d is %s, want an InputParam placeholder", n.BasePtr().ID(), n.KindOf())
		}
		if n.BasePtr().SchedSN() != ir.PrototypeSN || n.BasePtr().SchedBlock() != k1.Prototype {
			t.Fatalf("param %d must anchor at SN -1 in the Prototype block, got %d/%p",
				n.BasePtr().ID(), n.BasePtr().SchedSN(), n.BasePtr().SchedBlock())
		}
		if got := countOccurrences(k1.Body, n); got != 0 {
			t.Fatalf("param %d appears %d times in the body — placeholders live only in the Prototype", n.BasePtr().ID(), got)
		}
	})
	if got := len(itemNodes(k1.Prototype)); got != len(k1.Params) {
		t.Fatalf("prototype items = %d, want exactly the %d live params", got, len(k1.Params))
	}

	// Structure: one LNT; the body's top-level block items are exactly ONE
	// lexical Group embedding the LBBN root (quick-261023-lex).
	if len(k1.LNTs) != 1 {
		t.Fatalf("LNTs = %d, want 1", len(k1.LNTs))
	}
	blocks := bodyBlocks(t, k1)
	if len(blocks) != 1 {
		t.Fatalf("body top-level blocks = %d, want ONE lexical Group per LNT", len(blocks))
	}
	if blocks[0].Kind() != ir.BlockGroup {
		t.Fatal("the LNT cluster must be a lexical Group")
	}
	if blocks[0].Nest() != k1.LNTs[0] {
		t.Fatal("the lexical Group must carry the LNT record via Nest()")
	}
	// Loop-kind/depth structure — the observed stable dump: the tile-lowered
	// M→N→K chain materializes as six loops, CTA→Seq→Thr→Thr→Seq→Seq
	// (the C1b forced-FRAG precedence's optimum assigns both N and K
	// sequential — the single Thr loop is M's coupled inside walk).
	wantKinds := []ir.BlockKind{ir.BlockCTALoop, ir.BlockSeqLoop, ir.BlockThrLoop,
		ir.BlockThrLoop, ir.BlockSeqLoop, ir.BlockSeqLoop}
	var gotKinds []ir.BlockKind
	for b := rootBlockOf(t, blocks[0]); b != nil; {
		gotKinds = append(gotKinds, b.Kind())
		next := (*ir.ASTBlock)(nil)
		for _, it := range b.Items() { // nests are chains: at most one child block
			if it.Block != nil {
				if next != nil {
					t.Fatalf("loop block at depth %d has multiple child blocks", len(gotKinds))
				}
				next = it.Block
			}
		}
		b = next
	}
	if len(gotKinds) != len(wantKinds) {
		t.Fatalf("loop chain length = %d (%v), want %d", len(gotKinds), gotKinds, len(wantKinds))
	}
	for i, want := range wantKinds {
		if gotKinds[i] != want {
			t.Fatalf("loop chain[%d] = %s, want %s (chain: %v)", i, gotKinds[i], want, gotKinds)
		}
	}
	// Kernel metadata counts — the observed stable values.
	if len(k1.Params) != 6 {
		t.Fatalf("Params = %d, want 6 (3 dims + 3 tensors)", len(k1.Params))
	}
	if len(k1.Gmem) != 0 {
		t.Fatalf("Gmem = %d allocations, want 0 (the promoted buffers are on-chip: SMEM windows / per-thread fragments)", len(k1.Gmem))
	}
}

// TestPipelineFillInsideHost is THE CR-01 regression pin
// (quick-261022-b2m): with K=32768 (a multi-round fill-key extent) the fill
// store lands INSIDE its host loop — SN 4, a ThrLoop-kind block whose loop
// also sits at SN 4 — re-executing once per host trip (R1's once-per-host
// mandate), never hoisted to the once-only prologue at SN 0 (the old
// operand-only clamp left later trips reading a stale window). Under the
// C1b forced-FRAG precedence (2026-10-23) the fill is a per-thread
// FragStore (LoadTileThread): its host is M's coupled inside walk, so each
// thread re-fills its K fragment once per row trip. The fill's source load
// reloads per trip (SN 4), its K-between round base rides SN 3, and the
// fragment buffer declaration lands at SN 0 in the FIRST prologue (the
// allocation exemption); the end node lands in the LAST LNT's epilogue.
func TestPipelineFillInsideHost(t *testing.T) {
	g := matmulGraphK("mm-k32768", 32768)
	k := runPipeline(t, g)

	// Exactly one live fragment fill (lhs promoted, rhs declined by the
	// searched defaults).
	var fill ir.Node
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc != ir.StorageFrag {
			continue
		}
		if fill != nil {
			t.Fatalf("two live fragment fills (%d and %d), want exactly one", fill.BasePtr().ID(), n.BasePtr().ID())
		}
		fill = n
	}
	if fill == nil {
		t.Fatal("no live fragment fill found — the searched defaults must promote lhs")
	}

	// The fill sits INSIDE its host loop (SN 4, ThrLoop-kind block, the host
	// loop also at SN 4) — never in the prologue.
	if got := fill.BasePtr().SchedSN(); got != 4 {
		t.Fatalf("fill SN = %d, want 4 (inside the coupled-walk host loop; the old clamp gave 0)", got)
	}
	blk := fill.BasePtr().SchedBlock()
	if blk == nil || blk.Kind() == ir.BlockGroup {
		t.Fatalf("fill block kind = %v, want a ThrLoop block", blkKindOf(blk))
	}
	if host := blk.Loop(); host == nil || host.BasePtr().SchedSN() != 4 {
		t.Fatalf("the fill's host loop must also sit at SN 4 (the coupled-walk host)")
	}

	// Every live data dep of the fill is at SN <= 4 (executed before it).
	for _, d := range ir.DataDeps(fill) {
		if d == nil || d.BasePtr().Erased() {
			continue
		}
		if got := d.BasePtr().SchedSN(); got > 4 {
			t.Fatalf("fill dep node %d (%s) at SN %d, want <= 4", d.BasePtr().ID(), d.KindOf(), got)
		}
	}

	// Source tile load: per-trip reload at SN 4.
	ops := ir.OperandsOf(fill)
	if len(ops) < 2 || ops[0] == nil || ops[0].KindOf() != ir.KindMemLoad {
		t.Fatalf("fill ops[0] = %v, want the KindMemLoad source tile load", kindOf(ops[0]))
	}
	if got := ops[0].BasePtr().SchedSN(); got != 4 {
		t.Fatalf("fill source load SN = %d, want 4 (per-trip reload)", got)
	}

	// Fragment buffer: the allocation-exemption pin — SN 0 in the FIRST
	// prologue Group.
	if len(ops) < 2 || ops[1] == nil || ops[1].KindOf() != ir.KindMemNew {
		t.Fatalf("fill ops[1] = %v, want the FRAG-class MemNew buffer", kindOf(ops[1]))
	}
	buf := ops[1]
	if got := buf.BasePtr().SchedSN(); got != 0 {
		t.Fatalf("buffer SN = %d, want 0 (allocation exemption, first prologue)", got)
	}
	firstPro := bodyBlocks(t, k)[0]
	if buf.BasePtr().SchedBlock() != firstPro {
		t.Fatal("the buffer must land in the FIRST prologue Group")
	}

	// The end node lands in the LAST LNT's epilogue Group.
	blocks := bodyBlocks(t, k)
	lastEpi := blocks[len(blocks)-1]
	if g.End().BasePtr().SchedBlock() != lastEpi {
		t.Fatal("the end node must land in the LAST LNT's epilogue Group")
	}
	if nst := lastEpi.Nest(); nst != k.LNTs[len(k.LNTs)-1] {
		t.Fatal("the last epilogue Group's Nest() must be the last LNT record")
	}
}

// TestPipelineEndNodeEpilogue pins rulings 6-7 on the K=1024 run shape: the
// end node's block is the last cluster's epilogue Group, its SN equals its
// propagated depth requirement (the deepest live dep — the in-loop output
// store at SN 6; the observed stable value), and the owning nest is
// recoverable via g.End().SchedBlock().Nest() matching k.LNTs' last entry
// by Root identity.
func TestPipelineEndNodeEpilogue(t *testing.T) {
	g := matmulGraphK("mm-end", 1024)
	k := runPipeline(t, g)
	end := g.End()
	if end == nil {
		t.Fatal("the graph must carry an end node")
	}
	blocks := bodyBlocks(t, k)
	lastEpi := blocks[len(blocks)-1]
	if lastEpi.Kind() != ir.BlockGroup {
		t.Fatalf("last body block kind = %s, want the epilogue Group", lastEpi.Kind())
	}
	if end.BasePtr().SchedBlock() != lastEpi {
		t.Fatal("the end node must land in the last cluster's epilogue Group (CR-02 fix)")
	}
	if got := end.BasePtr().SchedSN(); got != 6 {
		t.Fatalf("end node SN = %d, want 6 (its propagated depth requirement — the deepest live dep, ruling 6)", got)
	}
	nst := end.BasePtr().SchedBlock().Nest()
	if nst == nil || nst.Root.BasePtr() != k.LNTs[len(k.LNTs)-1].Root.BasePtr() {
		t.Fatal("the end node's epilogue must identify the last LNT via Nest() by Root identity (ruling 7)")
	}
}

// TestPipelinePreservesTilingProvenance pins P-5's preservation chain
// (quick-261022-l1o): tile-lower publishes the tiling provenance under
// tiling.ProvenanceName and its own Preserved keeps it alive past its run;
// schedule's Preserved [tiling.ProvenanceName] keeps it alive past
// scheduling — after a pipeline run THROUGH schedule the manager's cache
// still holds the *Provenance, one per-LNT plan keyed by each scheduled
// nest's root loop, ready for final-lower's fail-closed fetch. (The prefix
// stops at schedule: final-lower CONSUMES the provenance and — running
// last with Preserved nil — would legitimately drop it from the cache.)
func TestPipelinePreservesTilingProvenance(t *testing.T) {
	g := matmulGraphK("mm-prov", 1024)
	var prefix []opt.Pass
	for _, p := range opt.BuildPipeline() {
		if p.ID() == "final-lower" {
			break
		}
		prefix = append(prefix, p)
	}
	pm := opt.NewManager(prefix)
	if err := pm.Run(g); err != nil {
		t.Fatalf("pipeline-through-schedule run = %v", err)
	}
	k := g.AST()
	if k == nil {
		t.Fatal("the pipeline must attach the schedule pass's AST")
	}
	got := pm.Cache().Get(tiling.ProvenanceName)
	if got == nil {
		t.Fatalf("the tiling provenance must survive scheduling through the Preserved chain (tile-lower + schedule)")
	}
	prov, ok := got.(*tiling.Provenance)
	if !ok {
		t.Fatalf("cache entry under %q is %T, want *tiling.Provenance", tiling.ProvenanceName, got)
	}
	if len(k.LNTs) == 0 {
		t.Fatal("the matmul fixture must schedule at least one LNT")
	}
	if len(prov.LNTs) != len(k.LNTs) {
		t.Fatalf("provenance plans = %d, want one per scheduled LNT (%d)", len(prov.LNTs), len(k.LNTs))
	}
	for _, lnt := range k.LNTs {
		plan, ok := prov.LNTs[lnt.Root]
		if !ok {
			t.Fatalf("LNT root node %d carries no mapping plan", lnt.Root.BasePtr().ID())
		}
		if len(plan.GridPortions) != 1 || plan.GridPortions[0] <= 0 {
			t.Fatalf("grid portions = %v, want [positive CTA LoC] (fix4 absolute-shape model)", plan.GridPortions)
		}
		if len(plan.ThreadPortions) != 1 || plan.ThreadPortions[0] <= 0 {
			t.Fatalf("thread portions = %v, want [positive warp K] (fix4 absolute-shape model)", plan.ThreadPortions)
		}
	}
}

// blkKindOf renders a block's kind nil-safely for failure messages.
func blkKindOf(b *ir.ASTBlock) ir.BlockKind {
	if b == nil {
		return ir.BlockKind(-1)
	}
	return b.Kind()
}

// kindOf renders a node's kind nil-safely for failure messages.
func kindOf(n ir.Node) ir.Kind {
	if n == nil {
		return ir.KindInvalid
	}
	return n.KindOf()
}
