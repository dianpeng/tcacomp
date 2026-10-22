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
	// its containing block, and no node appears twice anywhere.
	walkItems(k1.Body, func(n ir.Node, containing *ir.ASTBlock) {
		if n.BasePtr().SchedBlock() != containing {
			t.Fatalf("node %d (%s): SchedBlock is not its containing block (linkage broken)",
				n.BasePtr().ID(), n.KindOf())
		}
		if got := countOccurrences(k1.Body, n); got != 1 {
			t.Fatalf("node %d (%s) placed %d times, want exactly 1", n.BasePtr().ID(), n.KindOf(), got)
		}
	})

	// Structure: one LNT; the body's top-level items are exactly
	// [prologue Group, root LBBN block, epilogue Group].
	if len(k1.LNTs) != 1 {
		t.Fatalf("LNTs = %d, want 1", len(k1.LNTs))
	}
	blocks := bodyBlocks(t, k1)
	if len(blocks) != 3 {
		t.Fatalf("body top-level blocks = %d, want [prologue, LBBN, epilogue]", len(blocks))
	}
	if blocks[0].Kind() != ir.BlockGroup || blocks[2].Kind() != ir.BlockGroup {
		t.Fatal("the cluster pads must be Groups")
	}
	if blocks[0].Nest() != k1.LNTs[0] || blocks[2].Nest() != k1.LNTs[0] {
		t.Fatal("the prologue/epilogue Groups must carry the LNT record via Nest()")
	}
	// Loop-kind/depth structure — the observed stable dump: the tile-lowered
	// M→N→K chain materializes as six loops, CTA→Seq→Thr→Thr→Seq→Seq.
	wantKinds := []ir.BlockKind{ir.BlockCTALoop, ir.BlockSeqLoop, ir.BlockThrLoop,
		ir.BlockThrLoop, ir.BlockSeqLoop, ir.BlockSeqLoop}
	var gotKinds []ir.BlockKind
	for b := blocks[1]; b != nil; {
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
		t.Fatalf("Gmem = %d allocations, want 0 (the promoted buffers are SMEM)", len(k1.Gmem))
	}
}

// TestPipelineFillInsideHost is THE CR-01 regression pin
// (quick-261022-b2m): with K=32768 (a multi-round fill-key extent) the SMEM
// fill store lands INSIDE its K-between host loop — SN 3, a Loop-kind block
// whose loop also sits at SN 3 — re-executing once per round (R1's
// once-per-round mandate). The old operand-only clamp hoisted it to the
// once-only prologue at SN 0, leaving rounds 2+ reading a stale SMEM window.
// The fill's source load reloads per M-round (SN 1) and the SMEM buffer
// declaration lands at SN 0 in the FIRST prologue (the allocation
// exemption); the end node lands in the LAST LNT's epilogue.
func TestPipelineFillInsideHost(t *testing.T) {
	g := matmulGraphK("mm-k32768", 32768)
	k := runPipeline(t, g)

	// Exactly one live SMEM fill (lhs promoted, rhs declined by the searched
	// defaults).
	var fill ir.Node
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStoreSMem {
			continue
		}
		if fill != nil {
			t.Fatalf("two live SMEM fills (%d and %d), want exactly one", fill.BasePtr().ID(), n.BasePtr().ID())
		}
		fill = n
	}
	if fill == nil {
		t.Fatal("no live SMEM fill found — the searched defaults must promote lhs")
	}

	// The fill sits INSIDE its K-between host loop (SN 3, Loop-kind block,
	// the host loop also at SN 3) — never in the prologue.
	if got := fill.BasePtr().SchedSN(); got != 3 {
		t.Fatalf("fill SN = %d, want 3 (inside the K-between host loop; the old clamp gave 0)", got)
	}
	blk := fill.BasePtr().SchedBlock()
	if blk == nil || blk.Kind() == ir.BlockGroup {
		t.Fatalf("fill block kind = %v, want a Loop block", blkKindOf(blk))
	}
	if host := blk.Loop(); host == nil || host.BasePtr().SchedSN() != 3 {
		t.Fatalf("the fill's host loop must also sit at SN 3 (the K-between host)")
	}

	// Every live data dep of the fill is at SN <= 3 (executed before it).
	for _, d := range ir.DataDeps(fill) {
		if d == nil || d.BasePtr().Erased() {
			continue
		}
		if got := d.BasePtr().SchedSN(); got > 3 {
			t.Fatalf("fill dep node %d (%s) at SN %d, want <= 3", d.BasePtr().ID(), d.KindOf(), got)
		}
	}

	// Source tile load: per-M-round reload at SN 1.
	ops := ir.OperandsOf(fill)
	if len(ops) < 2 || ops[0] == nil || ops[0].KindOf() != ir.KindMemLoad {
		t.Fatalf("fill ops[0] = %v, want the KindMemLoad source tile load", kindOf(ops[0]))
	}
	if got := ops[0].BasePtr().SchedSN(); got != 1 {
		t.Fatalf("fill source load SN = %d, want 1 (per-M-round reload)", got)
	}

	// SMEM buffer: the allocation-exemption pin — SN 0 in the FIRST
	// prologue Group.
	if len(ops) < 2 || ops[1] == nil || ops[1].KindOf() != ir.KindMemNewSMem {
		t.Fatalf("fill ops[1] = %v, want the KindMemNewSMem buffer", kindOf(ops[1]))
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
