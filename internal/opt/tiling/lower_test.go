package tiling

// lower_test.go pins the loop-tile-lower pass (quick-261021-jay +
// 261021-o61 T3, doc/optana/loop_tile_lower.md as amended): the fail-closed
// record gates (incl. the F7 tiling-verify stamp), the §2 DoSoC numbering,
// the PHASE 3 canonical loop forms (SEQ/CTA/WARP, T==1 and T>1, the F1
// saturating tail guard), the F3 branched-LNT rejection, the LO-05 extent
// typing guard, the M1 annotations, the P6 chain re-pointing liveness, the
// pass determinism, and the PHASE 4 site surgery (variant-dependent fill
// placement per R1, R2 window buffers + rank-in-index accesses, budget
// fallback with the F4 drop rule) including the matmul worked example and
// the decide→verify→lower composition.

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/typeinfer"
)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

// newChain32 builds an n-deep iter chain like newChain but with u32 consts
// throughout — the tile-lower pass's umin bounds type-check under a
// typeinfer re-run only when the extents are u32 (the real pipeline's
// extents are TensorDimOf nodes, which type u32).
func newChain32(name string, maxes ...float64) fixture {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	var fx fixture
	fx.g = g
	for i := range maxes {
		fx.dims = append(fx.dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	fx.out = g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	var parent ir.Node
	for i := range maxes {
		dv := g.NewDimVar("ctor", fx.dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, maxes[i]))
		parent = g.NewIter("ctor", biv, parent)
		fx.iters = append(fx.iters, parent)
	}
	return fx
}

// bodyStore wires the chain fixture's innermost effectful body: a store
// over all the chain's IVs (axes reference every iter, so the store is the
// deepest body op), chained into the return over `out`.
func bodyStore(fx fixture) ir.Node {
	env := ir.NewEnv()
	var axes []ir.Node
	for _, it := range fx.iters {
		axes = append(axes, iv(fx.g, it))
	}
	st := fx.g.NewMemStoreChained("ctor", env, fx.out, axes,
		fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, st)
	return st
}

// recordOf crafts a decide-pass record directly (the search is pinned
// elsewhere): every iter's decision per the given table, no sites unless
// provided. bivOf resolves the record's BIV keys. The record carries the
// tiling-verify stamp (F7) — the crafted fixtures stand in for a verified
// pipeline product.
func recordOf(t *testing.T, fx fixture, decisions map[int]IterDecision, sites map[uint64]SiteDecision) *Record {
	t.Helper()
	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: sites, Verified: true}
	for i, it := range fx.iters {
		d, ok := decisions[i]
		if !ok {
			t.Fatalf("fixture decision table misses iter %d", i)
		}
		biv, err := bivOf(it)
		if err != nil {
			t.Fatal(err)
		}
		rec.Iters[biv] = d
	}
	if rec.Sites == nil {
		rec.Sites = make(map[uint64]SiteDecision)
	}
	return rec
}

// runLower runs the pass over g with a crafted record and default params.
func runLower(t *testing.T, g *ir.Graph, rec *Record) {
	t.Helper()
	runLowerP(t, g, rec, DefaultParams())
}

// runLowerP runs the pass over g with a crafted record and explicit
// params. The budget-sensitive crafted fixtures (matmulRecord's
// WARP-covered rhs) need a MaxCache that fits the R2 WINDOW pricing —
// resolveBudget prices that site at 4 x (8 warps x 32) x 32 x 32 = 1 MB,
// so the 64 KB default budget would (correctly, per the rulings) halve
// and drop it.
func runLowerP(t *testing.T, g *ir.Graph, rec *Record, p Params) {
	t.Helper()
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g); err != nil {
		t.Fatalf("tile-lower Run: %v", err)
	}
}

// loopsOf collects the live for-loops in allocation order.
func loopsOf(g *ir.Graph) []ir.Node {
	var out []ir.Node
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindForLoop {
			out = append(out, n)
		}
	}
	return out
}

// nestOf returns the live for-loops in NEST order (outermost first),
// following body operands. The pass builds nests inside-out, so allocation
// order is the reverse — tests reason in dim order.
func nestOf(t *testing.T, g *ir.Graph) []ir.Node {
	t.Helper()
	loops := loopsOf(g)
	isBody := make(map[*ir.Base]bool, len(loops))
	for _, l := range loops {
		if body := ir.OperandsOf(l)[2]; body != nil && body.KindOf() == ir.KindForLoop {
			isBody[body.BasePtr()] = true
		}
	}
	var root ir.Node
	for _, l := range loops {
		if !isBody[l.BasePtr()] {
			if root != nil {
				t.Fatalf("multiple nest roots: %d and %d", root.BasePtr().ID(), l.BasePtr().ID())
			}
			root = l
		}
	}
	if root == nil {
		t.Fatal("no nest root found")
	}
	var out []ir.Node
	for n := root; n != nil && n.KindOf() == ir.KindForLoop; n = ir.OperandsOf(n)[2] {
		out = append(out, n)
	}
	if len(out) != len(loops) {
		t.Fatalf("nest walk covers %d of %d loops (branched nest?)", len(out), len(loops))
	}
	return out
}

// dimvarStep reads a loop's iv dimvar step const.
func dimvarStep(t *testing.T, loop ir.Node) float64 {
	t.Helper()
	iv := ir.OperandsOf(loop)[0]
	step := ir.OperandsOf(iv)[2]
	v, ok := ir.ConstValueOf(step)
	if !ok {
		t.Fatalf("loop %d dimvar step is not a const", loop.BasePtr().ID())
	}
	return v
}

// boundOf reads a loop's bound operand (the BoundedDimVar) and returns its
// max operand.
func boundOf(t *testing.T, loop ir.Node) ir.Node {
	t.Helper()
	bdv := ir.OperandsOf(loop)[1]
	ops := ir.OperandsOf(bdv)
	if len(ops) < 2 {
		t.Fatalf("loop %d bound carries no max operand", loop.BasePtr().ID())
	}
	return ops[1]
}

// renderGraph renders the live graph deterministically (allocation order:
// kind + operand IDs) for equality comparison across runs.
func renderGraph(g *ir.Graph) string {
	var b strings.Builder
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		fmt.Fprintf(&b, "%d:%s", n.BasePtr().ID(), n.KindOf())
		if field := ir.IntrinsicFieldOf(n); field != "" {
			fmt.Fprintf(&b, "(%s)", field)
		}
		for _, op := range ir.OperandsOf(n) {
			if op == nil {
				b.WriteString(" nil")
			} else {
				fmt.Fprintf(&b, " %d", op.BasePtr().ID())
			}
		}
		for _, u := range ir.ImpactBy(n) {
			if d := u.Def(); d != nil {
				fmt.Fprintf(&b, " @%d", d.BasePtr().ID())
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Task 2: gates, numbering, rewrite
// ---------------------------------------------------------------------------

// TestLowerFailClosed pins the §6bis PHASE 1 gate: a missing record, a
// wrong-typed cache entry, a nil context, and an UNVERIFIED record (F7) are
// named errors, not silent skips; Requires pins the §1 record lifetime +
// the F7 tiling-verify precondition.
func TestLowerFailClosed(t *testing.T) {
	fx := newChain32("gates", 64)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 1},
	}, nil)

	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: opt.NewCache()}, fx.g); err == nil ||
		!strings.Contains(err.Error(), "no tiling record") {
		t.Fatalf("missing record must fail closed naming the record, got %v", err)
	}
	cache := opt.NewCache()
	cache.Put(fakeAnalysis("not a record"))
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g); err == nil ||
		!strings.Contains(err.Error(), "want *Record") {
		t.Fatalf("wrong-typed record must fail closed, got %v", err)
	}
	if err := (LowerPass{Params: DefaultParams()}).Run(nil, fx.g); err == nil {
		t.Fatal("nil context must fail closed")
	}
	// F7: an unstamped record must never reach the rewrite.
	rec.Verified = false
	unstamped := opt.NewCache()
	unstamped.Put(rec)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: unstamped}, fx.g); err == nil ||
		!strings.Contains(err.Error(), "tiling-verify") {
		t.Fatalf("an unverified record must fail closed naming tiling-verify (F7), got %v", err)
	}
	if got := (LowerPass{}).Requires(); len(got) != 2 || got[0] != "tiling" || got[1] != "tiling-verify" {
		t.Fatalf("Requires = %v, want [tiling tiling-verify]", got)
	}
	spec, ok := opt.Lookup("tile-lower")
	if !ok || spec.Priority != 116 || len(spec.Requires) != 2 || spec.Requires[0] != "tiling" || spec.Requires[1] != "tiling-verify" {
		t.Fatalf("registry spec = %+v, want tile-lower@116 requiring tiling + tiling-verify", spec)
	}
	// Sanity: with the stamped record present the same graph lowers clean.
	rec.Verified = true
	runLower(t, fx.g, rec)
}

// numberedFx builds the numbering fixture: one CTA dim (outer), one
// independent WARP dim, one sequential dim, plus a SECOND LNT (a separate
// chain) whose counters must restart.
func numberedFx(t *testing.T) (fixture, fixture, *Record) {
	fxA := newChain32("numA", 1024, 2048, 512)
	bodyStore(fxA)
	fxB := newChain32("numB", 128)
	bodyStore(fxB)
	rec := recordOf(t, fxA, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
	bivB, err := bivOf(fxB.iters[0])
	if err != nil {
		t.Fatal(err)
	}
	rec.Iters[bivB] = IterDecision{SoC: SoCTA, LoC: 100, TileSize: 16}
	return fxA, fxB, rec
}

// TestNumbering pins §2: the CTA dim gets GridMapping dim 0; the CTA dim's
// coupled warp and the independent warp dim get DISTINCT ThreadMap dims
// (coupled 0, independent 1); the sequential dim gets none; the second
// LNT restarts both counters.
func TestNumbering(t *testing.T) {
	fxA, fxB, rec := numberedFx(t)
	runLower(t, fxA.g, rec)
	runLower(t, fxB.g, rec)

	// Dim order outer→inner: M(CTA), N(WARP), K(SEQ). All betweens nest
	// in dim order, all insides inside them in dim order.
	loops := nestOf(t, fxA.g)
	if len(loops) != 6 {
		t.Fatalf("3 tiled dims must yield 6 loops, got %d", len(loops))
	}
	mB, nB, kB, mI, nI, kI := loops[0], loops[1], loops[2], loops[3], loops[4], loops[5]
	if got := loopAnnotations(mB); got["soc"] != 1 || got["dosoc"] != 0 || got["loc"] != 100 {
		t.Fatalf("M between annotations = %v, want soc=1 dosoc=0 loc=100", got)
	}
	if got := loopAnnotations(mI); got["soc"] != 2 || got["dosoc"] != 0 || got["loc"] != 256 {
		t.Fatalf("M inside (coupled walk) annotations = %v, want soc=2 dosoc=0 loc=256", got)
	}
	if got := loopAnnotations(nB); got["soc"] != 2 || got["dosoc"] != 1 || got["loc"] != 256 {
		t.Fatalf("N between annotations = %v, want soc=2 dosoc=1 loc=256", got)
	}
	if got := loopAnnotations(nI); got["soc"] != 0 || got["loc"] != 1 {
		t.Fatalf("N inside annotations = %v, want soc=0 loc=1", got)
	}
	if got := loopAnnotations(kB); got["soc"] != 0 || got["loc"] != 1 {
		t.Fatalf("K between annotations = %v, want soc=0 loc=1", got)
	}
	if got := loopAnnotations(kI); got["soc"] != 0 || got["loc"] != 1 {
		t.Fatalf("K inside annotations = %v, want soc=0 loc=1", got)
	}
	for _, l := range []ir.Node{nI, kB, kI} {
		if _, ok := loopAnnotations(l)["dosoc"]; ok {
			t.Fatalf("sequential loop %d must carry no dosoc", l.BasePtr().ID())
		}
	}

	// Strides: M between LoC×T = 3200; M inside = coupled 256; N between
	// (LoC×32)×T = 8192; N inside = 1; K between = T = 32.
	for _, tc := range []struct {
		loop ir.Node
		want float64
	}{
		{mB, 3200}, {mI, 256}, {nB, 8192}, {nI, 1}, {kB, 32}, {kI, 1},
	} {
		if got := dimvarStep(t, tc.loop); got != tc.want {
			t.Fatalf("loop %d step = %v, want %v", tc.loop.BasePtr().ID(), got, tc.want)
		}
	}

	// The second LNT restarts: its single CTA dim gets dosoc 0 again.
	loopsB := nestOf(t, fxB.g)
	if len(loopsB) != 2 {
		t.Fatalf("one tiled dim must yield 2 loops, got %d", len(loopsB))
	}
	if got := loopAnnotations(loopsB[0]); got["soc"] != 1 || got["dosoc"] != 0 {
		t.Fatalf("second LNT's CTA between annotations = %v, want soc=1 dosoc=0 (restart)", got)
	}
}

// TestRewriteSeqDim pins the SEQ template: between (step T, bound = raw
// extent), inside bounded by the materialized umin tile_max; the original
// induction variable has zero live uses and the original iter is erased.
func TestRewriteSeqDim(t *testing.T) {
	fx2 := newChain32("seq2", 512)
	bodyStore(fx2)
	rec2 := recordOf(t, fx2, map[int]IterDecision{
		0: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
	extent2 := bivMaxOf(t, fx2.iters[0]) // captured pre-erase
	runLower(t, fx2.g, rec2)

	loops := nestOf(t, fx2.g)
	if len(loops) != 2 {
		t.Fatalf("SEQ T>1 must yield between+inside, got %d loops", len(loops))
	}
	between, inside := loops[0], loops[1]
	if got := dimvarStep(t, between); got != 32 {
		t.Fatalf("SEQ between step = %v, want 32 (T)", got)
	}
	// Between bound = the raw extent (the original BIV max operand).
	if c, ok := ir.ConstValueOf(boundOf(t, between)); !ok || c != 512 {
		t.Fatalf("SEQ between bound = %v, want the raw extent const 512", boundOf(t, between))
	}
	// Inside bound = the SATURATING umin form (F1): a Call node named
	// "umin" wrapping (T, extent − umin(offset, extent)).
	assertSaturatingBound(t, fx2.g, boundOf(t, inside), 32, extent2, between)
	// Originals retired.
	for _, it := range fx2.iters {
		if !it.BasePtr().Erased() {
			t.Fatalf("iter node %d must be erased", it.BasePtr().ID())
		}
	}
	// Zero live induction variables anywhere.
	for _, n := range fx2.g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindInductionVariable {
			t.Fatalf("induction variable node %d survived", n.BasePtr().ID())
		}
	}
	// Graph health: verifier clean; the nest root reachable from the end.
	assertVerifyAndRootReachable(t, fx2.g)
}

// bivMaxOf resolves an iter's BIV max operand (the extent node).
func bivMaxOf(t *testing.T, iter ir.Node) ir.Node {
	t.Helper()
	biv, err := bivOf(iter)
	if err != nil {
		t.Fatal(err)
	}
	ops := ir.DataDeps(biv)
	if len(ops) < 2 {
		t.Fatalf("BIV of iter %d carries no max", iter.BasePtr().ID())
	}
	return ops[1]
}

// assertSaturatingBound pins the F1 SATURATING materialized bound:
//
//	umin(T_const, extent − umin(offset, extent))
//
// — the plain `umin(T, extent − offset)` WRAPS in u32 on tail rounds
// (CR-01): an out-of-range rank computes extent − offset near 4e9, the min
// picks T, and the inside loop runs its body out of bounds instead of
// no-op'ing. The structural walk asserts the clamp is present on the
// offset BEFORE the subtract, for BOTH the CTA and WARP offset forms.
func assertSaturatingBound(t *testing.T, g *ir.Graph, bound ir.Node, tile float64, extent ir.Node, offsetCarrier ir.Node) {
	t.Helper()
	if bound.KindOf() != ir.KindCall || ir.CallNameOf(bound) != "umin" {
		t.Fatalf("materialized bound must be a umin call, got %s (node %d)", bound.KindOf(), bound.BasePtr().ID())
	}
	outer := ir.OperandsOf(bound)
	if len(outer) != 2 {
		t.Fatalf("umin bound must be binary, got %d operands", len(outer))
	}
	if v, ok := ir.ConstValueOf(outer[0]); !ok || v != tile {
		t.Fatalf("umin bound operand 0 must be the tile const %v, got %v", tile, outer[0])
	}
	subOps0 := ir.OperandsOf(bound)[1]
	if subOps0.KindOf() != ir.KindBinOp {
		t.Fatalf("umin bound operand 1 must be a sub, got %v", subOps0)
	}
	sub := subOps0
	if code, _ := ir.BinOpCodeOf(sub); code != ir.BinOpSub {
		t.Fatalf("umin bound operand 1 must be a sub, got code %v", code)
	}
	subOps := ir.OperandsOf(sub)
	if subOps[0].BasePtr() != extent.BasePtr() {
		t.Fatalf("sub operand 0 must be the raw extent node %d, got %d", extent.BasePtr().ID(), subOps[0].BasePtr().ID())
	}
	inner := subOps[1]
	if inner.KindOf() != ir.KindCall || ir.CallNameOf(inner) != "umin" {
		t.Fatalf("sub operand 1 must be the SATURATING clamp umin(offset, extent) — the plain CR-01 form (extent − offset) wraps in u32; got %s", inner.KindOf())
	}
	innerOps := ir.OperandsOf(inner)
	if innerOps[1].BasePtr() != extent.BasePtr() {
		t.Fatalf("clamp operand 1 must be the raw extent node %d", extent.BasePtr().ID())
	}
	// The clamped offset is the carrier's index machinery: the in-body
	// offset expression (CTA/WARP rank form) or the between dimvar itself.
	if offsetCarrier != nil && !subtreeContains(innerOps[0], ir.OperandsOf(offsetCarrier)[0]) {
		t.Fatalf("clamp operand 0 must reference the offset carrier %d's dimvar", offsetCarrier.BasePtr().ID())
	}
}

// subtreeContains reports whether n's data subtree contains target.
func subtreeContains(n, target ir.Node) bool {
	if n == nil || target == nil {
		return false
	}
	found := false
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || found || x.BasePtr() == nil || x.BasePtr().Erased() {
			return
		}
		if x.BasePtr() == target.BasePtr() {
			found = true
			return
		}
		for _, dep := range ir.DataDeps(x) {
			visit(dep)
		}
	}
	visit(n)
	return found
}

// TestRewriteTailSaturatingGuard is the fixture that would have caught
// CR-01: extents NOT multiples of the round strides (M=1000 CTA with
// LoC·T = 3200; N=500 WARP with LoC·32·T = 8192) — every materialized
// inside bound must be the saturating umin(T, extent − umin(offset,
// extent)) form, for the CTA offset (grid-rank form) and the WARP offset
// (thread-rank form) alike.
func TestRewriteTailSaturatingGuard(t *testing.T) {
	fx := newChain32("tail", 1000, 500)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
	}, nil)
	mExtent := bivMaxOf(t, fx.iters[0]) // captured pre-erase
	nExtent := bivMaxOf(t, fx.iters[1])
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g) // M.b, N.b, M.i, N.i
	if len(loops) != 4 {
		t.Fatalf("two T>1 dims must yield 4 loops, got %d", len(loops))
	}
	mB, nB, mI, nI := loops[0], loops[1], loops[2], loops[3]
	assertSaturatingBound(t, fx.g, boundOf(t, mI), 32, mExtent, mB)
	assertSaturatingBound(t, fx.g, boundOf(t, nI), 32, nExtent, nB)
	// The clamped offsets are the RANK forms: the M inside bound's clamp
	// references the GridMap rank, the N inside bound's the ThreadMap rank.
	mClamp := ir.OperandsOf(ir.OperandsOf(boundOf(t, mI))[1])[1]
	if !subtreeHasKind(t, fx.g, mClamp, ir.KindIntrinsicGridMap) {
		t.Fatal("the CTA inside bound must clamp the grid-rank offset form")
	}
	nClamp := ir.OperandsOf(ir.OperandsOf(boundOf(t, nI))[1])[1]
	if !subtreeHasKind(t, fx.g, nClamp, ir.KindIntrinsicThreadMap) {
		t.Fatal("the WARP inside bound must clamp the thread-rank offset form")
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteCTADim pins the CTA template: between step LoC×T; the in-body
// GridMap rank expression feeds offset = rank×T + round; the inside loop
// strides the coupled thread count; the index expression chains
// offset + inside + coupled thread rank.
func TestRewriteCTADim(t *testing.T) {
	fx := newChain32("cta", 4096)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
	}, nil)
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g)
	if len(loops) != 2 {
		t.Fatalf("CTA T>1 must yield between+inside, got %d loops", len(loops))
	}
	between, inside := loops[0], loops[1]
	if got := dimvarStep(t, between); got != 3200 {
		t.Fatalf("CTA between step = %v, want LoC×T = 3200", got)
	}
	if got := dimvarStep(t, inside); got != 256 {
		t.Fatalf("CTA inside step = %v, want coupled threads = 256", got)
	}
	// The store's axis (its only operand tree referencing the loop) must
	// resolve to offset + inside + thread_rank: chase it down to a
	// GridMap and a ThreadMap intrinsic.
	axes := ir.OperandsOf(store)[2:]
	if len(axes) != 1 {
		t.Fatalf("store must carry one rewritten axis, got %d", len(axes))
	}
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindIntrinsicGridMap) {
		t.Fatal("rewritten index must contain the GridMap rank query")
	}
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindIntrinsicThreadMap) {
		t.Fatal("rewritten index must contain the coupled ThreadMap rank query")
	}
	// dosoc consts: grid dim 0 (GridMap) and thread dim 0 (ThreadMap).
	assertDosocConst(t, axes[0])
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteWarpDim pins the WARP template: between step (LoC×32)×T;
// in-body ThreadMap rank; inside sequential over the owned chunk bounded
// by the materialized tile_max.
func TestRewriteWarpDim(t *testing.T) {
	fx := newChain32("warp", 4096)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCWARP, LoC: 8, TileSize: 32},
	}, nil)
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g)
	if len(loops) != 2 {
		t.Fatalf("WARP T>1 must yield between+inside, got %d loops", len(loops))
	}
	between, inside := loops[0], loops[1]
	if got := dimvarStep(t, between); got != 8192 {
		t.Fatalf("WARP between step = %v, want (LoC×32)×T = 8192", got)
	}
	if got := dimvarStep(t, inside); got != 1 {
		t.Fatalf("WARP inside must be sequential (step 1), got %v", got)
	}
	if got := loopAnnotations(inside)["soc"]; got != 0 {
		t.Fatalf("WARP inside soc = %v, want 0 (sequential chunk walk)", got)
	}
	axes := ir.OperandsOf(store)[2:]
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindIntrinsicThreadMap) {
		t.Fatal("rewritten WARP index must contain the ThreadMap rank query")
	}
	if subtreeHasKind(t, fx.g, axes[0], ir.KindIntrinsicGridMap) {
		t.Fatal("a WARP-only LNT must contain no GridMap query")
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteTileOne pins the T==1 form: the dim keeps a SINGLE annotated
// loop and its IV uses map directly to the loop index expression.
func TestRewriteTileOne(t *testing.T) {
	fx := newChain32("tileone", 1024, 512)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 1},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 1},
	}, nil)
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g)
	if len(loops) != 2 {
		t.Fatalf("two T==1 dims must yield exactly two loops, got %d", len(loops))
	}
	// Nest: M loop body = K loop; K loop body = store.
	if got := ir.OperandsOf(loops[0])[2]; got == nil || got.BasePtr() != loops[1].BasePtr() {
		t.Fatalf("outer loop body must be the inner loop")
	}
	if got := ir.OperandsOf(loops[1])[2]; got == nil || got.BasePtr() != store.BasePtr() {
		t.Fatalf("inner loop body must be the store")
	}
	// The K axis maps DIRECTLY to the K loop's dimvar.
	axes := ir.OperandsOf(store)[2:]
	kAxis := axes[1]
	if kAxis == nil || kAxis.KindOf() != ir.KindDimVar {
		t.Fatalf("T==1 SEQ axis must be the loop dimvar directly, got %s", kAxis.KindOf())
	}
	if kAxis.BasePtr() != ir.OperandsOf(loops[1])[0].BasePtr() {
		t.Fatal("T==1 axis must be the inner loop's own dimvar")
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteBranchedLNTRejected pins F3/HI-05: a branched LNT (sibling
// iters under one parent) is REJECTED fail-closed — never silently
// linearized into one nested chain (the siblings' spaces would execute as
// a cross product).
func TestRewriteBranchedLNTRejected(t *testing.T) {
	g := ir.NewGraph("branched")
	u32 := ir.InternScalar(ir.TU32)
	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "d0", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d1", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d2", nil),
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	mk := func(i int, max float64) ir.Node {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, max))
		return biv
	}
	m := g.NewIter("ctor", mk(0, 1024), nil)
	k1 := g.NewIter("ctor", mk(1, 512), m) // sibling 1 under M
	k2 := g.NewIter("ctor", mk(2, 512), m) // sibling 2 under M
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(g, k1), iv(g, k2)},
		g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: map[uint64]SiteDecision{}, Verified: true}
	for _, it := range []ir.Node{m, k1, k2} {
		biv, err := bivOf(it)
		if err != nil {
			t.Fatal(err)
		}
		rec.Iters[biv] = IterDecision{SoC: SoCSequential, LoC: 1, TileSize: 32}
	}
	cache := opt.NewCache()
	cache.Put(rec)
	err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g)
	if err == nil || !strings.Contains(err.Error(), "branched") {
		t.Fatalf("a branched LNT must be rejected fail-closed naming the branching, got %v", err)
	}
}

// TestRewriteExtentTypeGuard pins LO-05: a non-u32 BIV max (an f32 extent,
// exactly what the decide-half newChain fixture builds) fails closed with
// a named error instead of emitting ill-typed umin/sub nodes.
func TestRewriteExtentTypeGuard(t *testing.T) {
	fx := newChain("f32ext", 64) // f32 maxes — the wrong extent type
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
	}, nil)
	cache := opt.NewCache()
	cache.Put(rec)
	err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g)
	if err == nil || !strings.Contains(err.Error(), "non-u32 extent") {
		t.Fatalf("a non-u32 extent must fail closed naming the u32 requirement (LO-05), got %v", err)
	}
}

// TestRewriteDeterminism pins §7: two fresh copies through the pass render
// byte-identically (allocation-order walks, no map iteration in emission).
func TestRewriteDeterminism(t *testing.T) {
	fxA := newChain32("detA", 1024, 2048, 512)
	bodyStore(fxA)
	fxB := newChain32("detB", 1024, 2048, 512)
	bodyStore(fxB)
	recA := recordOf(t, fxA, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
	recB := recordOf(t, fxB, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
	runLower(t, fxA.g, recA)
	runLower(t, fxB.g, recB)
	// Node IDs shift by construction order (identical here); compare the
	// renders modulo the graph name — they must be byte-identical.
	if gotA, gotB := renderGraph(fxA.g), renderGraph(fxB.g); gotA != gotB {
		t.Fatalf("two runs must render byte-identically:\n--- A ---\n%s\n--- B ---\n%s", gotA, gotB)
	}
}

// TestRewriteTypeinferRerun pins P8: a typeinfer re-run over the rewritten
// graph succeeds (the PHASE 0 rules — u32 intrinsics, the umin builtin —
// make the pass's own products typable).
func TestRewriteTypeinferRerun(t *testing.T) {
	fx := newChain32("ti", 1024, 2048, 512)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
	runLower(t, fx.g, rec)
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run over the rewritten graph must succeed (P8): %v", err)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// subtreeHasKind reports whether n's data subtree contains a live node of
// kind k.
func subtreeHasKind(t *testing.T, g *ir.Graph, n ir.Node, k ir.Kind) bool {
	t.Helper()
	found := false
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || found || x.BasePtr() == nil || x.BasePtr().Erased() {
			return
		}
		if x.KindOf() == k {
			found = true
			return
		}
		for _, dep := range ir.DataDeps(x) {
			visit(dep)
		}
	}
	visit(n)
	return found
}

// assertDosocConst pins that every intrinsic in the subtree carries a
// const-u32 DoSoC operand (the PHASE 0 shape in the generated product).
func assertDosocConst(t *testing.T, n ir.Node) {
	t.Helper()
	for _, x := range flattenSubtree(n) {
		if x.KindOf() != ir.KindIntrinsicGridMap && x.KindOf() != ir.KindIntrinsicThreadMap {
			continue
		}
		d := ir.OperandsOf(x)[0]
		if d == nil || d.KindOf() != ir.KindConst {
			t.Fatalf("intrinsic node %d carries a non-const DoSoC", x.BasePtr().ID())
		}
		if ty := ir.ConstTypeOf(d); ty == nil || !ty.Equal(ir.InternScalar(ir.TU32)) {
			t.Fatalf("intrinsic node %d DoSoC is not u32", x.BasePtr().ID())
		}
	}
}

// assertVerifyAndRootReachable pins the graph-health contract: ir.Verify
// reports zero violations AND the rewritten nest root is reachable from the
// end node (the P6 chain re-pointing).
func assertVerifyAndRootReachable(t *testing.T, g *ir.Graph) {
	t.Helper()
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("verifier must report zero violations, got %v", vs)
	}
	// Reachability: walk from the end node's operand+payload edges and
	// require at least one live for-loop in the visited set.
	var end ir.Node
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindReturn {
			end = n
		}
	}
	if end == nil {
		t.Fatal("fixture carries no return node")
	}
	seen := make(map[*ir.Base]bool)
	var visit func(n ir.Node)
	visit = func(n ir.Node) {
		if n == nil || n.BasePtr() == nil || seen[n.BasePtr()] || n.BasePtr().Erased() {
			return
		}
		seen[n.BasePtr()] = true
		for _, dep := range ir.DataDeps(n) {
			visit(dep)
		}
		for _, u := range ir.ImpactBy(n) {
			visit(u.Def())
		}
	}
	visit(end)
	loops := 0
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindForLoop && seen[n.BasePtr()] {
			loops++
		}
	}
	if loops == 0 {
		t.Fatal("no generated loop is reachable from the end node (P6 chain re-pointing broken)")
	}
}

// sortedKeys sorts a site-decision map's keys (deterministic test iteration).
func sortedKeys(m map[uint64]SiteDecision) []uint64 {
	out := make([]uint64, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// fakeAnalysis is a wrong-typed cache entry for the fail-closed cast test.
type fakeAnalysis string

func (f fakeAnalysis) Name() string { return RecordName }

// ---------------------------------------------------------------------------
// Task 3: cache insertion — worked example, window access, budget, checks
// ---------------------------------------------------------------------------

// matmulUsed builds the doc's worked example with the loads FEEDING the
// store (real data flow — the accesses anchor the fills through the
// effect chains): M→N→K chain, lhs[m][k], rhs[n][k],
// out[m][n][k] = lhs[m][k] * rhs[n][k].
func matmulUsed() (fx fixture, lhs, rhs, store ir.Node) {
	fx = newChain32("mm", 1024, 2048, 512) // M, N, K
	lhs = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[2])})
	rhs = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "rhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	prod := fx.g.NewBinOp("ctor", ir.BinOpMul, lhs, rhs)
	env := ir.NewEnv()
	axes := []ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])}
	store = fx.g.NewMemStoreChained("ctor", env, fx.out, axes, prod)
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, store)
	return fx, lhs, rhs, store
}

// matmulRecord crafts the decide record for matmulUsed per the R2 tier
// rule: M=CTA(T32), N=WARP(T32), K=SEQ(T32); lhs tie=N covering [·][K] —
// all covered dims sequential → FRAG class (LoadTileThread); rhs tie=nil
// covering [N][K] — N is WARP-assigned → SMEM window class (LoadTileCta).
func matmulRecord(t *testing.T, fx fixture, lhs, rhs ir.Node) *Record {
	t.Helper()
	return recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, map[uint64]SiteDecision{
		lhs.BasePtr().ID(): {Promoted: true, Tie: fx.iters[1], Class: ClassFRAG},
		rhs.BasePtr().ID(): {Promoted: true, Tie: nil, Class: ClassSMEM},
	})
}

// findBuffer returns the unique live buffer allocation of the given kind
// whose extent equals wantBytes, or fails.
func findBuffer(t *testing.T, g *ir.Graph, kind ir.Kind, wantBytes float64) ir.Node {
	t.Helper()
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != kind {
			continue
		}
		if v, ok := ir.ConstValueOf(ir.OperandsOf(n)[0]); ok && v == wantBytes {
			return n
		}
	}
	t.Fatalf("no live %s buffer with extent %v", kind, wantBytes)
	return nil
}

// findFillOf returns the unique live fill store targeting buf.
func findFillOf(t *testing.T, g *ir.Graph, buf ir.Node) ir.Node {
	t.Helper()
	var fill ir.Node
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() != ir.KindMemStoreSMem && n.KindOf() != ir.KindFragStore {
			continue
		}
		if dst := ir.OperandsOf(n)[1]; dst != nil && dst.BasePtr() == buf.BasePtr() {
			if fill != nil {
				t.Fatalf("two fills target buffer %d", buf.BasePtr().ID())
			}
			fill = n
		}
	}
	if fill == nil {
		t.Fatalf("no fill targets buffer %d", buf.BasePtr().ID())
	}
	return fill
}

// loadtileVariantOf parses a fill store's live loadtile annotation.
func loadtileVariantOf(t *testing.T, fill ir.Node) float64 {
	t.Helper()
	ops := ir.OperandsOf(fill)
	for i := 2; i < len(ops); i++ {
		op := ops[i]
		if op == nil || op.KindOf() != ir.KindKeyedAttribute {
			continue
		}
		if key, val := ir.KeyedAttributeOf(op); key == "loadtile" {
			v, ok := ir.ConstValueOf(val)
			if !ok {
				t.Fatalf("fill %d carries a non-const loadtile value", fill.BasePtr().ID())
			}
			return v
		}
	}
	t.Fatalf("fill %d carries no loadtile annotation", fill.BasePtr().ID())
	return 0
}

// TestLowerMatmulEndToEnd pins the PHASE 4 surgery over the doc's worked
// example (the R2 accepted outcome, both sites promoted under a fitting
// budget): lhs as a FRAG site (128 B per-thread FragNew buffer,
// LoadTileThread fill INSIDE M.inside) and rhs as an SMEM WINDOW site
// (1 MB MemNewSMem window — 4 B × 256 threads × 32 × 32 — cooperative
// LoadTileCta fill at the K between-round body): loads erased with uses
// RAUW'd to buffer accesses, verify-clean, typeinfer re-run clean, fill
// placement structurally sound.
func TestLowerMatmulEndToEnd(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	p := DefaultParams()
	p.MaxCache = 1 << 21 // fit the crafted rhs's R2 window (1 MB) — no halving
	runLowerP(t, fx.g, rec, p)

	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("both promoted loads must be erased")
	}
	// Buffers: lhs FRAG [T_k] = 4×32 = 128 bytes per thread; rhs SMEM
	// window [256×T_n][T_k] = 4×256×32×32 = 1 MB (the R2 group window).
	lbuf := findBuffer(t, fx.g, ir.KindFragNew, 128)
	rbuf := findBuffer(t, fx.g, ir.KindMemNewSMem, 1048576)
	if ir.TypeOf(lbuf) == nil || ir.TypeOf(rbuf) == nil {
		t.Fatal("buffers must carry their bound tile tensor types")
	}
	// The store's value subtree reads the buffers (access rewrite): lhs's
	// access is a FragLoad, rhs's a MemLoadSMem.
	var fragAccess, smemAccess ir.Node
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		switch n.KindOf() {
		case ir.KindFragLoad:
			fragAccess = n
			if src := ir.OperandsOf(n)[0]; src.BasePtr() != lbuf.BasePtr() {
				t.Fatalf("FRAG access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
			}
		case ir.KindMemLoadSMem:
			smemAccess = n
			if src := ir.OperandsOf(n)[0]; src.BasePtr() != rbuf.BasePtr() {
				t.Fatalf("SMEM access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
			}
		}
	}
	if fragAccess == nil || smemAccess == nil {
		t.Fatal("the multiply must read exactly one FragLoad and one MemLoadSMem buffer access")
	}
	// R2 access index: rhs's N coordinate carries thread_map.index(1)·T + i
	// (N's ThreadMap slot — NOT the hardcoded 0), K coordinates are plain
	// inside walks.
	smemOps := ir.OperandsOf(smemAccess)[1:]
	if len(smemOps) != 2 {
		t.Fatalf("the rhs access must carry 2 coordinates, got %d", len(smemOps))
	}
	if !subtreeHasKind(t, fx.g, smemOps[0], ir.KindIntrinsicThreadMap) {
		t.Fatal("the rhs access's N coordinate must carry the ThreadMap rank query (R2 rank-in-index)")
	}
	if subtreeHasKind(t, fx.g, smemOps[1], ir.KindIntrinsicThreadMap) {
		t.Fatal("the rhs access's K coordinate (SEQ covered) must be the plain inside walk")
	}
	// Fill variants: lhs FRAG → 2 (LoadTileThread); rhs SMEM → 1
	// (LoadTileCta).
	if v := loadtileVariantOf(t, findFillOf(t, fx.g, lbuf)); v != 2 {
		t.Fatalf("lhs fill variant = %v, want 2 (LoadTileThread)", v)
	}
	if v := loadtileVariantOf(t, findFillOf(t, fx.g, rbuf)); v != 1 {
		t.Fatalf("rhs fill variant = %v, want 1 (LoadTileCta)", v)
	}
	// Loads-only: the out store survives as a GMEM MemStore.
	if store.BasePtr().Erased() || store.KindOf() != ir.KindMemStore {
		t.Fatal("stores never route through the cache: the GMEM store must survive")
	}
	// Graph health + P8 + fill anchoring through the effect chains.
	assertVerifyAndRootReachable(t, fx.g)
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run over the lowered matmul must succeed (P8): %v", err)
	}
	assertReachable(t, fx.g, ir.KindMemStoreSMem, 1)
	assertReachable(t, fx.g, ir.KindFragStore, 1)
}

// TestMatmulWorkedCheck pins the R1 worked example: the lhs FRAG fill sits
// structurally INSIDE M.inside — its m coordinate is the enclosing m index
// expression (the index expressions defined AT that position), its k
// coordinate is the K between loop's round-base dimvar (refill per k_o
// round; the m-element epoch is the fill key that re-executes the fill),
// and the access reads the FRAG buffer at the plain k inside walk.
func TestMatmulWorkedCheck(t *testing.T) {
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	p := DefaultParams()
	p.MaxCache = 1 << 21 // fit the crafted rhs's R2 window (1 MB) — no halving
	runLowerP(t, fx.g, rec, p)

	lbuf := findBuffer(t, fx.g, ir.KindFragNew, 128)
	lhsFill := findFillOf(t, fx.g, lbuf)
	src := ir.OperandsOf(lhsFill)[0] // the tile-shaped source MemLoad
	axes := ir.OperandsOf(src)[1:]
	if len(axes) != 1 {
		t.Fatalf("lhs fill source must be the 1-axis partial deref lhs[m], got %d axes", len(axes))
	}
	// The m coordinate is the enclosing index expression: it references
	// the launch-map machinery (M is the CTA dim), not a bare buffer loop.
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindIntrinsicGridMap) {
		t.Fatal("the lhs fill's m coordinate must be the enclosing m index expression")
	}
	// The fill store's covered round base is the K between loop's dimvar.
	nest := nestOf(t, fx.g) // M.b, N.b, K.b, M.i, N.i, K.i
	kBetweenIV := ir.OperandsOf(nest[2])[0]
	mInside := nest[3]
	fillBivs := ir.OperandsOf(lhsFill)[2:]
	if len(fillBivs) != 2 { // covered round base + the loadtile annotation
		t.Fatalf("lhs fill must carry its covered round base plus the variant annotation, got %d trailing operands", len(fillBivs))
	}
	if fillBivs[0].BasePtr() != kBetweenIV.BasePtr() {
		t.Fatal("the lhs fill's round base must be the K between loop's dimvar (filled per k_o round)")
	}
	// STRUCTURAL PLACEMENT (R1/F8): the fill is reachable from M.inside's
	// BODY operand subtree (data + effect edges) — the fill key position —
	// and, being a per-thread fill, is exempt from the element-loop rule.
	lr := &lowerRun{g: fx.g}
	body := ir.OperandsOf(mInside)[2]
	if !lr.reachableFrom(body)[lhsFill.BasePtr()] {
		t.Fatal("the lhs fill must be data-reachable from the M tile-inside loop's body operand (the R1 fill-key position)")
	}
	// Out-store control: the GMEM store stays an operand-descendant of the
	// M.inside body subtree (the whole compute nest is controlled by it).
	if !lr.operandSubtreeOf(body)[storeOf(t, fx.g).BasePtr()] {
		t.Fatal("the out store must remain inside the M tile-inside loop's body subtree (out-store control)")
	}
}

// storeOf returns the graph's unique live GMEM MemStore.
func storeOf(t *testing.T, g *ir.Graph) ir.Node {
	t.Helper()
	var st ir.Node
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindMemStore {
			if st != nil {
				t.Fatal("fixture carries more than one GMEM store")
			}
			st = n
		}
	}
	if st == nil {
		t.Fatal("fixture carries no GMEM store")
	}
	return st
}

// smemWindowFx builds the SMALL-shape SMEM-window fixture: M(CTA LoC=100
// T=8), N(WARP LoC=2 = 64 threads T=8), K(SEQ T=8) — the R2 window on rhs's
// covered {N,K} is 4 × 64 × 8 × 8 = 16384 B, comfortably inside the 64 KB
// default budget, so the cooperative window path lowers un-halved. A second
// co-traversed load satisfies the overlap admission filter (set size >= 2)
// and stays declined (GMEM).
func smemWindowFx(t *testing.T) (fx fixture, rhs, other, store ir.Node) {
	t.Helper()
	fx = newChain32("smemwin", 128, 128, 64) // M, N, K
	rhs = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "rhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	other = fx.g.NewMemLoad("ctor", tensorParam(fx.g, "other", 2),
		[]ir.Node{iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])})
	prod := fx.g.NewBinOp("ctor", ir.BinOpMul, rhs, other)
	env := ir.NewEnv()
	axes := []ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[1]), iv(fx.g, fx.iters[2])}
	store = fx.g.NewMemStoreChained("ctor", env, fx.out, axes, prod)
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, store)
	return fx, rhs, other, store
}

// TestLowerSMEMWindow pins the R2 SMEM-window path (HI-03): a
// window-shaped buffer (#threads_d × T_d on the WARP dim), a cooperative
// LoadTileCta fill at the INNERMOST COVERED DIM's between-round body with
// CTA-UNIFORM bases only (no per-thread term), and the rank-in-index
// access thread_map.index(d)·T + i on the WARP dim.
func TestLowerSMEMWindow(t *testing.T) {
	fx, rhs, other, store := smemWindowFx(t)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 8},
		1: {SoC: SoCWARP, LoC: 2, TileSize: 8},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 8},
	}, map[uint64]SiteDecision{
		rhs.BasePtr().ID():   {Promoted: true, Tie: nil, Class: ClassSMEM},
		other.BasePtr().ID(): {Promoted: false, Reason: ReasonLowReuse},
	})
	runLower(t, fx.g, rec) // default 64 KB budget fits the 16384 B window

	if !rhs.BasePtr().Erased() {
		t.Fatal("the promoted rhs load must be erased")
	}
	if other.BasePtr().Erased() {
		t.Fatal("the declined co-traversed load must stay GMEM")
	}
	// Window-shaped buffer: 4 B × (64 threads × 8) × 8 = 16384 B.
	buf := findBuffer(t, fx.g, ir.KindMemNewSMem, 16384)
	ty, ok := ir.TypeOf(buf).(*ir.NumericType)
	if !ok || ty.Rank() != 2 {
		t.Fatalf("the rhs window buffer must be a rank-2 tensor type, got %v", ir.TypeOf(buf))
	}
	// The cooperative fill: variant 1 (LoadTileCta), at the K between-round
	// body — CTA-uniform round bases only.
	fill := findFillOf(t, fx.g, buf)
	if v := loadtileVariantOf(t, fill); v != 1 {
		t.Fatalf("rhs fill variant = %v, want 1 (LoadTileCta)", v)
	}
	src := ir.OperandsOf(fill)[0]
	for _, ax := range ir.OperandsOf(src)[1:] {
		if subtreeHasKind(t, fx.g, ax, ir.KindIntrinsicThreadMap) {
			t.Fatal("the cooperative fill's source coordinates must be CTA-uniform (no ThreadMap term, R1/HI-03)")
		}
	}
	nest := nestOf(t, fx.g) // M.b, N.b, K.b, M.i, N.i, K.i
	if len(nest) != 6 {
		t.Fatalf("3 tiled dims must yield 6 loops, got %d", len(nest))
	}
	kBetween := nest[2]
	nBetweenIV := ir.OperandsOf(nest[1])[0]
	kBetweenIV := ir.OperandsOf(kBetween)[0]
	// The covered round bases ride the fill's trailing bivs: N and K
	// between dimvars (uniform), plus the annotation.
	bases := ir.OperandsOf(fill)[2 : len(ir.OperandsOf(fill))-1]
	if len(bases) != 2 || bases[0].BasePtr() != nBetweenIV.BasePtr() || bases[1].BasePtr() != kBetweenIV.BasePtr() {
		t.Fatalf("the rhs fill's round bases must be the N,K between dimvars, got %v", bases)
	}
	// STRUCTURAL PLACEMENT: the fill is reachable from the K between
	// loop's body operand (the mandated SMEM position) and is NOT an
	// operand-descendant of any element (inside) loop's body — §4
	// layering, the HI-02 class.
	lr := &lowerRun{g: fx.g}
	if !lr.reachableFrom(ir.OperandsOf(kBetween)[2])[fill.BasePtr()] {
		t.Fatal("the cooperative fill must be data-reachable from the K between loop's body operand (R1 SMEM position)")
	}
	for _, l := range []ir.Node{nest[3], nest[4], nest[5]} { // M.i, N.i, K.i
		if lr.operandSubtreeOf(ir.OperandsOf(l)[2])[fill.BasePtr()] {
			t.Fatalf("a cooperative fill must never sit inside element loop %d's body operand subtree (§4 layering)", l.BasePtr().ID())
		}
	}
	// The access: N coordinate = thread_map.index(N's slot)·T + i with the
	// N dim's OWN ThreadMap slot (1, not the hardcoded 0), K coordinate =
	// the plain inside walk.
	var access ir.Node
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() == ir.KindMemLoadSMem {
			access = n
		}
	}
	if access == nil {
		t.Fatal("the store's value must read the window buffer through a MemLoadSMem access")
	}
	coords := ir.OperandsOf(access)[1:]
	if len(coords) != 2 {
		t.Fatalf("the window access must carry 2 coordinates, got %d", len(coords))
	}
	nCoord := coords[0]
	var dosoc float64
	var haveRank bool
	for _, n := range flattenSubtree(nCoord) {
		if n.KindOf() == ir.KindIntrinsicThreadMap {
			haveRank = true
			if v, ok := ir.ConstValueOf(ir.OperandsOf(n)[0]); ok {
				dosoc = v
			}
		}
	}
	if !haveRank || dosoc != 1 {
		t.Fatalf("the N access coordinate must query thread_map.index(1) (N's ThreadMap slot), got rank=%v dosoc=%v", haveRank, dosoc)
	}
	if got := flattenSubtree(nCoord)[0]; got.KindOf() != ir.KindBinOp {
		t.Fatalf("the N access coordinate must be an add (rank·T + i), got %s", got.KindOf())
	} else if code, _ := ir.BinOpCodeOf(got); code != ir.BinOpAdd {
		t.Fatalf("the N access coordinate must be an add (rank·T + i), got code %v", code)
	}
	assertVerifyAndRootReachable(t, fx.g)
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run over the SMEM-window fixture must succeed (P8): %v", err)
	}
}

// TestLowerDefaultShapeDecline pins the R2 accepted consequence: at the
// default worked-example SHAPE, the crafted rhs's SMEM window can never fit
// a tight budget (its minimum window is 4 × 256 × 1 × 1 = 1024 B), so the
// F4 fallback halve/drops it — both sites stay GMEM, their loads untouched.
func TestLowerDefaultShapeDecline(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	p := DefaultParams()
	p.MaxCache = 512 // below rhs's minimum window (1024 B): decline
	runLowerP(t, fx.g, rec, p)

	if lhs.BasePtr().Erased() || rhs.BasePtr().Erased() {
		t.Fatal("under a budget below the minimum window both sites must decline back to GMEM")
	}
	// No cache material was emitted at all.
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		switch n.KindOf() {
		case ir.KindMemNewSMem, ir.KindFragNew, ir.KindMemStoreSMem, ir.KindFragStore,
			ir.KindMemLoadSMem, ir.KindFragLoad:
			t.Fatalf("a declined kernel must carry no cache nodes; found %s node %d", n.KindOf(), n.BasePtr().ID())
		}
	}
	// The store's value still reads the ORIGINAL loads (untouched GMEM).
	liveLoads := 0
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() == ir.KindMemLoad {
			liveLoads++
		}
	}
	if liveLoads != 2 {
		t.Fatalf("the original GMEM accesses must be untouched, found %d live loads in the store's value", liveLoads)
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestLowerBodyRootIsLoad pins the DELETED HI-02 path through the standard
// splice: when the promoted load IS the LNT's body root (the innermost
// effectful op referencing the iters), the RAUW moves the innermost loop's
// body slot onto the buffer ACCESS — never onto the fill — and the fill
// lands at its variant-mandated position.
func TestLowerBodyRootIsLoad(t *testing.T) {
	fx := newChain32("brl", 1024, 512) // M, K
	lhs1 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs1", 2),
		[]ir.Node{fx.g.NewConst("ctor", ir.InternScalar(ir.TU32), 0), iv(fx.g, fx.iters[1])})
	lhs2 := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs2", 2),
		[]ir.Node{fx.g.NewConst("ctor", ir.InternScalar(ir.TU32), 0), iv(fx.g, fx.iters[1])})
	prod := fx.g.NewBinOp("ctor", ir.BinOpMul, lhs1, lhs2)
	env := ir.NewEnv()
	store := fx.g.NewMemStoreChained("ctor", env, fx.out, nil, prod) // references NO iter: the LOADS are the body roots
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, store)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, map[uint64]SiteDecision{
		lhs1.BasePtr().ID(): {Promoted: false, Reason: ReasonLowReuse},
		lhs2.BasePtr().ID(): {Promoted: true, Tie: nil, Class: ClassFRAG},
	})
	runLower(t, fx.g, rec)

	if !lhs2.BasePtr().Erased() {
		t.Fatal("the promoted body-root load must be erased")
	}
	if lhs1.BasePtr().Erased() {
		t.Fatal("the declined load must stay GMEM")
	}
	buf := findBuffer(t, fx.g, ir.KindFragNew, 128) // 4 B × T_k(32)
	fill := findFillOf(t, fx.g, buf)
	// No loop's body operand is the fill — the HI-02 shape is gone.
	for _, l := range loopsOf(fx.g) {
		if body := ir.OperandsOf(l)[2]; body != nil && body.BasePtr() == fill.BasePtr() {
			t.Fatalf("the fill must never BE a loop's body operand (the deleted HI-02 path placed a fill inside an element loop); loop %d", l.BasePtr().ID())
		}
	}
	// The K inside loop's body slot moved onto the buffer ACCESS (the
	// standard splice: RAUW of the body-root load).
	nest := nestOf(t, fx.g) // M.b, K.b, M.i, K.i
	kInside := nest[3]
	body := ir.OperandsOf(kInside)[2]
	if body == nil || body.KindOf() != ir.KindFragLoad {
		t.Fatalf("the K inside loop's body must be the buffer access after the RAUW, got %v", body)
	}
	if src := ir.OperandsOf(body)[0]; src.BasePtr() != buf.BasePtr() {
		t.Fatal("the body access must read the site's own buffer")
	}
	// The fill sits at its mandated host (no above-tie INS dim exists:
	// the innermost covered dim's between body) — reachable from there.
	lr := &lowerRun{g: fx.g}
	if !lr.reachableFrom(ir.OperandsOf(nest[1])[2])[fill.BasePtr()] {
		t.Fatal("the fill must be data-reachable from the K between loop's body operand (the fallback fill position)")
	}
	if vs := ir.Verify(fx.g); len(vs) != 0 {
		t.Fatalf("verifier must report zero violations, got %v", vs)
	}
}

// TestBudgetFallback pins §6's fallback (F4 + kernel-wide budgets): with a
// forced-tiny MaxCache the halving loop terminates — tile sizes halve while
// over budget; a site whose ANY covered iter reaches tile 1 drops back to
// GMEM; with a budget that fits the R2 window pricing the fallback is a
// no-op; the FRAG budget (F6) participates through the same loop.
func TestBudgetFallback(t *testing.T) {
	// No-op case: a budget fitting both sites' R2 windows leaves the
	// record untouched.
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	pNoop := DefaultParams()
	pNoop.MaxCache = 1 << 21
	runLowerP(t, fx.g, rec, pNoop)
	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("with a fitting budget both sites must promote")
	}

	// Halving survival: tile-8 decisions under a 4104-byte budget halve
	// the rhs window to (2,2) — 4·256·2·2 = 4096 B — while the lhs FRAG
	// site halves to 8 B; both stay promoted (the FRAG sum 8 is well
	// inside MaxFragPerThread).
	fx2, lhs2, rhs2, _ := matmulUsed()
	rec2 := matmulRecord(t, fx2, lhs2, rhs2)
	for biv := range rec2.Iters {
		d := rec2.Iters[biv]
		d.TileSize = 8
		rec2.Iters[biv] = d
	}
	cache := opt.NewCache()
	cache.Put(rec2)
	p := DefaultParams()
	p.MaxCache = 4104
	if err := (LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx2.g); err != nil {
		t.Fatalf("budget-fallback run: %v", err)
	}
	if !lhs2.BasePtr().Erased() || !rhs2.BasePtr().Erased() {
		t.Fatal("both sites must stay promoted after halving under a 4104-byte budget")
	}
	findBuffer(t, fx2.g, ir.KindFragNew, 8)       // lhs at T_k = 2
	findBuffer(t, fx2.g, ir.KindMemNewSMem, 4096) // rhs at window (2,2)
	assertVerifyAndRootReachable(t, fx2.g)

	// F4 drop: a 24-byte budget can never fit the WARP window (>= 1024 B
	// at any tile) — rhs halving reaches tile 1 and DROPS; the shared K
	// iter drags the lhs FRAG site to tile 1, which drops it too (ANY
	// covered iter at 1, not only all of them — ME-01).
	fx2b, lhs2b, rhs2b, _ := matmulUsed()
	rec2b := matmulRecord(t, fx2b, lhs2b, rhs2b)
	for biv := range rec2b.Iters {
		d := rec2b.Iters[biv]
		d.TileSize = 8
		rec2b.Iters[biv] = d
	}
	cache2b := opt.NewCache()
	cache2b.Put(rec2b)
	p2b := DefaultParams()
	p2b.MaxCache = 24
	if err := (LowerPass{Params: p2b}).Run(&opt.Context{Pass: "tile-lower", Cache: cache2b}, fx2b.g); err != nil {
		t.Fatalf("budget-drop run: %v", err)
	}
	if lhs2b.BasePtr().Erased() || rhs2b.BasePtr().Erased() {
		t.Fatal("both sites must drop back to GMEM under a 24-byte budget (F4: the shared K iter at tile 1 drops lhs too)")
	}
	assertVerifyAndRootReachable(t, fx2b.g)

	// Budget of 1: every site drops; the pass still terminates clean.
	fx3, lhs3, rhs3, _ := matmulUsed()
	rec3 := matmulRecord(t, fx3, lhs3, rhs3)
	cache3 := opt.NewCache()
	cache3.Put(rec3)
	p3 := DefaultParams()
	p3.MaxCache = 1
	if err := (LowerPass{Params: p3}).Run(&opt.Context{Pass: "tile-lower", Cache: cache3}, fx3.g); err != nil {
		t.Fatalf("maximal fallback run: %v", err)
	}
	if lhs3.BasePtr().Erased() || rhs3.BasePtr().Erased() {
		t.Fatal("with MaxCache 1 no site may promote")
	}
}

// TestSelfCheckInjections pins P7/§8 (F5: the checks are structural and
// individually provable): a crafted lowerRun carrying one violation must
// fail the corresponding check with a named error.
func TestSelfCheckInjections(t *testing.T) {
	t.Run("stale iter", func(t *testing.T) {
		fx := newChain32("inj1", 8)
		u32 := ir.InternScalar(ir.TU32)
		dv := fx.g.NewDimVar("ctor", fx.dims[0], fx.g.NewConst("ctor", u32, 0), fx.g.NewConst("ctor", u32, 1))
		biv := fx.g.NewBoundedDimVar("ctor", dv, fx.g.NewConst("ctor", u32, 8))
		fx.g.NewIter("ctor", biv, nil) // a LIVE iter survives
		lr := &lowerRun{g: fx.g}
		if err := lr.checkRetiredIters(); err == nil || !strings.Contains(err.Error(), "survived the rewrite") {
			t.Fatalf("a live iter must trip the retired-iters check, got %v", err)
		}
	})

	t.Run("fill placement unreachable", func(t *testing.T) {
		g := ir.NewGraph("inj-place")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		ivn := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		loop := g.NewForLoop("ctor", ivn, g.NewBoundedDimVar("ctor", ivn, g.NewConst("ctor", u32, 8)),
			g.NewConst("ctor", u32, 0))
		buf := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		// A fill floating OUTSIDE the host's body subtree: exactly the
		// CR-02 shape the placement check must catch.
		src := g.NewMemLoad("ctor", tensorParam(g, "t", 1), []ir.Node{ivn})
		fill := g.NewMemStoreSMem("ctor", src, buf)
		lr := &lowerRun{g: g, fills: []*siteFill{{siteID: 3, src: src, store: fill, host: loop, variant: loadTileCta}}}
		if err := lr.checkFillPlacement(); err == nil || !strings.Contains(err.Error(), "fill-position loop") {
			t.Fatalf("an unspliced fill must trip the placement check, got %v", err)
		}
	})

	t.Run("fill scope leak", func(t *testing.T) {
		g := ir.NewGraph("inj-scope")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		outerIV := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 4))
		innerIV := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		buf := g.NewFragNew("ctor", g.NewConst("ctor", u32, 32))
		// A fill HOSTED AT OUTER (host = the outer loop) whose source
		// reads INNER's dimvar — the inner loop does not enclose the host,
		// exactly the CR-02/HI-01 leakage class. The fill is properly
		// reachable from the host body (an access inside inner's body
		// chains after it), so ONLY the scope half fires.
		src := g.NewMemLoad("ctor", tensorParam(g, "t", 1), []ir.Node{innerIV})
		fill := g.NewFragStore("ctor", src, buf)
		bufEnv := ir.NewEnv()
		bufEnv.NoteStore(buf, fill)
		access := g.NewFragLoadChained("ctor", bufEnv, buf, innerIV)
		inner := g.NewForLoop("ctor", innerIV,
			g.NewBoundedDimVar("ctor", innerIV, g.NewConst("ctor", u32, 8)), access)
		outer := g.NewForLoop("ctor", outerIV,
			g.NewBoundedDimVar("ctor", outerIV, g.NewConst("ctor", u32, 64)), inner)
		lr := &lowerRun{
			g: g,
			rewrites: []*lntRewrite{{dims: []*dimRewrite{
				{betweenIV: outerIV, between: outer},
				{insideIV: innerIV, inside: inner},
			}}},
			fills: []*siteFill{{siteID: 5, src: src, store: fill, host: outer, variant: loadTileThread}},
		}
		if err := lr.checkFillPlacement(); err == nil || !strings.Contains(err.Error(), "enclose") {
			t.Fatalf("a fill reading a non-enclosing loop's dimvar must trip the scope check, got %v", err)
		}
		_ = outer
	})

	t.Run("cooperative fill inside element loop", func(t *testing.T) {
		g := ir.NewGraph("inj-layer")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		ivn := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		buf := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		src := g.NewMemLoad("ctor", tensorParam(g, "t", 1), []ir.Node{ivn})
		fill := g.NewMemStoreSMem("ctor", src, buf)
		// The loop's BODY OPERAND is the fill — the HI-02 shape: a
		// cooperative fill inside an element loop.
		loop := g.NewForLoop("ctor", ivn, g.NewBoundedDimVar("ctor", ivn, g.NewConst("ctor", u32, 8)), fill)
		lr := &lowerRun{
			g: g,
			rewrites: []*lntRewrite{{dims: []*dimRewrite{{
				soc: SoCSequential, inside: loop, between: loop, betweenIV: ivn, insideIV: ivn,
			}}}},
			fills: []*siteFill{{siteID: 7, src: src, store: fill, host: loop, variant: loadTileCta}},
		}
		if err := lr.checkFillPlacement(); err == nil || !strings.Contains(err.Error(), "element loop") {
			t.Fatalf("a cooperative fill inside an element loop must trip the layering check, got %v", err)
		}
	})

	t.Run("annotation missing", func(t *testing.T) {
		g := ir.NewGraph("inj3")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		ivn := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		loop := g.NewForLoop("ctor", ivn, g.NewBoundedDimVar("ctor", ivn, g.NewConst("ctor", u32, 8)), g.NewConst("ctor", u32, 0))
		lr := &lowerRun{g: g, rewrites: []*lntRewrite{{dims: []*dimRewrite{{between: loop}}}}}
		if err := lr.checkLoopAnnotations(); err == nil || !strings.Contains(err.Error(), "soc") {
			t.Fatalf("an annotation-less generated loop must trip the annotation check, got %v", err)
		}
	})

	t.Run("fill without loadtile annotation", func(t *testing.T) {
		g := ir.NewGraph("inj-attr")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		ivn := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		buf := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		src := g.NewMemLoad("ctor", tensorParam(g, "t", 1), []ir.Node{ivn})
		fill := g.NewMemStoreSMem("ctor", src, buf) // NO trailing annotation
		lr := &lowerRun{g: g, fills: []*siteFill{{siteID: 8, src: src, store: fill}}}
		if err := lr.checkNoBarriers(); err == nil || !strings.Contains(err.Error(), "loadtile") {
			t.Fatalf("a fill without a live loadtile annotation must trip the barrier check, got %v", err)
		}
	})

	t.Run("stale promoted load", func(t *testing.T) {
		g := ir.NewGraph("inj4")
		tensor := tensorParam(g, "t", 2) // node id 1: a live MemLoad
		rec := &Record{Sites: map[uint64]SiteDecision{
			1: {Promoted: true, Class: ClassSMEM},
		}}
		lr := &lowerRun{g: g}
		if err := lr.checkNoStaleLoads(rec, nil); err == nil || !strings.Contains(err.Error(), "stale MemLoad") {
			t.Fatalf("a surviving promoted load must trip the stale-load check, got %v", err)
		}
		_ = tensor
	})

	t.Run("frag fully-unroll", func(t *testing.T) {
		fx := newChain32("inj5", 64)
		ctaIter := fx.iters[0]
		lr := &lowerRun{
			g: fx.g,
			decisions: map[ir.Node]IterDecision{
				ctaIter: {SoC: SoCTA, LoC: 100, TileSize: 32},
			},
			fills: []*siteFill{{siteID: 9, covered: []*dimRewrite{{iter: ctaIter}}}},
		}
		rec := &Record{Sites: map[uint64]SiteDecision{9: {Promoted: true, Class: ClassFRAG}}}
		if err := lr.checkFragSites(rec, nil); err == nil || !strings.Contains(err.Error(), "fully-unroll") {
			t.Fatalf("a FRAG site over a CTA dim must trip the fully-unroll check, got %v", err)
		}
	})
}

// assertReachable fails unless at least want live nodes of kind k are
// reachable from the graph's end node.
func assertReachable(t *testing.T, g *ir.Graph, k ir.Kind, want int) {
	t.Helper()
	var end ir.Node
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindReturn {
			end = n
		}
	}
	seen := make(map[*ir.Base]bool)
	var visit func(n ir.Node)
	visit = func(n ir.Node) {
		if n == nil || n.BasePtr() == nil || seen[n.BasePtr()] || n.BasePtr().Erased() {
			return
		}
		seen[n.BasePtr()] = true
		for _, dep := range ir.DataDeps(n) {
			visit(dep)
		}
		for _, u := range ir.ImpactBy(n) {
			visit(u.Def())
		}
	}
	visit(end)
	got := 0
	for base := range seen {
		if base.KindOf() == k {
			got++
		}
	}
	if got < want {
		t.Fatalf("reachability from the end node: %d live %s nodes, want >= %d", got, k, want)
	}
}

// TestDecideLowerComposition pins the plan's headline success criterion:
// the decide→verify→lower triple composes — tiling (114) → tiling-verify
// (115) → tile-lower (116) runs clean over the matmul worked example with
// the REAL searched record (no crafted entries; the F7 stamp is the verify
// pass's own product), ending verifier-clean and typable.
func TestDecideLowerComposition(t *testing.T) {
	fx, _, _, _ := matmulUsed()
	cache := opt.NewCache()
	ctx := &opt.Context{Pass: "tiling", Cache: cache}
	if err := (Pass{Params: DefaultParams()}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tiling decide: %v", err)
	}
	ctx.Pass = "tiling-verify"
	if err := (VerifyPass{Params: DefaultParams()}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tiling-verify: %v", err)
	}
	ctx.Pass = "tile-lower"
	if err := (LowerPass{Params: DefaultParams()}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tile-lower over the real record: %v", err)
	}
	if vs := ir.Verify(fx.g); len(vs) != 0 {
		t.Fatalf("verifier after the triple: %v", vs)
	}
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run after the triple: %v", err)
	}
}
