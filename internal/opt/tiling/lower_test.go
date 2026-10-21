package tiling

// lower_test.go pins the loop-tile-LOWER pass (quick-261021-jay,
// doc/optana/loop_tile_lower.md): the fail-closed record gates, the §2
// DoSoC numbering, the PHASE 3 canonical loop forms (SEQ/CTA/WARP, T==1
// and T>1), the M1 annotations, the P6 chain re-pointing liveness, the
// pass determinism, and the PHASE 4 site surgery (buffers, fills, access
// rewrites, budget fallback) including the matmul worked example.

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
// provided. bivOf resolves the record's BIV keys.
func recordOf(t *testing.T, fx fixture, decisions map[int]IterDecision, sites map[uint64]SiteDecision) *Record {
	t.Helper()
	rec := &Record{Iters: make(map[ir.Node]IterDecision), Sites: sites}
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
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g); err != nil {
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
// wrong-typed cache entry, and a nil context are named errors, not silent
// skips; Requires pins the §1 record lifetime.
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
	if got := (LowerPass{}).Requires(); len(got) != 1 || got[0] != "tiling" {
		t.Fatalf("Requires = %v, want [tiling]", got)
	}
	spec, ok := opt.Lookup("tile-lower")
	if !ok || spec.Priority != 116 || len(spec.Requires) != 1 || spec.Requires[0] != "tiling" {
		t.Fatalf("registry spec = %+v, want tile-lower@116 requiring tiling", spec)
	}
	// Sanity: with the record present the same graph lowers clean.
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
	fx := newChain32("seq", 512)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 1}, // force a CTA dim so the record stays feasible
	}, nil)
	// Second chain dim is not present — use a 1-dim chain with SEQ only.
	fx2 := newChain32("seq2", 512)
	bodyStore(fx2)
	rec2 := recordOf(t, fx2, map[int]IterDecision{
		0: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, nil)
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
	// Inside bound = umin(T, extent − offset): a call node named "umin".
	bound := boundOf(t, inside)
	if bound.KindOf() != ir.KindCall || ir.CallNameOf(bound) != "umin" {
		t.Fatalf("SEQ inside bound must be a umin call, got %s", bound.KindOf())
	}
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
	_ = fx
	_ = rec
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
		t.Fatalf("CTA T>1 must yield between+inside, got %d", len(loops))
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
		t.Fatalf("WARP T>1 must yield between+inside, got %d", len(loops))
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

// TestRewriteDeterminism pins §7: two fresh copies through the pass render
// byte-identically (allocation-order walks, no map iteration in emission).
func TestRewriteDeterminism(t *testing.T) {
	build := func() *ir.Graph {
		fx := newChain32("det", 1024, 2048, 512)
		bodyStore(fx)
		return fx.g
	}
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
	_ = build
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

// flattenSubtree returns the data-subtree nodes (self first).
func flattenSubtree(n ir.Node) []ir.Node {
	var out []ir.Node
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() {
			return
		}
		out = append(out, x)
		for _, dep := range ir.DataDeps(x) {
			visit(dep)
		}
	}
	visit(n)
	return out
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
	for base := range seen {
		_ = base
	}
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
// Task 3: cache insertion — matmul end-to-end, budget fallback, self-checks
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

// matmulRecord crafts the decide record for matmulUsed: M=CTA(T32),
// N=WARP(T32), K=SEQ(T32); SET {lhs,rhs}@K — lhs tie N covering [·][K]
// (SMEM), rhs above the outermost covering [N][K] (SMEM).
func matmulRecord(t *testing.T, fx fixture, lhs, rhs ir.Node) *Record {
	t.Helper()
	return recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, map[uint64]SiteDecision{
		lhs.BasePtr().ID(): {Promoted: true, Tie: fx.iters[1], Class: ClassSMEM},
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

// TestMatmulEndToEnd pins the PHASE 4 surgery over the doc's worked
// example: both loads erased with their uses RAUW'd to buffer accesses,
// one buffer per site at the fill point, one annotation-carrying fill per
// site, verify-clean, typeinfer re-run clean.
func TestLowerMatmulEndToEnd(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec)

	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("both promoted loads must be erased")
	}
	// Buffers: lhs [T_k] = 4×32 = 128 bytes; rhs [T_n][T_k] = 4×32×32.
	lbuf := findBuffer(t, fx.g, ir.KindMemNewSMem, 128)
	rbuf := findBuffer(t, fx.g, ir.KindMemNewSMem, 4096)
	if ir.TypeOf(lbuf) == nil || ir.TypeOf(rbuf) == nil {
		t.Fatal("buffers must carry their bound tile tensor types")
	}
	// The store's value subtree reads the buffers (access rewrite).
	accesses := 0
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoadSMem {
			continue
		}
		accesses++
		src := ir.OperandsOf(n)[0]
		if src.BasePtr() != lbuf.BasePtr() && src.BasePtr() != rbuf.BasePtr() {
			t.Fatalf("access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
		}
	}
	if accesses != 2 {
		t.Fatalf("the multiply must read exactly the two buffer accesses, got %d", accesses)
	}
	// Fills: one MemStoreSMem per buffer, carrying the loadtile variant
	// annotation — lhs all-sequential covered → 2 (LoadTileThread); rhs
	// covers the WARP dim → 1 (LoadTileCta).
	fills := 0
	variants := map[float64]bool{}
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStoreSMem {
			continue
		}
		fills++
		attrs := loopAnnotations(n)
		v, ok := attrs["loadtile"]
		if !ok {
			t.Fatalf("fill node %d carries no loadtile annotation", n.BasePtr().ID())
		}
		variants[v] = true
		if dst := ir.OperandsOf(n)[1]; dst.BasePtr() != lbuf.BasePtr() && dst.BasePtr() != rbuf.BasePtr() {
			t.Fatalf("fill node %d targets a foreign buffer", n.BasePtr().ID())
		}
	}
	if fills != 2 || !variants[1] || !variants[2] {
		t.Fatalf("fills = %d variants %v, want 2 fills with variants {1, 2}", fills, variants)
	}
	// Loads-only: the out store survives as a GMEM MemStore.
	if store.BasePtr().Erased() || store.KindOf() != ir.KindMemStore {
		t.Fatal("stores never route through the cache: the GMEM store must survive")
	}
	// Graph health + P8.
	assertVerifyAndRootReachable(t, fx.g)
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run over the lowered matmul must succeed (P8): %v", err)
	}
	// Fill anchoring: with real data flow every fill is reachable from the
	// end node through its accesses' effect chains.
	assertReachable(t, fx.g, ir.KindMemStoreSMem, 2)
}

// TestMatmulWorkedCheck pins §6bis's worked check: the lhs [T_k] buffer
// fills per k_o round — the fill's k coordinate is the K between loop's
// round-base dimvar, and the m coordinate is the enclosing m index
// expression (INS above the tie baked as per-round fixed values), NOT a
// loop over the buffer.
func TestMatmulWorkedCheck(t *testing.T) {
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec)

	lbuf := findBuffer(t, fx.g, ir.KindMemNewSMem, 128)
	var lhsFill ir.Node
	for _, n := range fx.g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindMemStoreSMem &&
			ir.OperandsOf(n)[1].BasePtr() == lbuf.BasePtr() {
			lhsFill = n
		}
	}
	if lhsFill == nil {
		t.Fatal("lhs fill not found")
	}
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
	fillBivs := ir.OperandsOf(lhsFill)[2:]
	if len(fillBivs) != 2 { // covered round base + the loadtile annotation
		t.Fatalf("lhs fill must carry its covered round base plus the variant annotation, got %d trailing operands", len(fillBivs))
	}
	if fillBivs[0].BasePtr() != kBetweenIV.BasePtr() {
		t.Fatal("the lhs fill's round base must be the K between loop's dimvar (filled per k_o round)")
	}
}

// TestBudgetFallback pins §6: with a forced-tiny MaxCache the halving loop
// terminates — tile sizes halve while over budget, sites drop back to GMEM
// when halving reaches 1; with DefaultParams the fallback is a no-op.
func TestBudgetFallback(t *testing.T) {
	// No-op case: both sites promote at their full tiles.
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec)
	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("with DefaultParams both sites must promote")
	}

	// Halving survival: tile-8 decisions under a 24-byte budget halve both
	// sites (lhs K 8→2 = 8 bytes; rhs N,K 8→2,8→2 = 16 bytes) and keep
	// them promoted.
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
	p.MaxCache = 24
	if err := (LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx2.g); err != nil {
		t.Fatalf("budget-fallback run: %v", err)
	}
	if !lhs2.BasePtr().Erased() || !rhs2.BasePtr().Erased() {
		t.Fatal("both sites must stay promoted after halving under a 24-byte budget")
	}
	findBuffer(t, fx2.g, ir.KindMemNewSMem, 8)  // lhs at T_k = 2
	findBuffer(t, fx2.g, ir.KindMemNewSMem, 16) // rhs at (2,2)
	assertVerifyAndRootReachable(t, fx2.g)

	// Drop: a 20-byte budget drives rhs to (1,1) — it drops back to GMEM
	// (the shared K iter drags lhs to tile 1 as well; both survive as
	// GMEM loads, unwound).
	fx4, lhs4, rhs4, _ := matmulUsed()
	rec4 := matmulRecord(t, fx4, lhs4, rhs4)
	for biv := range rec4.Iters {
		d := rec4.Iters[biv]
		d.TileSize = 8
		rec4.Iters[biv] = d
	}
	cache4 := opt.NewCache()
	cache4.Put(rec4)
	p4 := DefaultParams()
	p4.MaxCache = 20
	if err := (LowerPass{Params: p4}).Run(&opt.Context{Pass: "tile-lower", Cache: cache4}, fx4.g); err != nil {
		t.Fatalf("budget-drop run: %v", err)
	}
	if rhs4.BasePtr().Erased() {
		t.Fatal("rhs must drop back to GMEM when halving reaches 1")
	}
	assertVerifyAndRootReachable(t, fx4.g)

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

// TestSelfCheckInjections pins P7/§8: each exit condition is individually
// provable — a crafted lowerRun carrying one violation must fail the
// corresponding check with a named error.
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

	t.Run("fill layering", func(t *testing.T) {
		g := ir.NewGraph("inj2")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		insideIV := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 256))
		tensor := tensorParam(g, "t", 2)
		src := g.NewMemLoad("ctor", tensor, []ir.Node{insideIV}) // a fill source referencing a covered dim's inside loop coordinate
		d := &dimRewrite{soc: SoCTA, insideIV: insideIV, inside: g.NewForLoop("ctor", insideIV,
			g.NewBoundedDimVar("ctor", insideIV, g.NewConst("ctor", u32, 8)), g.NewConst("ctor", u32, 0))}
		lr := &lowerRun{g: g, fills: []*siteFill{{siteID: 7, src: src, covered: []*dimRewrite{d}, annotated: true}}}
		if err := lr.checkFillLayering(); err == nil || !strings.Contains(err.Error(), "§4 layering") {
			t.Fatalf("a fill inside a parallel inside loop must trip the layering check, got %v", err)
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
		u32 := ir.InternScalar(ir.TU32)
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
		_ = u32
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
// the decide→lower pair composes — tiling (114) → tiling-verify (115) →
// tile-lower (116) runs clean over the matmul worked example with the REAL
// record the search publishes, ending verifier-clean and typable.
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
		t.Fatalf("verifier after the pair: %v", vs)
	}
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run after the pair: %v", err)
	}
}
