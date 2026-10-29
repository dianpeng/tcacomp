package tiling

// lower_test.go pins the loop-tile-lower pass (quick-261021-jay +
// 261021-o61 T3/T5, doc/optana/loop_tile_lower.md as amended): the fail-closed
// record gates (incl. the F7 tiling-verify stamp), the §2 DoSoC numbering,
// the PHASE 3 canonical loop forms (SEQ/CTA/WARP, every dim tiled T >= 2 —
// the 2026-10-24 tile-disable ruling deleted the T==1 single form, the F1
// saturating tail guard), the F3 branched-LNT rejection, the LO-05 extent
// typing guard, the M1 annotations, the P6 chain re-pointing liveness, the
// pass determinism, and the PHASE 4 site surgery (variant-dependent fill
// placement per R1, R2 window buffers + rank-in-index accesses, budget
// fallback with the F4 drop rule, halving floor 2) including the matmul
// worked example, the
// decide→verify→lower composition over the REAL searched record, and the
// Materialize+print round-trip golden of the lowered graph (IR-level pins
// only — no numerical or GPU execution exists in-tree).

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"tcacomp/internal/golden"
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
// WARP-covered rhs) price at the fix4 factor-model window —
// K×T_n × T_k × 4 = 8×32×32×4 = 32 KB — inside the 64 KB default, so the
// explicit-budget cases below tighten MaxCache deliberately.
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

// nestParent returns the loop's canonical-chain parent: the value of its
// "nest" KeyedAttribute (fix1.md §0: loop-to-loop structure is permitted
// explicitly; buildNest wires each generated loop to its parent this way).
// The root carries no nest attribute and answers nil.
func nestParent(l ir.Node) ir.Node {
	for _, op := range ir.OperandsOf(l) {
		if op == nil || op.KindOf() != ir.KindKeyedAttribute {
			continue
		}
		key, val := ir.KeyedAttributeOf(op)
		if key == "nest" {
			return val
		}
	}
	return nil
}

// nestOf returns the live for-loops in NEST order (outermost first). Each
// non-root loop carries a "nest" KeyedAttribute whose value is its parent
// (child→parent links), so the walk inverts those links: the unique loop
// with no parent is the root, and every parent has exactly one child (the
// canonical chain is linear). The pass builds nests inside-out, so
// allocation order is the reverse — tests reason in dim order.
func nestOf(t *testing.T, g *ir.Graph) []ir.Node {
	t.Helper()
	loops := loopsOf(g)
	childOf := make(map[*ir.Base]ir.Node, len(loops))
	hasParent := make(map[*ir.Base]bool, len(loops))
	for _, l := range loops {
		p := nestParent(l)
		if p == nil {
			continue
		}
		if p.KindOf() != ir.KindForLoop {
			t.Fatalf("loop %d nest parent %d is not a for-loop",
				l.BasePtr().ID(), p.BasePtr().ID())
		}
		if p.BasePtr() == l.BasePtr() {
			t.Fatalf("loop %d is its own nest parent", l.BasePtr().ID())
		}
		if hasParent[l.BasePtr()] {
			t.Fatalf("loop %d carries more than one nest attribute", l.BasePtr().ID())
		}
		hasParent[l.BasePtr()] = true
		if _, dup := childOf[p.BasePtr()]; dup {
			t.Fatalf("loop %d has two nest children (branched nest?)", p.BasePtr().ID())
		}
		childOf[p.BasePtr()] = l
	}
	var root ir.Node
	for _, l := range loops {
		if !hasParent[l.BasePtr()] {
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
	for n := root; n != nil; n = childOf[n.BasePtr()] {
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
		if field := ir.RuntimeAttributeOf(n); field != "" {
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
		0: {SoC: SoCTA, LoC: 100, TileSize: 2},
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

// TestNumbering pins §2 under the fix4 §3.2 dimensionality ruling: the CTA
// dim takes a GridMapping slot ONLY (its coupled inside walk rides the
// NO-ARG flat thread rank — no ThreadMap slot); the WARP dim takes the
// FIRST ThreadMap slot (WARP dims are the ThreadMap's only dimensions);
// the sequential dim gets none; the second LNT restarts both counters.
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
	// The coupled inside walk carries NO dosoc (the §3.2 ruling) and its LoC
	// is the LAUNCH blockDim (max over LNTs of (Π warp K) × 32 = 8×32 = 256).
	if got := loopAnnotations(mI); got["soc"] != 2 || got["loc"] != 256 {
		t.Fatalf("M inside (coupled walk) annotations = %v, want soc=2 loc=256 with NO dosoc", got)
	}
	if _, ok := loopAnnotations(mI)["dosoc"]; ok {
		t.Fatal("the CTA-coupled inside walk must carry NO dosoc annotation (fix4 3.2: the flat rank needs no ThreadMap slot)")
	}
	// The WARP between rides ThreadMap slot 0 (WARP-only numbering) with
	// loc = K_d in WARP units (absolute — not K_d×32).
	if got := loopAnnotations(nB); got["soc"] != 2 || got["dosoc"] != 0 || got["loc"] != 8 {
		t.Fatalf("N between annotations = %v, want soc=2 dosoc=0 loc=8 (K_d warp units)", got)
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

	// Strides: M between LoC×T = 3200; M inside = the launch blockDim 256;
	// N between K_d×T = 8×32 = 256 (warp rounds, NOT (K×32)×T); N inside = 1
	// (lane chunks, sequential per lane); K between = T = 32.
	for _, tc := range []struct {
		loop ir.Node
		want float64
	}{
		{mB, 3200}, {mI, 256}, {nB, 256}, {nI, 1}, {kB, 32}, {kI, 1},
	} {
		if got := dimvarStep(t, tc.loop); got != tc.want {
			t.Fatalf("loop %d step = %v, want %v", tc.loop.BasePtr().ID(), got, tc.want)
		}
	}

	// The second LNT restarts: its single CTA dim gets grid dosoc 0 again.
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

// assertFillPositionDerived is the fix3 D-i SAFETY NET: the R-I derivation
// — the innermost rewritten loop whose IV the fill composition's operand
// closure references (nest-chain depth breaks ties inward) — must pick
// EXACTLY the position fillHost mandates today. Step 4 (D-j) swaps the
// placement mechanism under this equality; if a shape's derivation
// disagreed, that swap would MOVE the fill and this pin fires first.
func assertFillPositionDerived(t *testing.T, g *ir.Graph, fill, mandated ir.Node) {
	t.Helper()
	// The composition's coordinate closure: the STORE's operand closure
	// covers the source load's axes and the covered round bases.
	closure := map[ir.Node]bool{}
	for _, n := range flattenSubtree(fill) {
		closure[n] = true
	}
	nestParent := func(loop ir.Node) ir.Node {
		for _, op := range ir.OperandsOf(loop)[2:] {
			if key, val := ir.KeyedAttributeOf(op); key == "nest" && val != nil {
				return val
			}
		}
		return nil
	}
	depth := func(loop ir.Node) int {
		d := 0
		for {
			p := nestParent(loop)
			if p == nil || p.BasePtr() == nil || p.BasePtr().Erased() {
				return d
			}
			loop = p
			d++
		}
	}
	var best ir.Node
	bestDepth := -1
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindForLoop {
			continue
		}
		iv := ir.OperandsOf(n)[0]
		if iv == nil || !closure[iv] {
			continue
		}
		if d := depth(n); d > bestDepth {
			best, bestDepth = n, d
		}
	}
	if best == nil {
		t.Fatal("the fill's closure references no rewritten loop — the R-I derivation has no position to pick")
	}
	if best.BasePtr() != mandated.BasePtr() {
		t.Fatalf("the R-I derivation picks loop %d (nest depth %d), want the mandated position loop %d — the D-j swap would move this fill",
			best.BasePtr().ID(), bestDepth, mandated.BasePtr().ID())
	}
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
	// The WARP inside bound is the LANE-CHUNK wrapper: umin((lane+1)*c,
	// F1-tileMax) — unwrap the chunk end, then assert the F1 form.
	nBound := boundOf(t, nI)
	if nBound.KindOf() != ir.KindCall || ir.CallNameOf(nBound) != "umin" {
		t.Fatalf("WARP inside bound must be the lane-chunk umin, got %s", kindOf(nBound))
	}
	f1 := ir.OperandsOf(nBound)[1]
	assertSaturatingBound(t, fx.g, f1, 32, nExtent, nB)
	// The clamped offsets are the RANK forms: the M inside bound's clamp
	// references the GridMap rank, the N inside bound's the ThreadMap rank.
	mClamp := ir.OperandsOf(ir.OperandsOf(boundOf(t, mI))[1])[1]
	if !subtreeHasKind(t, fx.g, mClamp, ir.KindRuntimeGridQuery) {
		t.Fatal("the CTA inside bound must clamp the grid-rank offset form")
	}
	nClamp := ir.OperandsOf(ir.OperandsOf(f1)[1])[1]
	if !subtreeHasKind(t, fx.g, nClamp, ir.KindRuntimeThreadQuery) {
		t.Fatal("the WARP inside bound must clamp the thread-rank offset form")
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteCTADim pins the CTA template: between step LoC×T; the in-body
// GridMap rank expression feeds offset = rank×T + round; the inside loop
// strides the coupled thread count; the flat-index ruling (2026-10-23)
// makes the inside IV's INIT the zero-operand flat thread query and the
// index expression offset + inside — the tuple-coordinate +index(threadDim)
// add is DELETED (finding C1).
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
	if got := dimvarStep(t, inside); got != 32 {
		t.Fatalf("CTA inside step = %v, want blockDim = 32 (no WARP dim: one warp)", got)
	}
	// The coupled inside walk carries NO dosoc (the §3.2 dimensionality
	// ruling: the flat rank needs no ThreadMap slot) and its loc IS the
	// launch blockDim (32 — this LNT has no WARP dim).
	if got := loopAnnotations(inside); got["soc"] != 2 || got["loc"] != 32 {
		t.Fatalf("CTA inside annotations = %v, want soc=2 loc=32 with NO dosoc", got)
	}
	if _, ok := loopAnnotations(inside)["dosoc"]; ok {
		t.Fatal("the CTA-coupled inside walk must carry NO dosoc annotation (fix4 3.2)")
	}
	// The inside IV's INIT is the FLAT zero-operand thread query
	// (Index() = flatRank + i*coupledThreads — the single-owner mechanism).
	iv := ir.OperandsOf(inside)[0]
	init := ir.OperandsOf(iv)[1]
	if init == nil || init.KindOf() != ir.KindRuntimeThreadQuery || len(ir.OperandsOf(init)) != 0 {
		t.Fatalf("CTA inside IV init = %v, want the zero-operand flat thread query", init)
	}
	// The store's axis (its only operand tree referencing the loop) must
	// resolve to offset + insideIV: a GridMap rank query for the offset and
	// NO one-operand (tuple-coordinate) thread query — the +index(threadDim)
	// term is deleted.
	axes := ir.OperandsOf(store)[2:]
	if len(axes) != 1 {
		t.Fatalf("store must carry one rewritten axis, got %d", len(axes))
	}
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindRuntimeGridQuery) {
		t.Fatal("rewritten index must contain the GridMap rank query")
	}
	for _, x := range flattenSubtree(axes[0]) {
		if x.KindOf() == ir.KindRuntimeThreadQuery && len(ir.OperandsOf(x)) != 0 {
			t.Fatalf("rewritten index carries a tuple-coordinate thread query %d — the +index(threadDim) add is deleted", x.BasePtr().ID())
		}
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestRewriteWarpDim pins the fix4 §3.1 WARP template: between rounds
// stride K_d×T_d (WARP units — the dim's warp count times its tile, NOT
// (K×32)×T); the in-body rank is the ThreadMap tuple coordinate in WARP
// units; the inside walk is LANE CHUNKS — the owning warp's 32 lanes split
// the tile into consecutive chunks of ceil(T/32), the IV init is
// lane×ceil(T/32) over the zero-operand lane query, the step is 1, and the
// bound is umin((lane+1)×ceil(T/32), tile_max) wrapping the F1 saturating
// tile_max product.
func TestRewriteWarpDim(t *testing.T) {
	fx := newChain32("warp", 4096)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCWARP, LoC: 8, TileSize: 64},
	}, nil)
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g)
	if len(loops) != 2 {
		t.Fatalf("WARP T>1 must yield between+inside, got %d loops", len(loops))
	}
	between, inside := loops[0], loops[1]
	// Between: stride K_d×T_d = 8×64 = 512; loc = K_d = 8 (warp units);
	// dosoc = the dim's ThreadMap slot 0 (WARP-only numbering).
	if got := dimvarStep(t, between); got != 512 {
		t.Fatalf("WARP between step = %v, want K_d×T = 8×64 = 512", got)
	}
	if got := loopAnnotations(between); got["soc"] != 2 || got["dosoc"] != 0 || got["loc"] != 8 {
		t.Fatalf("WARP between annotations = %v, want soc=2 dosoc=0 loc=8 (K_d warp units)", got)
	}
	// Inside: sequential per lane (step 1), soc 0.
	if got := dimvarStep(t, inside); got != 1 {
		t.Fatalf("WARP inside must be sequential (step 1), got %v", got)
	}
	if got := loopAnnotations(inside)["soc"]; got != 0 {
		t.Fatalf("WARP inside soc = %v, want 0 (sequential chunk walk)", got)
	}
	// LANE-CHUNK init: lane_id × ceil(T/32) with ceil(64/32) = 2.
	iv := ir.OperandsOf(inside)[0]
	init := ir.OperandsOf(iv)[1]
	if init == nil || init.KindOf() != ir.KindBinOp {
		t.Fatalf("WARP inside IV init = %v (%s), want the lane×c mul", init, kindOf(init))
	}
	if code, _ := ir.BinOpCodeOf(init); code != ir.BinOpMul {
		t.Fatalf("WARP inside IV init must be a mul (lane×c), got code %v", code)
	}
	mulOps := ir.OperandsOf(init)
	if v, ok := ir.ConstValueOf(mulOps[1]); !ok || v != 2 {
		t.Fatalf("WARP inside IV init chunk size = %v, want ceil(64/32) = 2", mulOps[1])
	}
	if lane := mulOps[0]; lane == nil || lane.KindOf() != ir.KindRuntimeThreadQuery ||
		len(ir.OperandsOf(lane)) != 0 || ir.RuntimeAttributeOf(lane) != "lane_id" {
		t.Fatalf("WARP inside IV init lane operand = %v, want the zero-operand lane_id query", lane)
	}
	// LANE-CHUNK bound: umin((lane+1)×c, F1-tileMax) — the chunk end wraps
	// the saturating tile_max product.
	bound := boundOf(t, inside)
	if bound.KindOf() != ir.KindCall || ir.CallNameOf(bound) != "umin" {
		t.Fatalf("WARP inside bound must be a umin call, got %s", kindOf(bound))
	}
	bOps := ir.OperandsOf(bound)
	end := bOps[0]
	if end.KindOf() != ir.KindBinOp {
		t.Fatalf("chunk end = %s, want the (lane+1)×c mul", kindOf(end))
	}
	if code, _ := ir.BinOpCodeOf(end); code != ir.BinOpMul {
		t.Fatalf("chunk end must be a mul, got code %v", code)
	}
	endLane := ir.OperandsOf(end)[0]
	if code, _ := ir.BinOpCodeOf(endLane); code != ir.BinOpAdd {
		t.Fatalf("chunk end's lane term must be (lane+1), got code %v", code)
	}
	if v, ok := ir.ConstValueOf(ir.OperandsOf(endLane)[1]); !ok || v != 1 {
		t.Fatalf("chunk end's lane term must add 1, got %v", ir.OperandsOf(endLane)[1])
	}
	// The umin's second operand is the F1 saturating tile_max: another umin
	// call (T, extent − umin(offset, extent)).
	if f1 := bOps[1]; f1.KindOf() != ir.KindCall || ir.CallNameOf(f1) != "umin" {
		t.Fatalf("the chunk bound must wrap the F1 tile_max umin, got %s", kindOf(f1))
	}
	axes := ir.OperandsOf(store)[2:]
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindRuntimeThreadQuery) {
		t.Fatal("rewritten WARP index must contain the ThreadMap rank query")
	}
	if subtreeHasKind(t, fx.g, axes[0], ir.KindRuntimeGridQuery) {
		t.Fatal("a WARP-only LNT must contain no GridMap query")
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// kindOf renders a node's kind nil-safely (test diagnostics).
func kindOf(n ir.Node) string {
	if n == nil || n.BasePtr() == nil {
		return "<nil>"
	}
	return n.KindOf().String()
}

// TestRewriteTileOneRejected pins the 2026-10-24 tile-disable ruling's
// gate 3 backstop: even a Verified crafted record (F7-stamped, bypassing
// Params.Validate and tiling-verify, gates 1-2) whose iter carries
// TileSize 1 reaches the rewrite and FAILS CLOSED — tile size 1 is not a
// legal schedule, the compiler never generates a non-tiling schedule. (The
// former TestRewriteTileOne pinned the T==1 single-form the ruling
// deleted; this test guards the hole it left against hand-stamped
// records.)
func TestRewriteTileOneRejected(t *testing.T) {
	fx := newChain32("tileone", 1024, 512)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 2},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 1}, // hand-stamped illegal
	}, nil)
	cache := opt.NewCache()
	cache.Put(rec)
	err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g)
	if err == nil {
		t.Fatal("a record decision at TileSize 1 must fail closed at the rewrite backstop (gate 3)")
	}
	msg := err.Error()
	if !strings.Contains(msg, "not a legal schedule") || !strings.Contains(msg, "gate 3") {
		t.Fatalf("the backstop error must name the ruling and the gate, got: %v", err)
	}
}

// TestRewriteMinimalTile pins the minimal legal schedule end: every dim
// tiled at the floor tile 2 ({CTA,100,2}/{SEQ,1,2} on a 2-deep chain)
// still takes the canonical between/inside form — FOUR loops, dim order
// [d0.between, d1.between, d0.inside, d1.inside]. There is no smaller
// schedule: tile 1 is not a legal schedule (2026-10-24 ruling), so this
// nest shape is the compiler's flat endpoint.
func TestRewriteMinimalTile(t *testing.T) {
	fx := newChain32("mintile", 1024, 512)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 2},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 2},
	}, nil)
	runLower(t, fx.g, rec)

	loops := nestOf(t, fx.g)
	if len(loops) != 4 {
		t.Fatalf("two dims at the floor tile 2 must yield exactly four loops (between+inside per dim), got %d", len(loops))
	}
	// Nest: [d0.between, d1.between, d0.inside, d1.inside]; sea-of-nodes
	// (fix1.md §0): the innermost loop has NO body operand — every operand
	// is a header (iv dimvar, bounded dimvar) or a keyed attribute. The
	// store's membership rides its IV references, pinned below.
	for i, op := range ir.OperandsOf(loops[3]) {
		if op == nil {
			t.Fatalf("innermost loop operand %d is nil", i)
		}
		switch op.KindOf() {
		case ir.KindDimVar, ir.KindBoundedDimVar, ir.KindKeyedAttribute:
		default:
			t.Fatalf("innermost loop operand %d (%s) is an expression link — loops carry no body (fix1.md §0)", i, op.KindOf())
		}
	}
	// The between rounds step by LoC·T for the CTA dim (the CTA-group round
	// stride) and by the tile for the SEQ dim; the inside walks step 1.
	if got := dimvarStep(t, loops[0]); got != 200 {
		t.Fatalf("d0's between IV must step by LoC×T = 200, got %v", got)
	}
	if got := dimvarStep(t, loops[1]); got != 2 {
		t.Fatalf("d1's between IV must step by the tile 2, got %v", got)
	}
	// The CTA coupled inside walk is the flat-init stride (blockDim =
	// (Π warp K) × 32 = 32 here — no WARP dim in this fixture); the SEQ
	// inside walk steps 1.
	if got := dimvarStep(t, loops[2]); got != 32 {
		t.Fatalf("d0's coupled inside IV must step by the blockDim 32 (no WARP dim), got %v", got)
	}
	if got := dimvarStep(t, loops[3]); got != 1 {
		t.Fatalf("d1's inside IV must step 1, got %v", got)
	}
	// The store's d1 axis rides betweenIV + insideIV of its dim.
	axes := ir.OperandsOf(store)[2:]
	kAxis := axes[1]
	d1 := loops[1] // d1 between
	d1in := loops[3]
	for _, want := range []ir.Node{ir.OperandsOf(d1)[0], ir.OperandsOf(d1in)[0]} {
		if !subtreeContains(kAxis, want) {
			t.Fatalf("the rewritten d1 axis must reference both d1 loops' dimvars")
		}
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestLowerMultiDimTemplates pins the fix4 §3.1 multi-dim rewrite (T4 —
// the R-SEAM fail-closed guard is GONE): a record carrying TWO CTA dims
// AND TWO WARP dims in one LNT lowers with the per-dim templates — each
// CTA between nests with its own LoC_d×T_d stride and its own GridMap
// slot, each WARP between with its own K_d×T_d stride and its own
// ThreadMap slot, every coupled inside walk strides the LAUNCH blockDim
// (max over LNTs of (Π warp K) × 32 — NOT the kernel-wide product), and
// the WARP rank queries read their OWN slot. The old tile-vs-coupled
// backstop is DELETED with R-HI03 (the flat-init coupled walk is
// single-owner exact for ANY tile_max), so tile sizes over the blockDim
// lower clean.
func TestLowerMultiDimTemplates(t *testing.T) {
	fx := newChain32("multidim", 4096, 4096, 4096, 4096)
	store := bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 4, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 2, TileSize: 32},
		2: {SoC: SoCTA, LoC: 4, TileSize: 32},
		3: {SoC: SoCWARP, LoC: 4, TileSize: 32},
	}, nil)
	runLower(t, fx.g, rec)

	// MALoC-legal: Π CTA LoC = 16 ≤ 512, Π warp K = 8 ≤ 32 → the launch
	// blockDim is 8×32 = 256.
	loops := nestOf(t, fx.g)
	if len(loops) != 8 {
		t.Fatalf("four tiled dims must yield 8 loops, got %d", len(loops))
	}
	d0b, d1b, d2b, d3b, d0i, d1i, d2i, d3i := loops[0], loops[1], loops[2], loops[3], loops[4], loops[5], loops[6], loops[7]
	for _, tc := range []struct {
		name  string
		loop  ir.Node
		soc   float64
		dosoc float64 // -1 = absent
		loc   float64
		step  float64
	}{
		{"d0 CTA between", d0b, 1, 0, 4, 4 * 32},
		{"d1 WARP between", d1b, 2, 0, 2, 2 * 32},
		{"d2 CTA between", d2b, 1, 1, 4, 4 * 32},
		{"d3 WARP between", d3b, 2, 1, 4, 4 * 32},
		{"d0 coupled inside", d0i, 2, -1, 256, 256},
		{"d1 lane chunks", d1i, 0, -1, 1, 1},
		{"d2 coupled inside", d2i, 2, -1, 256, 256},
		{"d3 lane chunks", d3i, 0, -1, 1, 1},
	} {
		attrs := loopAnnotations(tc.loop)
		if attrs["soc"] != tc.soc || attrs["loc"] != tc.loc {
			t.Fatalf("%s annotations = %v, want soc=%v loc=%v", tc.name, attrs, tc.soc, tc.loc)
		}
		got, has := attrs["dosoc"]
		if tc.dosoc < 0 && has {
			t.Fatalf("%s must carry NO dosoc, got %v", tc.name, got)
		}
		if tc.dosoc >= 0 && (!has || got != tc.dosoc) {
			t.Fatalf("%s dosoc = %v (present %v), want %v", tc.name, got, has, tc.dosoc)
		}
		if step := dimvarStep(t, tc.loop); step != tc.step {
			t.Fatalf("%s step = %v, want %v", tc.name, step, tc.step)
		}
	}
	// The rewritten index carries each CTA dim's OWN GridMap slot (0 and 2's
	// slot 1) and each WARP dim's OWN ThreadMap slot (1's slot 0 and 3's
	// slot 1) — the slot counters go past 1 and stay WARP-only on the thread
	// side. Every axis of the store participates (each dim's index
	// expression queries its own slots).
	axes := ir.OperandsOf(store)[2:]
	gridSlots := map[float64]bool{}
	threadSlots := map[float64]bool{}
	for _, axis := range axes {
		for _, x := range flattenSubtree(axis) {
			if x.KindOf() != ir.KindRuntimeGridQuery && x.KindOf() != ir.KindRuntimeThreadQuery {
				continue
			}
			if len(ir.OperandsOf(x)) == 0 {
				continue // the flat/lane forms
			}
			v, _ := ir.ConstValueOf(ir.OperandsOf(x)[0])
			if x.KindOf() == ir.KindRuntimeGridQuery {
				gridSlots[v] = true
			} else {
				threadSlots[v] = true
			}
		}
	}
	if !gridSlots[0] || !gridSlots[1] {
		t.Fatalf("the index must query BOTH GridMap slots, got %v", gridSlots)
	}
	if !threadSlots[0] || !threadSlots[1] {
		t.Fatalf("the index must query BOTH ThreadMap slots, got %v", threadSlots)
	}
	assertVerifyAndRootReachable(t, fx.g)

	// The 2-CTA-only arm: two CTA dims and no WARP dim — blockDim 32.
	fx2 := newChain32("multidim2", 4096, 4096)
	bodyStore(fx2)
	rec2 := recordOf(t, fx2, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 4, TileSize: 32},
		1: {SoC: SoCTA, LoC: 8, TileSize: 32},
	}, nil)
	runLower(t, fx2.g, rec2)
	loops2 := nestOf(t, fx2.g)
	if len(loops2) != 4 {
		t.Fatalf("two CTA dims must yield 4 loops, got %d", len(loops2))
	}
	if got := loopAnnotations(loops2[2]); got["soc"] != 2 || got["loc"] != 32 {
		t.Fatalf("2-CTA-only coupled inside annotations = %v, want soc=2 loc=32 (one warp)", got)
	}

	// The 2-WARP-only arm: two WARP dims, no CTA dim — launch blockDim
	// (2×4)×32 = 256; slots 0 and 1 both on the ThreadMap.
	fx3 := newChain32("multidim3", 4096, 4096)
	bodyStore(fx3)
	rec3 := recordOf(t, fx3, map[int]IterDecision{
		0: {SoC: SoCWARP, LoC: 2, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 4, TileSize: 32},
	}, nil)
	runLower(t, fx3.g, rec3)
	loops3 := nestOf(t, fx3.g)
	if len(loops3) != 4 {
		t.Fatalf("two WARP dims must yield 4 loops, got %d", len(loops3))
	}
	if got := loopAnnotations(loops3[0]); got["dosoc"] != 0 || got["loc"] != 2 {
		t.Fatalf("first WARP between = %v, want dosoc=0 loc=2", got)
	}
	if got := loopAnnotations(loops3[1]); got["dosoc"] != 1 || got["loc"] != 4 {
		t.Fatalf("second WARP between = %v, want dosoc=1 loc=4", got)
	}

	// Post-R-HI03: a CTA TileSize OVER the blockDim lowers clean (the
	// flat-init walk is single-owner exact for any tile_max).
	fx4 := newChain32("bignodim", 4096)
	bodyStore(fx4)
	rec4 := recordOf(t, fx4, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 2, TileSize: 512},
	}, nil)
	cache4 := opt.NewCache()
	cache4.Put(rec4)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache4}, fx4.g); err != nil {
		t.Fatalf("a CTA tile over the blockDim must lower clean post-R-HI03, got %v", err)
	}
}

// warpReduceFx builds the Wave-1 defense trigger shape: a 2-deep chain
// whose SECOND dim is the reduced dim of a real Map/Reduce (lhs[m][k] →
// map(as=[m,k]) → reduce(0,+) → out[m]) — the k dim's IV data-dep closure
// feeds the Reduce through the piped value.
func warpReduceFx(t *testing.T) fixture {
	t.Helper()
	fx := newChain32("warpred", 1024, 512)
	lhs := fx.g.NewMemLoad("ctor", tensorParam(fx.g, "lhs", 2),
		[]ir.Node{iv(fx.g, fx.iters[0]), iv(fx.g, fx.iters[1])})
	mp := fx.g.NewMap("ctor", []ir.Node{lhs}, ir.NewLambda(nil, lhs),
		append([]ir.Node(nil), fx.dims...))
	acc := fx.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := fx.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := fx.g.NewBinOp("ctor", ir.BinOpAdd,
		fx.g.NewParamRef("ctor", acc), fx.g.NewParamRef("ctor", now))
	red := fx.g.NewReduce("ctor", fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	env := ir.NewEnv()
	st := fx.g.NewMemStoreChained("ctor", env, fx.out, []ir.Node{iv(fx.g, fx.iters[0])}, red)
	fx.g.NewReturnChained("ctor", env, []ir.Node{fx.out}, st)
	return fx
}

// TestLowerWarpReducedDimRejected pins the fix4 §3.5 Wave-1 defense in
// depth, now the EXPLICIT-false opt-out mode: while Params.ReducedDimWarp
// is explicitly false, a WARP-assigned dim whose IV data-dep closure feeds
// a Reduce FAILS CLOSED with a named error — the pipeline must stop loudly
// rather than emit a lowering outside the policy. The decide-side
// feasible-area gate prunes these under the same flag; this is the lower's
// backstop over crafted records. Under DefaultParams (Wave 2) the defense
// stands down: the default admits §3.5 split-k lowering.
func TestLowerWarpReducedDimRejected(t *testing.T) {
	fx := warpReduceFx(t)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 64}, // the REDUCED dim, WARP
	}, nil)
	cache := opt.NewCache()
	cache.Put(rec)
	p := DefaultParams()
	p.ReducedDimWarp = false
	err := (LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g)
	if err == nil || !strings.Contains(err.Error(), "fix4 3.5") {
		t.Fatalf("a WARP-assigned reduced dim must fail closed naming fix4 3.5 (explicit-false opt-out), got %v", err)
	}
	if !strings.Contains(err.Error(), "ReducedDimWarp") {
		t.Fatalf("the defense error must name the ReducedDimWarp capability gate, got %v", err)
	}

	// Positive control: the same shape with the reduced dim SEQ lowers
	// clean under the explicit-false policy (the zero-WARP LNT).
	fx2 := warpReduceFx(t)
	rec2 := recordOf(t, fx2, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 64},
	}, nil)
	runLower(t, fx2.g, rec2)
	assertVerifyAndRootReachable(t, fx2.g)

	// Default-admits policy: with DefaultParams (ReducedDimWarp true — the
	// Wave-2 default) the defense must NOT fire on the same WARP-reduced
	// record — §3.5 split-k lowering is legal under the default.
	if !DefaultParams().ReducedDimWarp {
		t.Fatal("DefaultParams().ReducedDimWarp must be true (Wave 2) for the default-admits policy")
	}
	fx3 := warpReduceFx(t)
	rec3 := recordOf(t, fx3, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 64},
	}, nil)
	cache3 := opt.NewCache()
	cache3.Put(rec3)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache3}, fx3.g); err != nil {
		t.Fatalf("with DefaultParams (ReducedDimWarp=true) the Wave-1 opt-out defense stands down, got %v", err)
	}
}

// TestLaunchBlockDimMaxOverLNTs pins the §5.1 launch derivation seed: the
// launch blockDim is the MAX over the graph's LNTs of (Π warp K) × 32 —
// never the kernel-wide PRODUCT of all WARP LoCs. Two one-dim WARP LNTs
// (K=2 and K=4) launch 128 threads (max(64, 128)), not 256 (2×4×32); a
// CTA-only LNT contributes the empty product 32.
func TestLaunchBlockDimMaxOverLNTs(t *testing.T) {
	g := ir.NewGraph("bdmax")
	u32 := ir.InternScalar(ir.TU32)
	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "d0", nil),
		g.NewInputParam("ctor", ir.RoleDim, "d1", nil),
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}}))
	newChain1 := func(i int) ir.Node {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, 512))
		return g.NewIter("ctor", biv, nil)
	}
	it0, it1 := newChain1(0), newChain1(1)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out,
		[]ir.Node{iv(g, it0), iv(g, it1)}, g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	rec := &Record{Iters: map[ir.Node]IterDecision{}, Sites: map[uint64]SiteDecision{}, Verified: true}
	for _, it := range []ir.Node{it0, it1} {
		biv, err := bivOf(it)
		if err != nil {
			t.Fatal(err)
		}
		rec.Iters[biv] = IterDecision{SoC: SoCWARP, LoC: 2, TileSize: 32}
	}
	rec.Iters[mustBiv(t, it1)] = IterDecision{SoC: SoCWARP, LoC: 4, TileSize: 32}
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g); err != nil {
		t.Fatalf("two one-dim WARP LNTs must lower, got %v", err)
	}
	prov, ok := cache.Get(ProvenanceName).(*Provenance)
	if !ok {
		t.Fatal("the provenance record must be published")
	}
	if prov.MaxThreadsPerCTA != 128 {
		t.Fatalf("launch blockDim = %d, want max(2×32, 4×32) = 128 — NEVER the kernel-wide product 256", prov.MaxThreadsPerCTA)
	}
	if len(prov.LNTs) != 2 {
		t.Fatalf("two LNTs expected, got %d", len(prov.LNTs))
	}
	for _, plan := range prov.LNTs {
		if len(plan.ThreadPortions) != 1 {
			t.Fatalf("each one-dim LNT carries one warp count, got %v", plan.ThreadPortions)
		}
	}
}

// mustBiv resolves an iter's BIV, failing the test on error.
func mustBiv(t *testing.T, it ir.Node) ir.Node {
	t.Helper()
	b, err := bivOf(it)
	if err != nil {
		t.Fatal(err)
	}
	return b
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

// TestProvenancePublished pins P-5 (quick-261022-l1o, fix4 §3.2): after
// tile-lower over a decided LNT, the cache holds the provenance record
// with one per-dim entry per rewritten dim (Between/Inside/Soc/Tile;
// every dim is tiled, T >= 2 — the 2026-10-24 tile-disable ruling — so
// Inside is ALWAYS populated) and one per-LNT MappingPlan of ABSOLUTE
// factorization shapes derived from the record's own decisions: grid =
// the per-CTA-dim LoCs, thread = the per-WARP-dim warp counts.
func TestProvenancePublished(t *testing.T) {
	fx := newChain32("prov", 1024, 2048, 512)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 100, TileSize: 32},
		1: {SoC: SoCWARP, LoC: 8, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 2},
	}, nil)
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g); err != nil {
		t.Fatalf("tile-lower Run: %v", err)
	}
	got := cache.Get(ProvenanceName)
	if got == nil {
		t.Fatal("the provenance record must be published under tiling-provenance")
	}
	prov, ok := got.(*Provenance)
	if !ok {
		t.Fatalf("cache entry under %q is %T, want *Provenance", ProvenanceName, got)
	}

	// Per-dim entries keyed by the DIM InputParams (stable across the pass).
	if len(prov.Dims) != 3 {
		t.Fatalf("provenance must hold one entry per rewritten dim, got %d", len(prov.Dims))
	}
	nest := nestOf(t, fx.g) // M.b, N.b, K.b, M.i, N.i, K.i
	for i, d := range []*struct {
		dim  ir.Node
		soc  SoCType
		tile int
	}{
		{fx.dims[0], SoCTA, 32},
		{fx.dims[1], SoCWARP, 32},
		{fx.dims[2], SoCSequential, 2},
	} {
		e, ok := prov.Dims[d.dim]
		if !ok {
			t.Fatalf("dim %d carries no provenance entry", i)
		}
		if e.Between == nil || e.Inside == nil || e.Soc != d.soc || e.Tile != d.tile {
			t.Fatalf("dim %d entry = %+v, want Between!=nil Inside!=nil Soc=%v Tile=%d", i, e, d.soc, d.tile)
		}
	}
	// The CTA dim's coupled facts (fix4 §3.2): NO ThreadMap slot
	// (CoupledDoSoC == -1 — the flat-rank walk needs no dimension) and
	// CoupledThreads == the LAUNCH blockDim 256 ((Π warp K) × 32 = 8×32).
	if e := prov.Dims[fx.dims[0]]; e.CoupledThreads != 256 || e.CoupledDoSoC != -1 {
		t.Fatalf("CTA dim coupled facts = (%d threads, dosoc %d), want (256, -1 — no thread slot under 3.2)", e.CoupledThreads, e.CoupledDoSoC)
	}

	// Per-LNT plan: keyed by the generated nest root; grid [100] (the CTA
	// dim's absolute LoC); thread [8] (the WARP dim's warp count); the
	// derived launch blockDim rides MaxThreadsPerCTA = 256.
	if len(prov.LNTs) != 1 {
		t.Fatalf("provenance must hold one plan per rewritten LNT, got %d", len(prov.LNTs))
	}
	plan, ok := prov.LNTs[nest[0]]
	if !ok {
		t.Fatal("the per-LNT plan must be keyed by the generated nest root ForLoop")
	}
	if len(plan.GridPortions) != 1 || plan.GridPortions[0] != 100 {
		t.Fatalf("grid shapes = %v, want [100] (the CTA dim's absolute LoC)", plan.GridPortions)
	}
	if len(plan.ThreadPortions) != 1 || plan.ThreadPortions[0] != 8 {
		t.Fatalf("thread shapes = %v, want [8] (the WARP dim's warp count)", plan.ThreadPortions)
	}
	if prov.MaxThreadsPerCTA != 256 {
		t.Fatalf("derived launch blockDim = %d, want 256 (8 warps x 32)", prov.MaxThreadsPerCTA)
	}
	if prov.Name() != ProvenanceName {
		t.Fatalf("Name() = %q, want %q", prov.Name(), ProvenanceName)
	}
}

// TestProvenanceWarpOnlyPlan pins the shape vectors' other branch: an
// LNT with NO CTA dim carries an EMPTY grid vector and its single WARP
// dim's count — grid [], thread [8]; the launch is 8 warps = 256
// threads.
func TestProvenanceWarpOnlyPlan(t *testing.T) {
	fx := newChain32("prov-warp", 512)
	bodyStore(fx)
	rec := recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCWARP, LoC: 8, TileSize: 32},
	}, nil)
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (LowerPass{Params: DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx.g); err != nil {
		t.Fatalf("tile-lower Run: %v", err)
	}
	prov := cache.Get(ProvenanceName).(*Provenance)
	if len(prov.LNTs) != 1 {
		t.Fatalf("one plan expected, got %d", len(prov.LNTs))
	}
	for _, plan := range prov.LNTs {
		if len(plan.GridPortions) != 0 {
			t.Fatalf("grid shapes = %v, want [] (no CTA dim)", plan.GridPortions)
		}
		if len(plan.ThreadPortions) != 1 || plan.ThreadPortions[0] != 8 {
			t.Fatalf("thread shapes = %v, want [8] (the WARP dim's warp count)", plan.ThreadPortions)
		}
	}
	if prov.MaxThreadsPerCTA != 8*32 {
		t.Fatalf("derived launch blockDim = %d, want %d", prov.MaxThreadsPerCTA, 8*32)
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

// assertDosocConst pins that every TUPLE-FORM intrinsic in the subtree
// carries a const-u32 DoSoC operand (the PHASE 0 shape in the generated
// product). The zero-operand FLAT index() query (the 2026-10-23 ruling)
// carries no DoSoC by definition and is skipped.
func assertDosocConst(t *testing.T, n ir.Node) {
	t.Helper()
	for _, x := range flattenSubtree(n) {
		if x.KindOf() != ir.KindRuntimeGridQuery && x.KindOf() != ir.KindRuntimeThreadQuery {
			continue
		}
		ops := ir.OperandsOf(x)
		if len(ops) == 0 {
			continue // the flat index() form — no DoSoC operand
		}
		d := ops[0]
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

// matmulRecord crafts the decide record for matmulUsed at the fix4 §3.3
// WORKED-EXAMPLE SIGNATURE (the same vector TestFix33WorkedExample pins on
// the decide side): M=CTA(T256, R-LOCCAP 4 = ceil(1024/256)), N=SEQ(T256),
// K=WARP(K=2, T32) — both sites promoted, rhs at the MaxCache cap. Under
// the tier rules: lhs[m][k] covers [·][K] with M a thread-varying
// non-covered coordinate → C1b forced FRAG (the per-lane chunk
// ceil(T_k/32) = 1 element, 4 B); rhs[n][k] covers [N][K] with K
// WARP-assigned → SMEM window class (T_n × (K_k×T_k) × 4 = 256×64×4 =
// 65536 B — exactly the 64 KB cap).
func matmulRecord(t *testing.T, fx fixture, lhs, rhs ir.Node) *Record {
	t.Helper()
	return recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 4, TileSize: 256},
		1: {SoC: SoCSequential, LoC: 1, TileSize: 256},
		2: {SoC: SoCWARP, LoC: 2, TileSize: 32},
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
		if n.KindOf() != ir.KindMemStore {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc == ir.StorageGMEM {
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

// loaderVariantOf reads a fill store's DERIVED loader variant (fix3
// D-h: the annotation left the IR; the variant follows the storage class —
// SMEM window = 1, the cooperative CTA loader; FRAG = 2, per-thread).
func loaderVariantOf(t *testing.T, fill ir.Node) float64 {
	t.Helper()
	sc, ok := ir.StorageClassOf(fill)
	if !ok || sc == ir.StorageGMEM {
		t.Fatalf("fill %d is not a tiered store", fill.BasePtr().ID())
	}
	if sc == ir.StorageSMem {
		return 1
	}
	return 2
}

// TestLowerMatmulSiteSurgery pins the PHASE 4 surgery over the §3.3
// worked-example signature (both sites promoted, rhs at the cap): lhs as a
// C1b forced-FRAG site (4 B per-lane FragNew buffer — the ceil(T_k/32) = 1
// element chunk, LoadTileThread fill INSIDE M.inside) and rhs as an SMEM
// WINDOW site (65536 B MemNewSMem window = T_n × (K_k×T_k) × 4 —
// cooperative LoadTileCta fill at the K between-round body): loads erased
// with uses RAUW'd to buffer accesses, verify-clean, typeinfer re-run
// clean, fill placement structurally sound. IR-LEVEL ONLY (ME-05,
// 261021-o61 T5): this pins node surgery over a crafted record under a
// fitting budget — no numerical execution, GPU run, or Materialize/print
// round-trip happens here (the real-record pipeline is
// TestDecideLowerComposition; the print round-trip is
// TestLoweredMatmulPrintGolden).
func TestLowerMatmulSiteSurgery(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec) // default budget: rhs lands exactly at the 64 KB cap

	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("both promoted loads must be erased")
	}
	// Buffers: lhs FRAG = ceil(T_k/32) × 4 B = 4 B (the lane's chunk under
	// C1b); rhs SMEM window = T_n × (K_k×T_k) × 4 = 256 × 64 × 4 = 65536 B
	// (the fix4 factor window — one K_k×T_k slab per warp coordinate).
	lbuf := findBuffer(t, fx.g, ir.KindMemNew, 4)
	rbuf := findBuffer(t, fx.g, ir.KindMemNew, 65536)
	if ir.TypeOf(lbuf) == nil || ir.TypeOf(rbuf) == nil {
		t.Fatal("buffers must carry their bound tile tensor types")
	}
	// The store's value subtree reads the buffers (access rewrite): lhs's
	// access is a FragLoad, rhs's a MemLoadSMem.
	var fragAccess, smemAccess ir.Node
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoad {
			continue
		}
		sc, _ := ir.StorageClassOf(n)
		if sc == ir.StorageFrag {
			fragAccess = n
			if src := ir.OperandsOf(n)[0]; src.BasePtr() != lbuf.BasePtr() {
				t.Fatalf("FRAG access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
			}
		} else if sc == ir.StorageSMem {
			smemAccess = n
			if src := ir.OperandsOf(n)[0]; src.BasePtr() != rbuf.BasePtr() {
				t.Fatalf("SMEM access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
			}
		}
	}
	if fragAccess == nil || smemAccess == nil {
		t.Fatal("the multiply must read exactly one FragLoad and one MemLoadSMem buffer access")
	}
	// R2 access index: the WARP-covered K coordinate carries
	// thread_map.index(0)·T_k + i (K's ThreadMap slot 0 — the WARP-only
	// numbering), the SEQ-covered N coordinate is the plain inside walk.
	smemOps := ir.OperandsOf(smemAccess)[1:]
	if len(smemOps) != 2 {
		t.Fatalf("the rhs access must carry 2 coordinates, got %d", len(smemOps))
	}
	if subtreeHasKind(t, fx.g, smemOps[0], ir.KindRuntimeThreadQuery) {
		t.Fatal("the rhs access's N coordinate (SEQ covered) must be the plain inside walk")
	}
	if !subtreeHasKind(t, fx.g, smemOps[1], ir.KindRuntimeThreadQuery) {
		t.Fatal("the rhs access's K coordinate must carry the ThreadMap rank query (R2 rank-in-index)")
	}
	// Fill variants: lhs FRAG → 2 (LoadTileThread); rhs SMEM → 1
	// (LoadTileCta).
	if v := loaderVariantOf(t, findFillOf(t, fx.g, lbuf)); v != 2 {
		t.Fatalf("lhs fill variant = %v, want 2 (LoadTileThread)", v)
	}
	if v := loaderVariantOf(t, findFillOf(t, fx.g, rbuf)); v != 1 {
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
	assertReachable(t, fx.g, ir.KindMemStore, 1)
	assertReachable(t, fx.g, ir.KindMemStore, 1)
}

// TestLowerC1bFragChunkLocalAccess pins the C1b forced-FRAG ACCESS index on
// a WARP-covered dim (fix4 §3.3: the fragment holds the LANE'S CHUNK of
// ceil(T_d/32) elements, so the access coordinate must be CHUNK-LOCAL —
// insideIV − lane·ceil(T_d/32) — never the raw lane-chunk walk IV, whose
// values START at lane·c and run past the fragment's extent). The raw form
// read out of bounds on the GPU (quick-261027-iq6 probe: compute-sanitizer
// memcheck on the golden matmul reported "Invalid __local__ read of size 4"
// for every lane ≥ 1 — lanes indexed the 1-element Frag past its extent).
func TestLowerC1bFragChunkLocalAccess(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec) // lhs is the C1b forced-FRAG site (K WARP-covered)

	lbuf := findBuffer(t, fx.g, ir.KindMemNew, 4)
	var fragAccess ir.Node
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoad {
			continue
		}
		sc, _ := ir.StorageClassOf(n)
		if sc == ir.StorageFrag {
			fragAccess = n
			if src := ir.OperandsOf(n)[0]; src.BasePtr() != lbuf.BasePtr() {
				t.Fatalf("FRAG access node %d reads a foreign buffer %d", n.BasePtr().ID(), src.BasePtr().ID())
			}
		}
	}
	if fragAccess == nil {
		t.Fatal("the C1b lhs access (FragLoad of the 4 B lane-chunk buffer) must exist")
	}
	coords := ir.OperandsOf(fragAccess)[1:]
	if len(coords) != 1 {
		t.Fatalf("the lhs FRAG access carries %d coords, want 1 (the WARP-covered K dim)", len(coords))
	}
	coord := coords[0]
	if coord.KindOf() != ir.KindBinOp {
		t.Fatalf("the C1b FRAG access coord must be the chunk-local subtraction insideIV − lane·c, got %s", kindOf(coord))
	}
	if code, _ := ir.BinOpCodeOf(coord); code != ir.BinOpSub {
		t.Fatalf("the C1b FRAG access coord must SUBTRACT the chunk origin lane·c, got binop %v", code)
	}
	subOps := ir.OperandsOf(coord)
	if subOps[0] == nil || subOps[0].KindOf() != ir.KindDimVar {
		t.Fatalf("the subtraction's minuend must be the lane-chunk inside DimVar, got %s", kindOf(subOps[0]))
	}
	if code, _ := ir.BinOpCodeOf(subOps[1]); code != ir.BinOpMul {
		t.Fatalf("the subtraction's subtrahend must be the chunk origin mul lane·c, got %s", kindOf(subOps[1]))
	}
	mulOps := ir.OperandsOf(subOps[1])
	if lane := mulOps[0]; lane == nil || lane.KindOf() != ir.KindRuntimeThreadQuery ||
		len(ir.OperandsOf(lane)) != 0 || ir.RuntimeAttributeOf(lane) != "lane_id" {
		t.Fatalf("the chunk origin must carry the zero-operand lane_id query, got %v", lane)
	}
	if v, ok := ir.ConstValueOf(mulOps[1]); !ok || v != 1 {
		t.Fatalf("the chunk size operand = %v, want ceil(T_k/32) = ceil(32/32) = 1", mulOps[1])
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestLowerC1bFragChunkOriginWindow pins the C1b forced-FRAG FILL WINDOW
// base on a WARP-covered dim (fix4 §3.3): the per-thread LoadTileThread
// loads THIS LANE'S CHUNK — ceil(T_d/32) consecutive elements starting at
// the lane's chunk origin (rank·T_d + round) + lane·ceil(T_d/32) — so the
// fill source's covered-dim axis must ADD the lane chunk term to the warp
// window origin. The chunk-LOCAL access (insideIV − lane·c, pinned by
// TestLowerC1bFragChunkLocalAccess) then indexes the fragment the window
// actually filled. A window base of rank·T_d + round alone loads element
// rank·T_d + round for EVERY lane — all 32 lanes of a warp read the same
// GMEM element while their lane-chunk walks consume 32 DISTINCT elements
// (quick-261027-kzv T8b: the split-k matmul e2e failed on the A6000 with
// max abs diff 28.99 — the first cross-warp NUMERICAL exercise of this
// path; every prior pin was structural only).
func TestLowerC1bFragChunkOriginWindow(t *testing.T) {
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec) // lhs is the C1b forced-FRAG site (K WARP-covered)

	lbuf := findBuffer(t, fx.g, ir.KindMemNew, 4)
	lhsFill := findFillOf(t, fx.g, lbuf)
	src := ir.OperandsOf(lhsFill)[0] // the tile-shaped source MemLoad
	axes := ir.OperandsOf(src)[1:]
	if len(axes) != 2 {
		t.Fatalf("lhs fill source must be the 2-axis deref lhs[m][k], got %d axes", len(axes))
	}
	kAxis := axes[1]
	// The window base must carry the warp window origin (the ThreadMap rank
	// query — rank·T_d) AND the lane chunk term (the zero-operand lane_id
	// query multiplied by ceil(T_d/32)) ADDITIVELY: base = (rank·T + round)
	// + lane·c.
	if !subtreeHasKind(t, fx.g, kAxis, ir.KindRuntimeThreadQuery) {
		t.Fatal("the lhs fill's k window base must carry the warp rank query (rank·T + round)")
	}
	var laneMul ir.Node
	var walk func(n ir.Node)
	walk = func(n ir.Node) {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			return
		}
		if n.KindOf() == ir.KindBinOp {
			if code, _ := ir.BinOpCodeOf(n); code == ir.BinOpMul {
				if ops := ir.OperandsOf(n); len(ops) == 2 && ops[0] != nil &&
					ops[0].KindOf() == ir.KindRuntimeThreadQuery &&
					len(ir.OperandsOf(ops[0])) == 0 &&
					ir.RuntimeAttributeOf(ops[0]) == "lane_id" {
					laneMul = n
				}
			}
		}
		for _, op := range ir.OperandsOf(n) {
			walk(op)
		}
	}
	walk(kAxis)
	if laneMul == nil {
		t.Fatal("the lhs fill's k window base must ADD the lane chunk origin lane·ceil(T_d/32) to rank·T + round — the window loads this lane's chunk (quick-261027-kzv T8b GPU defect)")
	}
	mulOps := ir.OperandsOf(laneMul)
	if v, ok := ir.ConstValueOf(mulOps[1]); !ok || v != 1 {
		t.Fatalf("the window's lane chunk size = %v, want ceil(T_k/32) = ceil(32/32) = 1", mulOps[1])
	}
	if code, _ := ir.BinOpCodeOf(kAxis); code != ir.BinOpAdd {
		t.Fatalf("the k window base must be (rank·T + round) + lane·c as an additive chain rooted at Add, got binop %v", code)
	}
	assertVerifyAndRootReachable(t, fx.g)
}

// TestMatmulWorkedCheck pins the R1 worked example at the §3.3 signature:
// the lhs C1b FRAG fill sits structurally INSIDE M.inside (the fill-key
// epoch — the m element changes invalidate the fragment) — its m axis is
// baked as the enclosing m index expression, its k axis is the WARP dim's
// chunk origin (rank·T + round on the AXES path per C1b — the biv path
// cannot carry the per-thread rank term), so the fill store carries ZERO
// trailing round bases.
func TestMatmulWorkedCheck(t *testing.T) {
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec) // default budget: rhs exactly at the 64 KB cap

	lbuf := findBuffer(t, fx.g, ir.KindMemNew, 4)
	lhsFill := findFillOf(t, fx.g, lbuf)
	src := ir.OperandsOf(lhsFill)[0] // the tile-shaped source MemLoad
	axes := ir.OperandsOf(src)[1:]
	if len(axes) != 2 {
		t.Fatalf("lhs fill source must be the 2-axis deref lhs[m][k-round], got %d axes", len(axes))
	}
	// The m coordinate is the enclosing index expression: it references
	// the launch-map machinery (M is the CTA dim), not a bare buffer loop.
	if !subtreeHasKind(t, fx.g, axes[0], ir.KindRuntimeGridQuery) {
		t.Fatal("the lhs fill's m coordinate must be the enclosing m index expression")
	}
	// The k coordinate is the WARP-covered dim's chunk origin: it carries
	// the ThreadMap rank query (C1b: rank·T + round rides the AXES path).
	if !subtreeHasKind(t, fx.g, axes[1], ir.KindRuntimeThreadQuery) {
		t.Fatal("the lhs fill's k coordinate must be the chunk origin rank·T + round (C1b axes path)")
	}
	// The fill store carries NO trailing round bases: the lone covered dim
	// is WARP-covered under a forced-FRAG site, so the biv path is empty
	// and the chunk origin rides the axes (stmtFill's bivDim demands
	// dimvars — the rank term cannot).
	fillBivs := ir.OperandsOf(lhsFill)[2:]
	if len(fillBivs) != 0 {
		t.Fatalf("the C1b WARP-covered fill must carry no biv round bases, got %d", len(fillBivs))
	}
	// STRUCTURAL PLACEMENT (R1/F8), sea-of-nodes (fix1.md §0/D-e): there is
	// no body operand — membership is IV-reference. The fill is inside
	// M.inside because its operand closure references M.inside's IV (the
	// fill-key position), and the whole compute nest (out store included)
	// is controlled by the same reference.
	nest := nestOf(t, fx.g) // M.b, N.b, K.b, M.i, N.i, K.i
	mInside := nest[3]
	mInsideIV := ir.OperandsOf(mInside)[0]
	if !subtreeContains(lhsFill, mInsideIV) {
		t.Fatal("the lhs fill's operand closure must reference the M tile-inside loop's IV (the R1 fill-key position)")
	}
	if !subtreeContains(storeOf(t, fx.g), mInsideIV) {
		t.Fatal("the out store's operand closure must reference the M tile-inside loop's IV (out-store control)")
	}
	// The fix3 D-i safety net: the R-I derivation (innermost referenced
	// loop) picks exactly the mandated position.
	assertFillPositionDerived(t, fx.g, lhsFill, mInside)
}

// storeOf returns the graph's unique live GMEM MemStore.
func storeOf(t *testing.T, g *ir.Graph) ir.Node {
	t.Helper()
	var st ir.Node
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc != ir.StorageGMEM {
			continue // the tiered buffer stores are not GMEM stores
		}
		if st != nil {
			t.Fatal("fixture carries more than one GMEM store")
		}
		st = n
	}
	if st == nil {
		t.Fatal("fixture carries no GMEM store")
	}
	return st
}

// smemWindowFx builds the SMALL-shape SMEM-window fixture: M(CTA LoC=100
// T=8), N(WARP K=2 T=8), K(SEQ T=8) — the fix4 factor window on rhs's
// covered {N,K} is (K_n×T_n) × T_k × 4 = 2×8 × 8 × 4 = 512 B, comfortably
// inside the 64 KB default budget, so the cooperative window path lowers
// un-halved. A second co-traversed load satisfies the overlap admission
// filter (set size >= 2) and stays declined (GMEM).
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
	runLower(t, fx.g, rec) // default 64 KB budget fits the 512 B window

	if !rhs.BasePtr().Erased() {
		t.Fatal("the promoted rhs load must be erased")
	}
	if other.BasePtr().Erased() {
		t.Fatal("the declined co-traversed load must stay GMEM")
	}
	// Window-shaped buffer: (K_n×T_n) × T_k × 4 B = 2×8 × 8 × 4 = 512 B.
	buf := findBuffer(t, fx.g, ir.KindMemNew, 512)
	ty, ok := ir.TypeOf(buf).(*ir.NumericType)
	if !ok || ty.Rank() != 2 {
		t.Fatalf("the rhs window buffer must be a rank-2 tensor type, got %v", ir.TypeOf(buf))
	}
	// The cooperative fill: variant 1 (LoadTileCta), at the K between-round
	// body — CTA-uniform round bases only.
	fill := findFillOf(t, fx.g, buf)
	if v := loaderVariantOf(t, fill); v != 1 {
		t.Fatalf("rhs fill variant = %v, want 1 (LoadTileCta)", v)
	}
	src := ir.OperandsOf(fill)[0]
	for _, ax := range ir.OperandsOf(src)[1:] {
		if subtreeHasKind(t, fx.g, ax, ir.KindRuntimeThreadQuery) {
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
	bases := ir.OperandsOf(fill)[2:]
	if len(bases) != 2 || bases[0].BasePtr() != nBetweenIV.BasePtr() || bases[1].BasePtr() != kBetweenIV.BasePtr() {
		t.Fatalf("the rhs fill's round bases must be the N,K between dimvars, got %v", bases)
	}
	// STRUCTURAL PLACEMENT, sea-of-nodes (fix1.md §0/D-e): there is no body
	// operand — placement is IV-reference. The fill sits at the K between
	// round position (its closure references the K between IV) and, being
	// cooperative, its closure references no element (inside) loop's IV —
	// §4 layering, the HI-02 class.
	if !subtreeContains(fill, kBetweenIV) {
		t.Fatal("the cooperative fill's operand closure must reference the K between loop's IV (R1 SMEM position)")
	}
	for _, l := range []ir.Node{nest[3], nest[4], nest[5]} { // M.i, N.i, K.i
		if subtreeContains(fill, ir.OperandsOf(l)[0]) {
			t.Fatalf("a cooperative fill's operand closure must never reference element loop %d's IV (§4 layering)", l.BasePtr().ID())
		}
	}
	// The fix3 D-i safety net: the R-I derivation picks exactly the
	// mandated position (innermost covered dim's between round).
	assertFillPositionDerived(t, fx.g, fill, kBetween)
	// The access: N coordinate = thread_map.index(N's slot)·T + i with the
	// N dim's OWN ThreadMap slot (0 — the WARP-only numbering of fix4 §3.2:
	// this LNT's CTA dim takes no thread slot), K coordinate = the plain
	// inside walk.
	var access ir.Node
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoad {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc == ir.StorageSMem {
			access = n
		}
	}
	if access == nil {
		t.Fatal("the store's value must read the window buffer through an SMEM-class access")
	}
	coords := ir.OperandsOf(access)[1:]
	if len(coords) != 2 {
		t.Fatalf("the window access must carry 2 coordinates, got %d", len(coords))
	}
	nCoord := coords[0]
	var dosoc float64
	var haveRank bool
	for _, n := range flattenSubtree(nCoord) {
		if n.KindOf() != ir.KindRuntimeThreadQuery {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) == 0 {
			continue // the lane query of the chunk walk (zero-operand form)
		}
		haveRank = true
		if v, ok := ir.ConstValueOf(ops[0]); ok {
			dosoc = v
		}
	}
	if !haveRank || dosoc != 0 {
		t.Fatalf("the N access coordinate must query thread_map.index(0) (N's ThreadMap slot under the WARP-only numbering), got rank=%v dosoc=%v", haveRank, dosoc)
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

// TestLowerDefaultShapeDecline pins the budget floor drop: at the §3.3
// worked-example signature, the crafted rhs's SMEM window
// (T_n×(K_k×T_k)×4 with K=WARP/2) can never fit a budget below its
// minimum legal window — floor tiles {N=2, K=2} price 2×(2×2)×4 = 32 B —
// so the F4 fallback halving hits the floor and DROPS the site to GMEM.
// The lhs C1b FRAG site rides the shared K iter down to the floor tile 2
// (ceil(2/32) = 1 element × 4 B) and STAYS PROMOTED.
func TestLowerDefaultShapeDecline(t *testing.T) {
	fx, lhs, rhs, store := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	p := DefaultParams()
	p.MaxCache = 16 // below rhs's minimum legal window (32 B): floor drop
	runLowerP(t, fx.g, rec, p)

	if rhs.BasePtr().Erased() {
		t.Fatal("under a budget below the minimum legal window the rhs SMEM site must drop back to GMEM")
	}
	if !lhs.BasePtr().Erased() {
		t.Fatal("the lhs FRAG site must stay promoted at the floor tile 2 (4 B per-lane chunk)")
	}
	// No SMEM material was emitted at all; the FRAG site survived.
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if sc, ok := ir.StorageClassOf(n); ok && sc == ir.StorageSMem {
			t.Fatalf("a dropped-SMEM kernel must carry no SMEM-class nodes; found %s node %d", n.KindOf(), n.BasePtr().ID())
		}
	}
	findBuffer(t, fx.g, ir.KindMemNew, 4) // lhs at the shared K floor tile 2
	// The store's value still reads the ORIGINAL rhs load (untouched GMEM);
	// the lhs is served from the FRAG buffer.
	liveLoads := 0
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoad {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc == ir.StorageGMEM {
			liveLoads++
		}
	}
	if liveLoads != 1 {
		t.Fatalf("exactly the original rhs GMEM access must remain live in the store's value, found %d live loads", liveLoads)
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
	buf := findBuffer(t, fx.g, ir.KindMemNew, 128) // 4 B × T_k(32)
	fill := findFillOf(t, fx.g, buf)
	// Sea-of-nodes (fix1.md §0): no loop carries an expression operand — the
	// deleted HI-02 shape (a fill hanging off a loop) is unrepresentable.
	for _, l := range loopsOf(fx.g) {
		for i, op := range ir.OperandsOf(l) {
			switch op.KindOf() {
			case ir.KindDimVar, ir.KindBoundedDimVar, ir.KindKeyedAttribute:
			default:
				t.Fatalf("loop %d operand %d (%s) is an expression link — loops carry no body (fix1.md §0)",
					l.BasePtr().ID(), i, op.KindOf())
			}
		}
	}
	nest := nestOf(t, fx.g) // M.b, K.b, M.i, K.i
	kInside := nest[3]
	kInsideIV := ir.OperandsOf(kInside)[0]
	kBetweenIV := ir.OperandsOf(nest[1])[0]
	// The buffer ACCESS holds the K inside membership after the RAUW (the
	// standard splice: the promoted load's uses moved onto the access).
	var access ir.Node
	for _, n := range fx.g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindMemLoad {
			access = n
		}
	}
	if access == nil {
		t.Fatal("the buffer access must survive the rewrite")
	}
	if !subtreeContains(access, kInsideIV) {
		t.Fatalf("the buffer access must reference the K inside loop's IV (membership, fix1.md §0), got %v", access)
	}
	if src := ir.OperandsOf(access)[0]; src.BasePtr() != buf.BasePtr() {
		t.Fatal("the access must read the site's own buffer")
	}
	// The fill sits at its mandated host (the innermost covered dim's
	// between round): its closure references the K between IV and no
	// element (inside) loop's IV — §4 layering.
	if !subtreeContains(fill, kBetweenIV) {
		t.Fatal("the fill's operand closure must reference the K between loop's IV (the fallback fill position)")
	}
	if subtreeContains(fill, kInsideIV) || subtreeContains(fill, ir.OperandsOf(nest[2])[0]) {
		t.Fatal("the fill's operand closure must reference no inside loop's IV (§4 layering)")
	}
	// The fix3 D-i safety net: the R-I derivation picks exactly the
	// mandated position (the degenerate fallback: no above-tie INS dim).
	assertFillPositionDerived(t, fx.g, fill, nest[1])
	if vs := ir.Verify(fx.g); len(vs) != 0 {
		t.Fatalf("verifier must report zero violations, got %v", vs)
	}
}

// TestBudgetFallback pins §6's fallback (F4 + kernel-wide budgets): with a
// forced-tiny MaxCache the halving loop terminates — tile sizes halve while
// over budget, with the 2026-10-24 ruling's FLOOR OF 2 (an override below 2
// is never written: the picked site whose halved covered dim would fall
// below 2 is DROPPED to GMEM, never demoted to a non-tiling schedule); with
// the §3.3-signature budget (rhs exactly at the MaxCache cap) the fallback
// is a no-op; the FRAG budget (F6) participates through the same loop.
// NOTE (fix4 §3.3): the halve-then-drop path is the UNREACHABLE fail-closed
// backstop — the record is feasible by construction via decide-side
// eviction — these pins exercise the defense on hand-crafted records.
func TestBudgetFallback(t *testing.T) {
	// No-op case: the §3.3 signature lands rhs EXACTLY at the 64 KB cap —
	// the default budget fits, nothing halves.
	fx, lhs, rhs, _ := matmulUsed()
	rec := matmulRecord(t, fx, lhs, rhs)
	runLower(t, fx.g, rec)
	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("with the cap-fitting §3.3 budget both sites must promote un-halved")
	}
	findBuffer(t, fx.g, ir.KindMemNew, 65536) // rhs at the cap, no halving

	// Halving survival: tile-8 decisions under a 256-byte budget halve the
	// rhs covered dims {N,K} to (4,4) — 4×(2×4)×4 = 128 B — while the lhs
	// C1b chunk stays ceil/32-pinned at 4 B; both stay promoted (the FRAG
	// sum is well inside MaxFragPerThread).
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
	p.MaxCache = 256
	if err := (LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, fx2.g); err != nil {
		t.Fatalf("budget-fallback run: %v", err)
	}
	if !lhs2.BasePtr().Erased() || !rhs2.BasePtr().Erased() {
		t.Fatal("both sites must stay promoted after halving under a 256-byte budget")
	}
	findBuffer(t, fx2.g, ir.KindMemNew, 4)   // lhs: ceil(4/32) = 1 element chunk
	findBuffer(t, fx2.g, ir.KindMemNew, 128) // rhs at window (N=4, K=2×4)
	assertVerifyAndRootReachable(t, fx2.g)

	// F4 floor drop: a 16-byte budget can never fit the WARP window (the
	// floor-tile-2 minimum is 2×(2×2)×4 = 32 B) — rhs halving reaches the
	// floor and DROPS; the lhs FRAG site rides the shared K iter to the
	// floor tile 2 (4 B, inside MaxFragPerThread) and STAYS PROMOTED.
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
	p2b.MaxCache = 16
	if err := (LowerPass{Params: p2b}).Run(&opt.Context{Pass: "tile-lower", Cache: cache2b}, fx2b.g); err != nil {
		t.Fatalf("budget-drop run: %v", err)
	}
	if rhs2b.BasePtr().Erased() {
		t.Fatal("the rhs SMEM site must drop back to GMEM under a 16-byte budget (F4 floor drop)")
	}
	if !lhs2b.BasePtr().Erased() {
		t.Fatal("the lhs FRAG site must stay promoted at the floor tile 2 (4 B per-lane chunk)")
	}
	findBuffer(t, fx2b.g, ir.KindMemNew, 4)
	assertVerifyAndRootReachable(t, fx2b.g)

	// Budget of 1: the SMEM site drops; the FRAG site survives at the
	// floor tile 2; the pass still terminates clean.
	fx3, lhs3, rhs3, _ := matmulUsed()
	rec3 := matmulRecord(t, fx3, lhs3, rhs3)
	cache3 := opt.NewCache()
	cache3.Put(rec3)
	p3 := DefaultParams()
	p3.MaxCache = 1
	if err := (LowerPass{Params: p3}).Run(&opt.Context{Pass: "tile-lower", Cache: cache3}, fx3.g); err != nil {
		t.Fatalf("maximal fallback run: %v", err)
	}
	if rhs3.BasePtr().Erased() {
		t.Fatal("with MaxCache 1 the rhs SMEM site must drop")
	}
	if !lhs3.BasePtr().Erased() {
		t.Fatal("with MaxCache 1 the lhs FRAG site must still promote at the floor tile 2 (4 B)")
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

	t.Run("fill ordering missing chain edge", func(t *testing.T) {
		g := ir.NewGraph("inj-order")
		u32 := ir.InternScalar(ir.TU32)
		dim := g.NewInputParam("ctor", ir.RoleDim, "d", nil)
		ivn := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		loop := g.NewForLoop("ctor", ivn, g.NewBoundedDimVar("ctor", ivn, g.NewConst("ctor", u32, 8)),
			g.NewConst("ctor", u32, 0))
		buf := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		src := g.NewMemLoad("ctor", tensorParam(g, "t", 1), []ir.Node{ivn})
		fill := g.NewMemStoreSMem("ctor", src, buf)
		// An access on the buffer WITHOUT the chain edge to the fill: the
		// F8 splice edge is the structural ORDERING fact (fix1.md D-e), and
		// this is the 7.2a non-vacuity pin — a fill whose buffer has live
		// reads that do not chain back to it must fail.
		access := g.NewMemLoadSMem("ctor", buf, ivn)
		lr := &lowerRun{g: g, fills: []*siteFill{{siteID: 3, src: src, store: fill, buffer: buf, host: loop, variant: loadTileCta}}}
		if err := lr.checkFillPlacement(); err == nil || !strings.Contains(err.Error(), "not a chain predecessor") {
			t.Fatalf("an access without the fill's chain edge must trip ORDERING, got %v", err)
		}
		_ = access
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
		// The fill's coordinates ride the inside (element-loop) walk's IV
		// — the HI-02 shape, unrepresentable as a body operand post-fix1.md
		// and caught by the LAYERING-IV recast (D-e).
		loop := g.NewForLoop("ctor", ivn, g.NewBoundedDimVar("ctor", ivn, g.NewConst("ctor", u32, 8)))
		lr := &lowerRun{
			g: g,
			rewrites: []*lntRewrite{{dims: []*dimRewrite{{
				soc: SoCSequential, inside: loop, between: loop, betweenIV: ivn, insideIV: ivn,
			}}}},
			fills: []*siteFill{{siteID: 7, src: src, store: fill, buffer: buf, host: loop, variant: loadTileCta}},
		}
		if err := lr.checkFillPlacement(); err == nil || !strings.Contains(err.Error(), "inside-loop dimvar") {
			t.Fatalf("a cooperative fill referencing an element loop's IV must trip the layering check, got %v", err)
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

	t.Run("fill without its derivation facts", func(t *testing.T) {
		g := ir.NewGraph("inj-attr")
		u32 := ir.InternScalar(ir.TU32)
		buf := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		// fix3 D-h: the derivation facts are the tiered class + the GMEM
		// tile-load source — inject a fill whose value is NOT a load (the
		// annotation-era injection equivalent).
		fill := g.NewMemStoreSMem("ctor", g.NewConst("ctor", u32, 0), buf)
		lr := &lowerRun{g: g, fills: []*siteFill{{siteID: 8, store: fill}}}
		if err := lr.checkNoBarriers(); err == nil || !strings.Contains(err.Error(), "stores no GMEM tile load") {
			t.Fatalf("a fill without its derivation facts must trip the barrier check, got %v", err)
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

	t.Run("frag cta-covered refusal", func(t *testing.T) {
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
		if err := lr.checkFragSites(rec, nil); err == nil || !strings.Contains(err.Error(), "R2 refusal") {
			t.Fatalf("a FRAG site covering a CTA dim must trip the R2 refusal, got %v", err)
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
// the decide→verify→lower triple composes — tiling → tiling-verify →
// tile-lower runs clean over the matmul worked example with the REAL
// searched record (no crafted entries; the F7 stamp is the verify pass's
// own product), ending verifier-clean and typable. The searched record's
// own outcome is pinned: under the C1b forced-FRAG precedence
// (2026-10-23), the winning schedule promotes lhs as a per-thread FRAGMENT
// and declines rhs with its reason.
func TestDecideLowerComposition(t *testing.T) {
	p := DefaultParams()
	fx, lhs, rhs, _ := matmulUsed()
	cache := opt.NewCache()
	ctx := &opt.Context{Pass: "tiling", Cache: cache}
	if err := (Pass{Params: p}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tiling decide: %v", err)
	}
	rec, ok := cache.Get(RecordName).(*Record)
	if !ok {
		t.Fatalf("cache entry under %q is %T, want *Record", RecordName, cache.Get(RecordName))
	}

	// The REAL searched record over the worked example, RE-DERIVED under
	// fix5 R3 (quick-261029-v9k — the actual post-R3 search output, pinned
	// per the "re-derive, never weaken" rule): the R3 ranking makes N the
	// per-LNT WARP winner (L1 extent — N=2048 is the largest const extent
	// of the 1024/2048/512 fixture), so the pre-R3 winner's K=WARP leaves
	// the feasible area and the new argmax is M=CTA/4/256 (the R-LOCCAP cap
	// at T=256: ceil(1024/256) = 4), N=WARP/2/256, K=SEQ/1/32, blockDim 64.
	// lhs (tie N, covered {K}, forced FRAG under the C1b precedence — M is a
	// thread-varying non-covered coordinate) promotes as a T_k×4 = 128 B
	// per-thread window with per-thread reuse 8 (the N WARP dim is not in
	// lhs's subscripts: sharing factor 32 across the 256-element N walk ->
	// ceil? no — the lane chunk ceil(256/32) = 8 trips, §2.7 per-thread);
	// rhs (covered {N WARP, K SEQ}) promotes as the (K_n×T_n)×T_k×4 =
	// 512×32×4 = 65536 B SMEM window at the MaxCache cap with COLLECTIVE
	// reuse T_M = 256 (§2.4: CTA trips, blockDim-independent; sharing 1
	// since the only WARP dim N ∈ rhs's subscripts). The decide pass
	// publishes the record UNSTAMPED (F7: tiling-verify owns the stamp).
	dM, dN, dK := decisionOf(t, rec, fx.iters[0]), decisionOf(t, rec, fx.iters[1]), decisionOf(t, rec, fx.iters[2])
	if dM.SoC != SoCTA || dM.LoC != 4 || dM.TileSize != 256 ||
		dN.SoC != SoCWARP || dN.LoC != 2 || dN.TileSize != 256 ||
		dK.SoC != SoCSequential || dK.LoC != 1 || dK.TileSize != 32 {
		t.Fatalf("searched schedule = M%s/%d/%d N%s/%d/%d K%s/%d/%d, want the probed winner (M=CTA/4/256 N=WARP/2/256 K=SEQ/1/32 — re-derived under R3, derivation in the test comment)",
			dM.SoC, dM.LoC, dM.TileSize, dN.SoC, dN.LoC, dN.TileSize, dK.SoC, dK.LoC, dK.TileSize)
	}
	sl, ok := rec.SiteDecisionOf(lhs)
	if !ok {
		t.Fatal("the searched record must carry lhs's site entry")
	}
	if !sl.Promoted || sl.Class != ClassFRAG || sl.Bytes != 128 || sl.Reuse != 8 {
		t.Fatalf("searched lhs decision = %+v, want promoted %s of 128 B (T_k = 32 x 4 B, the whole K SEQ tile per thread) with reuse 8 (the N WARP lane chunk ceil(256/32))",
			sl, ClassFRAG)
	}
	sr, ok := rec.SiteDecisionOf(rhs)
	if !ok {
		t.Fatal("the searched record must carry rhs's site entry")
	}
	if !sr.Promoted || sr.Class != ClassSMEM || sr.Bytes != 65536 || sr.Reuse != 256 {
		t.Fatalf("searched rhs decision = %+v, want promoted %s of 65536 B (at the MaxCache cap) with collective reuse 256 (= T_M)",
			sr, ClassSMEM)
	}
	if rec.Verified {
		t.Fatal("the decide pass must publish the record unstamped (F7)")
	}

	ctx.Pass = "tiling-verify"
	if err := (VerifyPass{Params: p}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tiling-verify: %v", err)
	}
	if !rec.Verified {
		t.Fatal("tiling-verify must stamp the record (the F7 stamp tile-lower consumes)")
	}
	ctx.Pass = "tile-lower"
	if err := (LowerPass{Params: p}).Run(ctx, fx.g); err != nil {
		t.Fatalf("tile-lower over the real record: %v", err)
	}

	// Lowered products: BOTH sites promote — lhs as a 128 B FragNew buffer
	// (variant 2, the per-thread whole-K-tile window T_k = 32 x 4 B under
	// the R3 winner; pre-R3 it was the 4 B K lane chunk) and rhs as a
	// 65536 B MemNewSMem factor window (variant 1, the cooperative loader)
	// — and the loads are erased.
	if !lhs.BasePtr().Erased() || !rhs.BasePtr().Erased() {
		t.Fatal("both promoted loads must be erased")
	}
	lbuf := findBuffer(t, fx.g, ir.KindMemNew, 128)
	rbuf := findBuffer(t, fx.g, ir.KindMemNew, 65536)
	if v := loaderVariantOf(t, findFillOf(t, fx.g, lbuf)); v != 2 {
		t.Fatalf("the real-record fragment fill variant = %v, want 2 (LoadTileThread)", v)
	}
	if v := loaderVariantOf(t, findFillOf(t, fx.g, rbuf)); v != 1 {
		t.Fatalf("the real-record smem fill variant = %v, want 1 (LoadTileCta)", v)
	}

	if vs := ir.Verify(fx.g); len(vs) != 0 {
		t.Fatalf("verifier after the triple: %v", vs)
	}
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run after the triple: %v", err)
	}
}

// TestLoweredMatmulPrintGolden pins ME-05's missing round-trip: the
// decide→verify→lower triple over the matmul worked example with the REAL
// searched record, followed by Materialize + Print, renders byte-exactly —
// the first golden covering a tile-lowered print (the fill store with its
// loader-annotation suffix, the window buffer, the lowered index
// expressions on the surviving GMEM store, the declined rhs's GMEM load).
//
// ⚠ DEFERRED CONTRACT, deliberate (261021-o61 T3/T5): Materialize schedules
// one top-level statement per live EFFECTFUL node in allocation order —
// ForLoop is not an effect kind and no statement tree references a loop
// node, so the loop NEST renders nowhere, and the fill prints as a
// top-level statement in allocation order rather than nested inside its
// host loop. Statement-level nesting of effect statements inside the loop
// nest is a DEFERRED Materialize/codegen contract: when it lands, this
// golden changes as a CONSCIOUS contract change (golden update in the same
// commit), not as a regression. Until then the loop-owned omissions are
// asserted complete below (nothing outside the loop nest is dropped).
func TestLoweredMatmulPrintGolden(t *testing.T) {
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
	if err := fx.g.Materialize(); err != nil {
		t.Fatalf("Materialize over the lowered graph: %v", err)
	}
	out := ir.Print(fx.g)
	golden.Assert(t, "testdata/tile_lowered_matmul.ir", []byte(out))

	// Round-trip honesty (ME-05): everything the CURRENT statement contract
	// prints is well-formed — no placeholder fallback may survive where a
	// coordinate or operand should have rendered (the store's lowered axes
	// are computed index expressions now, not dim params; the pre-T5
	// printer rendered them all as [@?] — silently losing every live
	// coordinate).
	for _, marker := range []string{"<nil>", "@?", "%?", "%<"} {
		if strings.Contains(out, marker) {
			t.Fatalf("the print carries placeholder marker %q — a live node's rendering was lost or corrupted:\n%s", marker, out)
		}
	}
	// The deferred contract owns EVERYTHING the print omits: each node that
	// is LIVE (reachable from the end node over data+payload edges —
	// Materialize's own liveness test; retired-rewrite debris with zero live
	// uses is out of scope) must be either printed or owned by a live
	// for-loop (the nest's fixed slots: dimvar, bound, body, annotations) —
	// nothing outside the loop nest is dropped silently.
	num := ir.ComputeNumbering(fx.g)
	lr := &lowerRun{g: fx.g}
	live := lr.reachableFrom(fx.g.End())
	loopOwned := make(map[*ir.Base]bool)
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindForLoop {
			continue
		}
		for b := range lr.reachableFrom(n) {
			loopOwned[b] = true
		}
	}
	if len(loopOwned) == 0 {
		t.Fatal("the lowered graph must carry live for-loops")
	}
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || !live[n.BasePtr()] {
			continue // dead debris (e.g. a retired dimvar's init const): DCE's prey
		}
		if _, printed := num.Num(n); printed {
			continue
		}
		if !loopOwned[n.BasePtr()] {
			t.Fatalf("live node %d (%s) is neither printed nor owned by the deferred loop-nest contract — silently dropped",
				n.BasePtr().ID(), n.KindOf())
		}
	}
	// Tripwire for the deferred contract itself: no loop renders today. When
	// statement nesting lands, this fires — update the pin AND the golden
	// consciously in the same commit.
	if strings.Contains(out, "for(") {
		t.Fatal("a for-loop rendered: the deferred statement-nesting contract landed — update this pin and the golden as a conscious contract change")
	}
}

// ---------------------------------------------------------------------------
// quick-261029-v9k (fix5 T2 R3): WARP-on-a-NON-reduced-dim lowering flow
// ---------------------------------------------------------------------------

// r3ScaleCopyUsed builds the scale_copy schedule class with real data flow:
// R=512, C=512, K=4 chain; x[r][c][k], z[r][c][k] (the C1b forced-FRAG
// sites — R is a thread-varying non-covered coordinate of their INS), a
// w[c][k] site (SMEM window — covered {C, K}, no thread-varying
// non-covered INS dim), a reduce over K, and the single-owner [R][C]
// store. Under the R3 collapse record (R=CTA, C=WARP, K=SEQ) this is the
// exact configuration the suite never exercised before T2: the WARP digit
// on a covered NON-reduced dim.
func r3ScaleCopyUsed(t *testing.T) (fx fixture, x, z, w, store ir.Node) {
	t.Helper()
	fx = newChain32("r3sc", 512, 512, 4) // R, C, K
	xP := fx.g.NewInputParam("ctor", ir.RoleTensor, "x",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	zP := fx.g.NewInputParam("ctor", ir.RoleTensor, "z",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}, {Name: "d2"}}))
	wP := fx.g.NewInputParam("ctor", ir.RoleTensor, "w",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d1"}, {Name: "d2"}}))
	out2 := fx.g.NewInputParam("ctor", ir.RoleTensor, "out2",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}, {Name: "d1"}}))
	iv := func(i int) ir.Node { return fx.g.NewInductionVariable("ctor", fx.iters[i]) }
	x = fx.g.NewMemLoad("ctor", xP, []ir.Node{iv(0), iv(1), iv(2)})
	z = fx.g.NewMemLoad("ctor", zP, []ir.Node{iv(0), iv(1), iv(2)})
	w = fx.g.NewMemLoad("ctor", wP, []ir.Node{iv(1), iv(2)})
	prod := fx.g.NewBinOp("ctor", ir.BinOpMul, fx.g.NewBinOp("ctor", ir.BinOpMul, x, z), w)
	// The reduce pipes the Map's tensor output (rank 3 [R,C,K] — the
	// map/reduce idiom of warpReduceFx/reduce3_test; a scalar pipe fails
	// typeinfer's "nothing to reduce"), dropping the LAST dim (K).
	mp := fx.g.NewMap("ctor", []ir.Node{x, z, w}, ir.NewLambda(nil, prod),
		append([]ir.Node(nil), fx.dims...))
	acc := fx.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := fx.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := fx.g.NewBinOp("ctor", ir.BinOpAdd, fx.g.NewParamRef("ctor", acc), fx.g.NewParamRef("ctor", now))
	val := fx.g.NewReduce("ctor", fx.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	env := ir.NewEnv()
	store = fx.g.NewMemStoreChained("ctor", env, out2, []ir.Node{iv(0), iv(1)}, val)
	fx.g.NewReturnChained("ctor", env, []ir.Node{out2}, store)
	return fx, x, z, w, store
}

// r3ScaleCopyRecord crafts the R3 collapse record for r3ScaleCopyUsed:
// R=CTA(16,T256), C=WARP(2,T32), K=SEQ(T32) — the qwo-derived winner
// class on the const-K scale_copy shape. Site tier derivations (the judge
// model of fix4 §3.3): x/z covered {K, C} with R a thread-varying
// non-covered INS dim -> C1b forced FRAG, per-thread window
// ceil(T_c/32)=1 x T_k=32 x 4 B = 128 B; w covered {C, K} (tie nil) with
// C WARP-covered and no thread-varying non-covered INS dim -> SMEM window
// (K_c x T_c) x T_k x 4 = 64x32x4 = 8192 B.
func r3ScaleCopyRecord(t *testing.T, fx fixture, x, z, w ir.Node) *Record {
	t.Helper()
	return recordOf(t, fx, map[int]IterDecision{
		0: {SoC: SoCTA, LoC: 16, TileSize: 256},
		1: {SoC: SoCWARP, LoC: 2, TileSize: 32},
		2: {SoC: SoCSequential, LoC: 1, TileSize: 32},
	}, map[uint64]SiteDecision{
		x.BasePtr().ID(): {Promoted: true, Tie: fx.iters[0], Class: ClassFRAG},
		z.BasePtr().ID(): {Promoted: true, Tie: fx.iters[0], Class: ClassFRAG},
		w.BasePtr().ID(): {Promoted: true, Tie: nil, Class: ClassSMEM},
	})
}

// laneChunkMul reports whether n's subtree contains a lane_id x c mul
// (the zero-operand lane query multiplied by a chunk size).
func laneChunkMul(n ir.Node) bool {
	if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
		return false
	}
	if n.KindOf() == ir.KindBinOp {
		if code, _ := ir.BinOpCodeOf(n); code == ir.BinOpMul {
			if ops := ir.OperandsOf(n); len(ops) == 2 && ops[0] != nil &&
				ops[0].KindOf() == ir.KindRuntimeThreadQuery &&
				len(ir.OperandsOf(ops[0])) == 0 &&
				ir.RuntimeAttributeOf(ops[0]) == "lane_id" {
				return true
			}
		}
	}
	for _, op := range ir.OperandsOf(n) {
		if laneChunkMul(op) {
			return true
		}
	}
	return false
}

// chunkLocalCoord reports whether n is the C1b chunk-LOCAL coordinate
// (inside - lane x c) — the iq6 access form.
func chunkLocalCoord(n ir.Node) bool {
	if n == nil || n.KindOf() != ir.KindBinOp {
		return false
	}
	if code, _ := ir.BinOpCodeOf(n); code != ir.BinOpSub {
		return false
	}
	ops := ir.OperandsOf(n)
	if len(ops) != 2 {
		return false
	}
	return laneChunkMul(ops[1])
}

// TestLowerR3WarpNonReducedFlow is the fix5 §3 guard pin for T2: the
// WARP-on-a-NON-reduced-dim layout (the R3 collapse schedule class:
// covered dims INCLUDING the WARP dim, the reduced dim SEQ) lowers through
// tile-lower EXACTLY like the K-WARP split-k class the suite exercises —
// the machinery is dim-agnostic, only the CHOICE of dim is new (fix5 §3
// "already flows"). Locked: the WARP template on C (between stride
// K_d x T_d, lane-chunk inside walk with the F1-clamped bound), the C1b
// forced-FRAG chunk-LOCAL access (the iq6 form) and the kzv chunk-origin
// fill window on the FRAG sites, the SMEM window access with the warp
// rank in the WARP-covered coordinate (R2 rank-in-index), the per-thread
// LoadTileThread FRAG fill and the cooperative LoadTileCta SMEM fill, the
// single-owner GMEM store, and verifier + typeinfer health. The one-tier
// reducer face (reduced dim SEQ -> SeqReducer, zero WarpReducer/
// GlobalReducer/ThreadMapping-reducer operands) is pinned at the
// final-lower level by quick-261029-qwo's TestReducerOneTierSeqLadder
// (unchanged, kept green by the full suite); the multi-tier WarpReducer
// path's byte-stability rides on the same suite. Every assertion here
// passes on the UNCHANGED lower code — confirming the fix5 §3 expectation
// (a RED pin would have been a minimal mechanical defect in T2 scope).
func TestLowerR3WarpNonReducedFlow(t *testing.T) {
	fx, x, z, w, store := r3ScaleCopyUsed(t)
	rec := r3ScaleCopyRecord(t, fx, x, z, w)
	runLower(t, fx.g, rec)

	// The WARP template on the NON-reduced C dim: between stride K_d x T_d
	// = 2 x 32 = 64, in warp units (soc 2, loc = K_d = 2); inside is the
	// lane-chunk walk (step 1, soc 0).
	loops := nestOf(t, fx.g)
	var cBetween, cInside ir.Node
	for _, l := range loops {
		if ann := loopAnnotations(l); ann["soc"] == 2 {
			cBetween = l
			break
		}
	}
	// The multi-dim canonical nest orders ALL betweens before the insides
	// ([R.b, C.b, K.b, R.i, C.i, K.i] — TestLowerMultiDimTemplates), so C's
	// inside is NOT adjacent to C's between: it is the soc-0 step-1 loop
	// whose IV init is the lane x c mul.
	for _, l := range loops {
		if loopAnnotations(l)["soc"] != 0 || dimvarStep(t, l) != 1 {
			continue
		}
		liv := ir.OperandsOf(l)[0]
		lin := ir.OperandsOf(liv)[1]
		if lin != nil && lin.KindOf() == ir.KindBinOp {
			if code, _ := ir.BinOpCodeOf(lin); code == ir.BinOpMul && laneChunkMul(lin) {
				cInside = l
				break
			}
		}
	}
	if cBetween == nil {
		t.Fatal("the C WARP between loop must exist (soc 2)")
	}
	if got := dimvarStep(t, cBetween); got != 64 {
		t.Fatalf("C WARP between step = %v, want K_d x T_d = 2 x 32 = 64", got)
	}
	if ann := loopAnnotations(cBetween); ann["loc"] != 2 {
		t.Fatalf("C WARP between loc = %v, want K_d = 2 (warp units)", ann["loc"])
	}
	if cInside == nil || dimvarStep(t, cInside) != 1 || loopAnnotations(cInside)["soc"] != 0 {
		t.Fatalf("C WARP inside = %v, want the lane-chunk walk (step 1, soc 0)", cInside)
	}
	// Lane-chunk init: lane_id x ceil(T_c/32) = lane x 1.
	iv := ir.OperandsOf(cInside)[0]
	init := ir.OperandsOf(iv)[1]
	if init == nil || init.KindOf() != ir.KindBinOp {
		t.Fatalf("C inside IV init = %v (%s), want the lane x c mul", init, kindOf(init))
	}
	if code, _ := ir.BinOpCodeOf(init); code != ir.BinOpMul {
		t.Fatalf("C inside IV init must be a mul (lane x c), got code %v", code)
	}
	mulOps := ir.OperandsOf(init)
	if v, ok := ir.ConstValueOf(mulOps[1]); !ok || v != 1 {
		t.Fatalf("C inside chunk size = %v, want ceil(32/32) = 1", mulOps[1])
	}
	if lane := mulOps[0]; lane == nil || lane.KindOf() != ir.KindRuntimeThreadQuery ||
		len(ir.OperandsOf(lane)) != 0 || ir.RuntimeAttributeOf(lane) != "lane_id" {
		t.Fatalf("C inside chunk lane operand = %v, want the zero-operand lane_id query", lane)
	}
	// Lane-chunk bound: umin((lane+1) x c, F1 tile_max).
	bound := boundOf(t, cInside)
	if bound.KindOf() != ir.KindCall || ir.CallNameOf(bound) != "umin" {
		t.Fatalf("C inside bound must be a umin call, got %s", kindOf(bound))
	}

	// Buffers: two 128 B FRAG windows (x, z) + one 8192 B SMEM window (w);
	// the FRAG fills are per-thread (variant 2) and their source's C axis
	// carries the kzv chunk-origin window (the lane chunk term added to
	// the warp window origin — each lane fills ITS OWN chunk).
	var fragBufs []ir.Node
	smemSeen := false
	for _, n := range fx.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemNew {
			continue
		}
		v, ok := ir.ConstValueOf(ir.OperandsOf(n)[0])
		if !ok {
			continue
		}
		switch v {
		case 128:
			fragBufs = append(fragBufs, n)
		case 8192:
			smemSeen = true
		}
	}
	if len(fragBufs) != 2 || !smemSeen {
		t.Fatalf("buffers: %d x 128 B FRAG + smem %v, want exactly 2 FRAG + 1 x 8192 B SMEM", len(fragBufs), smemSeen)
	}
	for _, buf := range fragBufs {
		fill := findFillOf(t, fx.g, buf)
		if v := loaderVariantOf(t, fill); v != 2 {
			t.Fatalf("FRAG fill variant = %v, want 2 (LoadTileThread)", v)
		}
		src := ir.OperandsOf(fill)[0]
		if !laneChunkMul(src) {
			t.Fatalf("the FRAG fill source %d must carry the lane chunk term (kzv chunk-origin window: rank x T + round + lane x c)", fill.BasePtr().ID())
		}
	}

	// Access rewrite: x/z read their FRAG windows with the CHUNK-LOCAL C
	// coordinate (inside_C - lane x 1 — the iq6 form, here on a
	// NON-reduced dim); w reads its SMEM window with the warp rank in the
	// WARP-covered C coordinate (R2 rank-in-index) and the plain walk in
	// the SEQ-covered K coordinate (no thread query).
	fragChunkLocal, smemRanked := 0, 0
	seenAccess := make(map[*ir.Base]bool)
	for _, n := range flattenSubtree(ir.OperandsOf(store)[0]) {
		if n.KindOf() != ir.KindMemLoad || seenAccess[n.BasePtr()] {
			continue // shared Map-body nodes flatten twice — count once
		}
		seenAccess[n.BasePtr()] = true
		sc, _ := ir.StorageClassOf(n)
		hasChunkLocal, hasRank := false, false
		for _, c := range ir.OperandsOf(n)[1:] {
			if chunkLocalCoord(c) {
				hasChunkLocal = true
			}
			if subtreeHasKind(t, fx.g, c, ir.KindRuntimeThreadQuery) {
				hasRank = true
			}
		}
		switch sc {
		case ir.StorageFrag:
			if !hasChunkLocal {
				t.Fatalf("FRAG access %d must carry the chunk-local C coordinate (inside - lane x c)", n.BasePtr().ID())
			}
			fragChunkLocal++
		case ir.StorageSMem:
			if !hasRank {
				t.Fatalf("SMEM access %d must carry the warp rank in the WARP-covered C coordinate", n.BasePtr().ID())
			}
			smemRanked++
		}
	}
	if fragChunkLocal != 2 || smemRanked != 1 {
		t.Fatalf("access rewrite: %d chunk-local FRAG + %d rank-in-index SMEM, want 2 + 1", fragChunkLocal, smemRanked)
	}

	// Loads erased, single-owner GMEM store survives.
	if !x.BasePtr().Erased() || !z.BasePtr().Erased() || !w.BasePtr().Erased() {
		t.Fatal("all three promoted loads must be erased")
	}
	if store.BasePtr().Erased() || store.KindOf() != ir.KindMemStore {
		t.Fatal("the single-owner GMEM store must survive (stores never route through the cache)")
	}

	assertVerifyAndRootReachable(t, fx.g)
	if err := (typeinfer.Pass{}).Run(nil, fx.g); err != nil {
		t.Fatalf("typeinfer re-run over the R3-lowered graph: %v", err)
	}
}
