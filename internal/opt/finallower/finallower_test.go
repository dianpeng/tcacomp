package finallower_test

// finallower_test.go pins the final lowering pass (quick-261022-l1o,
// doc/optana/final_lower.md as amended): the fail-closed gates, Map Lower
// (LR-9: RAUW onto the body + full debris disposal), the runtime-object
// construction (LR-11: one KernelInfo first of the kernel Body, one
// GridMapping+ThreadMapping pair per LNT prologue), and the launch-map
// query rewrite (LR-12: explicit RuntimeGetAttribute on the owning LNT's
// mapping object; the query kinds vanish).
//
// Fixture idiom (the tiling lower_test + schedule pipeline_test patterns):
// IR-level chain fixtures (u32 extents, post-simplify shapes built
// directly), a CRAFTED decide record standing in for the verified search
// product, then the REAL tile-lower → schedule → final-lower Run calls.

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/finallower"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling"
	"tcacomp/internal/opt/typeinfer"
	"tcacomp/internal/pipeline"
)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

// fx bundles a chain fixture: u32 dim params + a HIR iter chain (the
// post-simplify shape tile-lower consumes), in allocation order.
type fx struct {
	g     *ir.Graph
	u32   ir.Type
	dims  []ir.Node
	iters []ir.Node
}

// newChain builds an n-deep iter chain with u32 const extents.
func newChain(name string, extents ...float64) *fx {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	f := &fx{g: g, u32: u32}
	for i := range extents {
		f.dims = append(f.dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	var parent ir.Node
	for i := range extents {
		dv := g.NewDimVar("ctor", f.dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extents[i]))
		parent = g.NewIter("ctor", biv, parent)
		f.iters = append(f.iters, parent)
	}
	return f
}

// tensor adds a rank-len(dims) f32 tensor param.
func (f *fx) tensor(name string, dims ...int) ir.Node {
	refs := make([]ir.DimRef, len(dims))
	for i, d := range dims {
		refs[i] = ir.DimRef{Name: fmt.Sprintf("d%d", d)}
	}
	return f.g.NewInputParam("ctor", ir.RoleTensor, name, ir.InternTensor(ir.TF32, refs))
}

// iv returns a fresh InductionVariable of iters[i].
func (f *fx) iv(i int) ir.Node {
	return f.g.NewInductionVariable("ctor", f.iters[i])
}

// bodyStore wires a store of val over every chain IV into `out`, chained
// into the return (the LNT's effectful body).
func (f *fx) bodyStore(out ir.Node, val ir.Node) {
	env := ir.NewEnv()
	axes := make([]ir.Node, len(f.iters))
	for i := range f.iters {
		axes[i] = f.iv(i)
	}
	st := f.g.NewMemStoreChained("ctor", env, out, axes, val)
	f.g.NewReturnChained("ctor", env, []ir.Node{out}, st)
}

// fakeAnalysis is a wrong-typed cache entry under the provenance name.
type fakeAnalysis struct{}

func (fakeAnalysis) Name() string { return tiling.ProvenanceName }

// runTileLowerSchedule runs the REAL tile-lower + schedule passes over g
// with a crafted verified record (one decision per chain iter, by index),
// returning the cache that now holds the published provenance. The WARP-
// reduced reducer fixtures route through runTileLowerScheduleSplitK — the
// fix4 §3.5 Wave-1 defense fails the shape closed under the default
// ReducedDimWarp=false, and the capability flag is what admits it.
func runTileLowerSchedule(t *testing.T, f *fx, decisions []tiling.IterDecision) *opt.Cache {
	t.Helper()
	return runTileLowerScheduleP(t, f, decisions, tiling.DefaultParams())
}

// runTileLowerScheduleSplitK runs the pair with ReducedDimWarp=true — the
// fix4 §3.5 capability gate that admits WARP-assigned reduced dims. The
// reducer-lowering fixtures that pin the WarpReducer face (LR-7/R1/R2/R3
// machinery) run under it: the machinery is flag-independent, and the
// default-flag policy is pinned on the tiling side
// (tiling.TestLowerWarpReducedDimRejected).
func runTileLowerScheduleSplitK(t *testing.T, f *fx, decisions []tiling.IterDecision) *opt.Cache {
	t.Helper()
	p := tiling.DefaultParams()
	p.ReducedDimWarp = true
	return runTileLowerScheduleP(t, f, decisions, p)
}

// runTileLowerScheduleP is the params-carrying core of the pair above.
func runTileLowerScheduleP(t *testing.T, f *fx, decisions []tiling.IterDecision, p tiling.Params) *opt.Cache {
	t.Helper()
	rec := &tiling.Record{Iters: make(map[ir.Node]tiling.IterDecision), Sites: map[uint64]tiling.SiteDecision{}, Verified: true}
	for i, it := range f.iters {
		biv := ir.DataDeps(it)[0]
		if biv == nil {
			t.Fatalf("iter %d carries no BIV", i)
		}
		rec.Iters[biv] = decisions[i]
	}
	cache := opt.NewCache()
	cache.Put(rec)
	if err := (tiling.LowerPass{Params: p}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, f.g); err != nil {
		t.Fatalf("tile-lower: %v", err)
	}
	if err := (schedule.Scheduler{}).Run(nil, f.g); err != nil {
		t.Fatalf("schedule: %v", err)
	}
	if cache.Get(tiling.ProvenanceName) == nil {
		t.Fatal("fixture setup: the provenance must be published and preserved past schedule")
	}
	return cache
}

// runFinalLower runs the pass over the prepared graph + cache.
func runFinalLower(t *testing.T, g *ir.Graph, cache *opt.Cache) error {
	t.Helper()
	return (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, g)
}

// runTail runs the pipeline's TAIL composition (fix2 D-e: the tail memopt +
// the AST-item-roots dce) directly over a post-final-lower graph — the
// re-pin substrate for the D-a/D-b deletions: what final-lower now leaves
// dead in the sea is the TAIL's to collect (direct invocation, the
// runMemopt/runFinalLower pattern; the manager-level Requires threading is
// pinned in internal/pipeline's own tests).
func runTail(t *testing.T, g *ir.Graph) {
	t.Helper()
	for _, p := range pipeline.TailPasses() {
		if err := p.Run(&opt.Context{Pass: p.ID()}, g); err != nil {
			t.Fatalf("tail pass %s: %v", p.ID(), err)
		}
	}
}

// liveOfKind collects the graph's live nodes of kind k in allocation order.
func liveOfKind(g *ir.Graph, k ir.Kind) []ir.Node {
	var out []ir.Node
	for _, n := range g.AllocatedNodes() {
		if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == k {
			out = append(out, n)
		}
	}
	return out
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

// ---------------------------------------------------------------------------
// contract + gates
// ---------------------------------------------------------------------------

// TestPassContract pins the registry wiring: final-lower at priority 950
// (after schedule's 900 — the registry.go anchor), Requires [schedule],
// Produces StageConstructed, Preserved nil, and visible through opt/all.
func TestPassContract(t *testing.T) {
	spec, ok := opt.Lookup("final-lower")
	if !ok {
		t.Fatal("opt.Lookup(\"final-lower\") must find the pass (import via opt/all)")
	}
	if spec.Priority != 950 {
		t.Fatalf("final-lower priority = %d, want 950 (after schedule 900)", spec.Priority)
	}
	if len(spec.Requires) != 1 || spec.Requires[0] != "schedule" {
		t.Fatalf("final-lower Requires = %v, want [schedule]", spec.Requires)
	}
	p := finallower.Pass{}
	if p.ID() != "final-lower" || p.Produces() != opt.StageConstructed || p.Preserved() != nil {
		t.Fatalf("pass contract: ID=%q Produces=%v Preserved=%v", p.ID(), p.Produces(), p.Preserved())
	}
	if s, ok := opt.Lookup("schedule"); !ok || s.Priority != 900 {
		t.Fatal("the schedule anchor must stay at 900")
	}
}

// TestGates pins the fail-closed entry surface: nil graph, no attached AST
// ("schedule must run first"), a nil context/cache, a MISSING provenance
// entry, a WRONG-TYPED entry — and the interim reducer gate (a live
// KindReduce errors naming the reducer task).
func TestGates(t *testing.T) {
	// nil graph.
	if err := (finallower.Pass{}).Run(nil, nil); err == nil || !strings.Contains(err.Error(), "nil graph") {
		t.Fatalf("nil graph must fail closed, got %v", err)
	}
	// No AST: an unscheduled graph.
	f := newChain("noast", 64)
	f.bodyStore(f.tensor("out", 0), f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: opt.NewCache()}, f.g); err == nil ||
		!strings.Contains(err.Error(), "schedule must run first") {
		t.Fatalf("an unattached AST must fail closed naming schedule, got %v", err)
	}

	// A scheduled fixture for the cache/reducer gates.
	f2 := newChain("gates", 1024)
	f2.bodyStore(f2.tensor("out", 0), f2.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0))
	_ = runTileLowerSchedule(t, f2, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 2},
	})

	// nil context.
	if err := (finallower.Pass{}).Run(nil, f2.g); err == nil || !strings.Contains(err.Error(), "no analysis cache") {
		t.Fatalf("nil context must fail closed, got %v", err)
	}
	// Missing provenance entry.
	if err := runFinalLower(t, f2.g, opt.NewCache()); err == nil ||
		!strings.Contains(err.Error(), tiling.ProvenanceName) {
		t.Fatalf("a missing provenance entry must fail closed naming it, got %v", err)
	}
	// Wrong-typed entry.
	wrong := opt.NewCache()
	wrong.Put(fakeAnalysis{})
	if err := runFinalLower(t, f2.g, wrong); err == nil ||
		!strings.Contains(err.Error(), "want *tiling.Provenance") {
		t.Fatalf("a wrong-typed provenance entry must fail closed naming the type, got %v", err)
	}

	// The @r derivation gate: a Reduce piping a value with NO producing
	// Map/As chain is a hard diagnostic (LR-2 — never a guess).
	f3 := newChain("redgate", 512)
	lhs := f3.tensor("lhs", 0)
	load := f3.g.NewMemLoad("ctor", lhs, []ir.Node{f3.iv(0)})
	acc := f3.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f3.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f3.g.NewBinOp("ctor", ir.BinOpAdd, f3.g.NewParamRef("ctor", acc), f3.g.NewParamRef("ctor", now))
	red := f3.g.NewReduce("ctor", f3.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), load,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	f3.bodyStore(f3.tensor("out", 0), red)
	cache3 := runTileLowerSchedule(t, f3, []tiling.IterDecision{
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
	})
	if err := runFinalLower(t, f3.g, cache3); err == nil ||
		!strings.Contains(err.Error(), "no producing Map/As chain") {
		t.Fatalf("a broken producing chain must fail closed naming the derivation, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// Map Lower + objects + query rewrite (the map-only end-to-end)
// ---------------------------------------------------------------------------

// mapOnlyFx builds the map-only kernel: a 2-dim chain (CTA tiled + SEQ
// tiled at the floor 2), a GMEM load over both IVs, an INLINED-shape Map
// over it with
// as=[d0,d1] (zero declared params — the post-simplify-inline shape), and a
// chained store of the map into out[d0,d1]. The CTA dim makes tile-lower
// emit launch-map queries in the rewritten index expressions.
func mapOnlyFx(t *testing.T) (*fx, ir.Node, ir.Node, ir.Node) {
	t.Helper()
	f := newChain("maponly", 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0, 1)
	load := f.g.NewMemLoad("ctor", lhs, []ir.Node{f.iv(0), f.iv(1)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), f.dims)
	f.bodyStore(out, mp)
	return f, load, mp, out
}

// runMapOnly lowers + schedules the map-only fixture and runs final-lower.
func runMapOnly(t *testing.T) (*fx, *ir.ASTKernel, ir.Node, ir.Node, ir.Node) {
	t.Helper()
	f, load, mp, out := mapOnlyFx(t)
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
	})
	if err := runFinalLower(t, f.g, cache); err != nil {
		t.Fatalf("final-lower over the map-only kernel: %v", err)
	}
	return f, f.g.AST(), load, mp, out
}

// TestMapLowerEliminatesMap pins LR-9's VERB (fix2 D-b): the Map's every
// use is RAUW'd onto its body expression (the store now reads the load),
// the Map node is erased, and the trailing As is erased. The orphaned
// lambda is NO LONGER this pass's to sweep: it is an AST ITEM, so it
// survives as item-anchored debris (the tail dce's item-roots policy keeps
// it — D-d's post-schedule liveness truth — and the emitter skips lambda
// closures); the pin asserts it is dead from the SEA's perspective (zero
// SSA uses) instead.
func TestMapLowerEliminatesMap(t *testing.T) {
	f, _, load, mp, out := runMapOnly(t)
	if !mp.BasePtr().Erased() {
		t.Fatal("the Map node must be erased")
	}
	// The store's value operand resolves at the Map's body (the load).
	var st ir.Node
	for _, n := range liveOfKind(f.g, ir.KindMemStore) {
		st = n
	}
	if st == nil {
		t.Fatal("the GMEM store must survive")
	}
	if ops := ir.OperandsOf(st); len(ops) < 1 || ops[0] == nil || ops[0].BasePtr() != load.BasePtr() {
		t.Fatalf("the store's value must be RAUW'd onto the map's body (the load), got %v", ops[0])
	}
	_ = out
	// The verb's own erasures: no live Map / As survives.
	for _, k := range []ir.Kind{ir.KindMap, ir.KindAs} {
		if got := liveOfKind(f.g, k); len(got) != 0 {
			t.Fatalf("no live %s may survive Map Lower, got %d", k, len(got))
		}
	}
	// The lambda debris: dead in the sea (zero SSA uses — the Map was its
	// only user), alive only as an AST item (fix2 D-b/D-d).
	for _, lam := range liveOfKind(f.g, ir.KindLambda) {
		if uses := ir.UsesOf(lam); len(uses) != 0 {
			t.Fatalf("the orphaned lambda node %d still carries %d SSA use(s) — the verb must leave it dead in the sea",
				lam.BasePtr().ID(), len(uses))
		}
	}
	if vs := ir.Verify(f.g); len(vs) != 0 {
		t.Fatalf("the lowered graph must verify clean, got %v", vs)
	}
}

// TestRuntimeObjectsInserted pins LR-11: exactly one RuntimeKernelInfo as
// item 0 of the kernel Body Group (SN 0, Prototype untouched — its items
// are exactly the kernel params), and every LNT's prologue Group starts
// with [RuntimeGridMapping, RuntimeThreadMapping] built from the
// provenance plan's ABSOLUTE factorization shapes (fix4 §3.2; this
// fixture's crafted record carries one CTA dim and no WARP dim: grid
// [100], thread []), both stamped SN 0/prologue.
func TestRuntimeObjectsInserted(t *testing.T) {
	f, k, _, _, _ := runMapOnly(t)

	// KernelInfo: exactly one, first item of the Body Group.
	kis := liveOfKind(f.g, ir.KindRuntimeKernelInfo)
	if len(kis) != 1 {
		t.Fatalf("exactly one RuntimeKernelInfo must exist, got %d", len(kis))
	}
	ki := kis[0]
	items := k.Body.Items()
	if len(items) == 0 || items[0].Node == nil || items[0].Node.BasePtr() != ki.BasePtr() {
		t.Fatal("the KernelInfo must be the FIRST item of the kernel Body Group")
	}
	if ki.BasePtr().SchedSN() != 0 || ki.BasePtr().SchedBlock() != k.Body {
		t.Fatal("the KernelInfo must be stamped SN 0 / kernel Body Group")
	}
	// The Prototype is signature scope: its items are exactly the kernel
	// params (no runtime objects).
	for _, it := range k.Prototype.Items() {
		if it.Node == nil || it.Node.KindOf() != ir.KindInputParam {
			t.Fatalf("prototype item kind = %v, want an InputParam placeholder (the Prototype is untouched)",
				kindOf(it.Node))
		}
	}

	// Per-LNT mappings: the prologue Group's first two items, grid before
	// thread, portions per the provenance plan.
	if len(k.LNTs) == 0 {
		t.Fatal("the fixture must schedule at least one LNT")
	}
	for _, lnt := range k.LNTs {
		var prologue *ir.ASTBlock
		for _, it := range k.Body.Items() {
			if it.Block != nil && it.Block.Kind() == ir.BlockGroup && it.Block.Nest() == lnt {
				prologue = it.Block
				break
			}
		}
		if prologue == nil {
			t.Fatalf("LNT root %d has no prologue Group", lnt.Root.BasePtr().ID())
		}
		pro := prologue.Items()
		if len(pro) < 2 || pro[0].Node == nil || pro[1].Node == nil ||
			pro[0].Node.KindOf() != ir.KindRuntimeGridMapping ||
			pro[1].Node.KindOf() != ir.KindRuntimeThreadMapping {
			t.Fatalf("the prologue must start with [GridMapping, ThreadMapping], got %v", pro)
		}
		grid, thread := pro[0].Node, pro[1].Node
		if grid.BasePtr().SchedSN() != 0 || grid.BasePtr().SchedBlock() != prologue ||
			thread.BasePtr().SchedSN() != 0 || thread.BasePtr().SchedBlock() != prologue {
			t.Fatal("both mappings must be stamped SN 0 / prologue")
		}
		// Operands per P-3 (fix4 §3.2): [kernel_info, ndim, shapes...] —
		// the crafted record carries one CTA dim (LoC 100) and no WARP
		// dim, so grid [100] and thread [].
		checkMappingOperands(t, grid, ki, []int{100}, "grid")
		checkMappingOperands(t, thread, ki, []int{}, "thread")
		// Both mappings' operand 0 is THE KernelInfo (explicit SSA).
		if op := ir.OperandsOf(grid)[0]; op.BasePtr() != ki.BasePtr() {
			t.Fatal("the grid mapping must consume the KernelInfo")
		}
	}
}

// checkMappingOperands pins the P-3 encoding: [kernel_info, TI32 ndim,
// TI32 portions...] with the given portion values.
func checkMappingOperands(t *testing.T, m, ki ir.Node, want []int, what string) {
	t.Helper()
	ops := ir.OperandsOf(m)
	if len(ops) != len(want)+2 {
		t.Fatalf("%s mapping carries %d operands, want %d", what, len(ops), len(want)+2)
	}
	if ops[0].BasePtr() != ki.BasePtr() {
		t.Fatalf("%s mapping operand 0 must be the KernelInfo", what)
	}
	if v, ok := ir.ConstValueOf(ops[1]); !ok || int(v) != len(want) {
		t.Fatalf("%s mapping ndim = %v, want %d", what, v, len(want))
	}
	for i, w := range want {
		v, ok := ir.ConstValueOf(ops[2+i])
		if !ok || int(v) != w {
			t.Fatalf("%s mapping portion %d = %v, want %d", what, i, v, w)
		}
	}
}

// TestQueryRewrite pins LR-12: every launch-map query node is erased, a
// RuntimeGetAttribute carrying the same field payload, the resolved LNT
// mapping object as operand 0, and the query's DoSoC const as its single
// arg replaces it (all former query uses RAUW'd); the query kinds vanish
// from the graph.
func TestQueryRewrite(t *testing.T) {
	f, _, _, _, _ := runMapOnly(t)

	// No live query nodes remain.
	for _, k := range []ir.Kind{ir.KindRuntimeGridQuery, ir.KindRuntimeThreadQuery} {
		if got := liveOfKind(f.g, k); len(got) != 0 {
			t.Fatalf("no live %s may survive the rewrite, got %d", k, len(got))
		}
	}
	// Every live getattr is well-formed: attribute ∈ {index, dimension}
	// (the queries' vocabulary), operand 0 is a mapping construction, and the
	// args follow the flat-index() ruling (2026-10-23): "index" may carry NO
	// extra arg (the zero-arg flat form) or the DoSoC const (the tuple
	// coordinate form); "dimension" keeps exactly one const-u32 arg.
	getattrs := liveOfKind(f.g, ir.KindRuntimeGetAttribute)
	if len(getattrs) == 0 {
		t.Fatal("the CTA dim's index expressions must have produced queries, so getattrs must exist")
	}
	for _, ga := range getattrs {
		attr := ir.RuntimeAttributeOf(ga)
		if attr != "index" && attr != "dimension" {
			t.Fatalf("getattr attribute = %q, want index|dimension (the query vocabulary)", attr)
		}
		ops := ir.OperandsOf(ga)
		if len(ops) < 1 || len(ops) > 2 || ops[0] == nil {
			t.Fatalf("index/dimension getattr must carry [mapping] or [mapping, DoSoC], got %v", ops)
		}
		if k := ops[0].KindOf(); k != ir.KindRuntimeGridMapping && k != ir.KindRuntimeThreadMapping {
			t.Fatalf("getattr operand 0 kind = %s, want a mapping construction", k)
		}
		if len(ops) == 2 {
			if v, ok := ir.ConstValueOf(ops[1]); !ok || v < 0 {
				t.Fatalf("getattr DoSoC arg must be a const, got %v", ops[1])
			}
		} else if attr != "index" {
			t.Fatalf("attribute %q must carry its DoSoC arg (only the flat index() form goes argless)", attr)
		}
	}
	// The mapping objects are USED (the binding is explicit SSA).
	used := 0
	for _, m := range append(liveOfKind(f.g, ir.KindRuntimeGridMapping), liveOfKind(f.g, ir.KindRuntimeThreadMapping)...) {
		if len(ir.UsesOf(m)) > 0 {
			used++
		}
	}
	if used == 0 {
		t.Fatal("at least one mapping object must be consumed by the rewritten getattrs")
	}
}

// TestFlatQueryRewrite pins the flat-index() ruling's final-lower half: a
// ZERO-OPERAND thread query (grafted as an LNT-prologue item over the
// scheduled map-only kernel) rewrites to a RuntimeGetAttribute with attribute
// "index" over the owning LNT's ThreadMapping carrying NO extra args, and
// every query kind vanishes from the graph.
func TestFlatQueryRewrite(t *testing.T) {
	f, load, mp, out := mapOnlyFx(t)
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
	})
	k := f.g.AST()
	var prologue *ir.ASTBlock
	for _, it := range k.Body.Items() {
		if it.Block != nil && it.Block.Kind() == ir.BlockGroup && it.Block.Nest() == k.LNTs[0] {
			prologue = it.Block
			break
		}
	}
	if prologue == nil {
		t.Fatal("fixture setup: the LNT prologue Group must exist post-schedule")
	}
	flat := f.g.NewRuntimeThreadQueryFlat("ctor")
	flat.BasePtr().SetSched(0, prologue)
	prologue.AppendItem(ir.ASTItem{Node: flat})

	if err := runFinalLower(t, f.g, cache); err != nil {
		t.Fatalf("final-lower over the grafted flat query: %v", err)
	}
	for _, kind := range []ir.Kind{ir.KindRuntimeGridQuery, ir.KindRuntimeThreadQuery} {
		if got := liveOfKind(f.g, kind); len(got) != 0 {
			t.Fatalf("no live %s may survive the rewrite (flat included), got %d", kind, len(got))
		}
	}
	found := false
	for _, ga := range liveOfKind(f.g, ir.KindRuntimeGetAttribute) {
		ops := ir.OperandsOf(ga)
		if ir.RuntimeAttributeOf(ga) != "index" || len(ops) != 1 || ops[0] == nil {
			continue
		}
		if ops[0].KindOf() == ir.KindRuntimeThreadMapping {
			found = true // the zero-arg flat form
		}
	}
	if !found {
		t.Fatal("the flat query must have rewritten to a zero-arg index getattr over the ThreadMapping")
	}
	_ = load
	_ = mp
	_ = out
}

// TestMapOnlyEndToEnd pins the map-only kernel's full post-pass shape:
// zero Map/lambda-debris/query nodes, exactly one KernelInfo as the first
// statement, one mapping pair per LNT prologue, every former query use
// resolved through getattrs, no AST item referencing an erased node, verify
// green, and determinism (twin runs produce identical AST item orders).
func TestMapOnlyEndToEnd(t *testing.T) {
	f, k, _, _, _ := runMapOnly(t)

	// No query / map kinds anywhere. (The producing Map's LAMBDA is
	// deliberately absent from this list since fix2 D-b: the lower no
	// longer sweeps it, and it is an AST item — item-anchored debris the
	// tail's item-roots policy keeps and the emitter skips.)
	for _, kind := range []ir.Kind{ir.KindMap, ir.KindAs,
		ir.KindRuntimeGridQuery, ir.KindRuntimeThreadQuery} {
		if got := liveOfKind(f.g, kind); len(got) != 0 {
			t.Fatalf("live %s nodes survive, got %d", kind, len(got))
		}
	}
	// No AST item references an erased node; every walked node's
	// SchedBlock is its containing block.
	walkItems(k.Body, func(n ir.Node, containing *ir.ASTBlock) {
		if n.BasePtr().Erased() {
			t.Fatalf("AST item references erased node %d (%s)", n.BasePtr().ID(), n.KindOf())
		}
		if n.BasePtr().SchedBlock() != containing {
			t.Fatalf("node %d (%s): SchedBlock is not its containing block", n.BasePtr().ID(), n.KindOf())
		}
	})
	// The first statement of the kernel Body is the KernelInfo; the first
	// items of the single LNT's prologue are the mapping pair.
	if items := k.Body.Items(); items[0].Node == nil || items[0].Node.KindOf() != ir.KindRuntimeKernelInfo {
		t.Fatalf("kernel Body item 0 = %v, want the KernelInfo", items[0].Node)
	}

	// Determinism: a twin fixture lowers to the identical AST item-kind
	// sequence.
	f2, _, _, _, _ := runMapOnly(t)
	dump, dump2 := astShape(k), astShape(f2.g.AST())
	if dump != dump2 {
		t.Fatalf("twin runs differ:\n--- 1 ---\n%s\n--- 2 ---\n%s", dump, dump2)
	}
}

// astShape renders the AST's per-block item-kind sequence (deterministic).
func astShape(k *ir.ASTKernel) string {
	var b strings.Builder
	var dump func(blk *ir.ASTBlock)
	dump = func(blk *ir.ASTBlock) {
		fmt.Fprintf(&b, "[%s:", blk.Kind())
		for _, it := range blk.Items() {
			if it.Node != nil {
				fmt.Fprintf(&b, "%s", it.Node.KindOf())
				if a := ir.RuntimeAttributeOf(it.Node); a != "" {
					fmt.Fprintf(&b, ":%s", a)
				}
				if m := ir.RuntimeMethodOf(it.Node); m != "" {
					fmt.Fprintf(&b, ":%s", m)
				}
			} else if it.Block != nil {
				dump(it.Block)
			}
		}
		b.WriteString("]")
	}
	dump(k.Body)
	return b.String()
}

// kindOf renders a node's kind nil-safely.
func kindOf(n ir.Node) ir.Kind {
	if n == nil {
		return ir.KindInvalid
	}
	return n.KindOf()
}

// ---------------------------------------------------------------------------
// Reducer lower (Task 3): bracket law, specialization, slot IVs, RAUW,
// re-iteration, integration
// ---------------------------------------------------------------------------

// redFx is the tiled-reduce fixture: dims m (CTA T=32) and k (WARP T=64),
// lhs[m,k] loaded over both IVs (chained), piped through an INLINED-shape
// Map as=[m,k], reduced by a two-formal combiner (the exemption shape), and
// stored into out[m] (m-only axes — the reduction-shape tail the scheduler
// lands in the nest epilogue). typeinfer runs for the homogeneity types.
type redFx struct {
	f      *fx
	load   ir.Node
	mp     ir.Node
	red    ir.Node
	init   ir.Node
	lambda ir.Node
	store  ir.Node
}

// newRedFx builds the fixture; variant knobs: extraDim (append a foreign
// dim to the Map's as — the no-provenance probe) and poison (a return value
// referencing the k IV — the reduced-dimvar probe).
func newRedFx(t *testing.T, extraDim, poison bool) *redFx {
	t.Helper()
	f := newChain("red", 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1)})
	as := append([]ir.Node(nil), f.dims...)
	if extraDim {
		as = append(as, f.g.NewInputParam("ctor", ir.RoleDim, "dx", nil))
	}
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), as)
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	init := f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0)
	red := f.g.NewReduce("ctor", init, mp, ir.NewLambda([]ir.Node{acc, now}, comb))
	lambda := ir.LambdaOf(red)
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0)}, red)
	values := []ir.Node{store}
	if poison {
		// A type-legal epilogue consumer referencing BOTH the Reduce (→
		// Combine after the RAUW) and the reduced k dimvar: cast<kIV> + red.
		// tile-lower RAUWs the HIR k-IV use onto k's lowered index
		// expression, so the moved consumer's closure reaches the reduced
		// loop dimvars.
		kf32 := f.g.NewCast("ctor", ir.InternScalar(ir.TF32), f.iv(1))
		values = append(values, f.g.NewBinOp("ctor", ir.BinOpAdd, kf32, red))
	}
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, values...)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	return &redFx{f: f, load: load, mp: mp, red: red, init: init, lambda: lambda, store: store}
}

// runRedFx drives the REAL tile-lower + schedule over the fixture with the
// pinned decisions (m CTA T=32, k WARP T=64 — the tiled matmul-like shape
// with m strictly between X and home), then final-lower, failing on error.
func runRedFx(t *testing.T, fx *redFx) *opt.Cache {
	t.Helper()
	cache := runTileLowerScheduleSplitK(t, fx.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	if err := runFinalLower(t, fx.f.g, cache); err != nil {
		t.Fatalf("final-lower over the tiled reduce: %v", err)
	}
	return cache
}

// provenanceOf resolves the fixture's per-dim provenance entries.
func provenanceOf(t *testing.T, cache *opt.Cache) *tiling.Provenance {
	t.Helper()
	prov, ok := cache.Get(tiling.ProvenanceName).(*tiling.Provenance)
	if !ok || prov == nil {
		t.Fatal("fixture setup: no provenance")
	}
	return prov
}

// TestReducerBracketPlacement pins the R3 bracket law on the tiled path:
// the construction is a RuntimeWarpReducer (X's Soc 2, LR-7 + R1/R2)
// carrying (%init, %lambda, ThreadMapping, staging MemNewSMem — one
// #warps × sizeof(T) = 64 B row at 512 threads/f32), inserted with the
// staging row at the owning prologue's head end, both stamped SN 0; the
// between (m tile-inside) loop DISSOLVES and a fresh UNIFORM sequential
// ELEMENT loop takes X's position in X's parental block, hosting per
// element [Init (item 0) → X's block → Combine → the moved store]; the
// Reduce call sits inside home (nested in X) right after the piped
// producer's item with the VALUE as its single argument (R1 — the slot
// dimvars are gone).
func TestReducerBracketPlacement(t *testing.T) {
	rx := newRedFx(t, false, false)
	mDim, kDim := rx.f.dims[0], rx.f.dims[1]
	cache := runTileLowerScheduleSplitK(t, rx.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	prov := provenanceOf(t, cache)
	kprov := prov.Dims[kDim]
	X, home := kprov.Between, kprov.Inside
	if X == nil || home == nil {
		t.Fatal("fixture setup: k must be tiled (Between+Inside)")
	}
	mInside := prov.Dims[mDim].Inside
	if mInside == nil {
		t.Fatal("fixture setup: m must be tiled (the between dim)")
	}
	// Capture the dissolved loop's facts BEFORE final-lower (its operand
	// slots read unwired once erased).
	mInsideIV := ir.OperandsOf(mInside)[0]
	mDimParam := ir.OperandsOf(mInsideIV)[0]
	mMax := ir.OperandsOf(ir.OperandsOf(mInside)[1])[1]

	if err := runFinalLower(t, rx.f.g, cache); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	g, k := rx.f.g, rx.f.g.AST()

	// The construction: exactly one WarpReducer with the R1 4-operand face
	// (init, lambda, ThreadMapping, staging), both it and the staging row
	// in the owning LNT's prologue at SN 0. The staging row is 64 B — 16
	// warps (512/32) × 4 B of f32.
	reds := liveOfKind(g, ir.KindRuntimeWarpReducer)
	if len(reds) != 1 || len(liveOfKind(g, ir.KindRuntimeSequentialReducer)) != 0 ||
		len(liveOfKind(g, ir.KindRuntimeGlobalReducer)) != 0 {
		t.Fatalf("exactly one WARP reducer construction expected (Soc 2), got %d warp", len(reds))
	}
	construction := reds[0]
	ops := ir.OperandsOf(construction)
	if len(ops) != 4 || ops[0].BasePtr() != rx.init.BasePtr() || ops[1].BasePtr() != rx.lambda.BasePtr() {
		t.Fatalf("the construction must carry (init, lambda, ThreadMapping, staging) — construction carries ALL state (LR-3 + R1/R2), got %d operands", len(ops))
	}
	if ops[2].KindOf() != ir.KindRuntimeThreadMapping {
		t.Fatalf("operand 2 must be the owning LNT's ThreadMapping (R2), got %s", ops[2].KindOf())
	}
	if sc, _ := ir.StorageClassOf(ops[3]); ops[3].KindOf() != ir.KindMemNew || sc != ir.StorageSMem {
		t.Fatalf("operand 3 must be the staging SMEM-class MemNew (R1), got %s", ops[3].KindOf())
	}
	staging := ops[3]
	if v, ok := ir.ConstValueOf(ir.OperandsOf(staging)[0]); !ok || v != 32 {
		t.Fatalf("the staging row must be 32 B (8 warps x f32 at 256 threads — K=8 WARP dim → blockDim=256), got %v", v)
	}
	if got := len(liveOfKind(g, ir.KindRuntimeThreadMapping)); got != 1 {
		t.Fatalf("exactly one ThreadMapping (the staging/mapping operand), got %d", got)
	}
	if ops[2].BasePtr() != liveOfKind(g, ir.KindRuntimeThreadMapping)[0].BasePtr() {
		t.Fatal("the construction's mapping operand must be the owning LNT's ThreadMapping")
	}
	if staging.BasePtr().SchedSN() != 0 || construction.BasePtr().SchedSN() != 0 {
		t.Fatal("the staging row and the construction must be stamped SN 0")
	}

	// The element loop: X's block moved INTO a fresh uniform SeqLoop that
	// took X's position in X's parental block. Locate the element block as
	// Init's containing block.
	initCall, combineCall, reduceCall := callOf(g, "Init"), callOf(g, "Combine"), callOf(g, "Reduce")
	if initCall == nil || combineCall == nil || reduceCall == nil {
		t.Fatal("exactly one Init, one Reduce, and one Combine call must exist")
	}
	elemBlk, initIdx := findItemIn(k, initCall)
	if elemBlk == nil || initIdx < 0 {
		t.Fatalf("Init must be an item of the element loop block, got index %d", initIdx)
	}
	// fix3 D-j: items RE-PLACED from the dissolved between blocks may precede
	// Init (in order, at the block head); nothing else may.
	if elemBlk.Kind() != ir.BlockSeqLoop {
		t.Fatalf("the element bracket host must be a SeqLoop (uniform), got %s", elemBlk.Kind())
	}
	items := elemBlk.Items()
	if len(items) < initIdx+3 || items[initIdx+1].Block == nil || items[initIdx+1].Block.Loop().BasePtr() != X.BasePtr() {
		t.Fatal("X's block must be the item right after Init inside the element loop")
	}
	if items[initIdx+2].Node == nil || items[initIdx+2].Node.BasePtr() != combineCall.BasePtr() {
		t.Fatal("Combine must sit immediately after X's block inside the element loop")
	}
	// The element loop: same dim and same bound expression as the DISSOLVED
	// m tile-inside loop; uniform init 0 / step 1; a FRESH dimvar.
	elemLoop := elemBlk.Loop()
	elemIV := ir.OperandsOf(elemLoop)[0]
	if mInsideIV.BasePtr() == elemIV.BasePtr() {
		t.Fatal("the element loop must mint a FRESH dimvar, never reuse the dissolved between IV")
	}
	if d := ir.OperandsOf(elemIV)[0]; d.BasePtr() != mDimParam.BasePtr() {
		t.Fatal("the element loop must walk the SAME dim (m) as the dissolved between loop")
	}
	elemMax := ir.OperandsOf(ir.OperandsOf(elemLoop)[1])[1]
	if mMax.BasePtr() != elemMax.BasePtr() {
		t.Fatal("the element loop must carry the SAME clamped bound expression node")
	}
	if v, ok := ir.ConstValueOf(ir.OperandsOf(elemIV)[1]); !ok || v != 0 {
		t.Fatal("the element loop's init must be the const 0 (uniform walk)")
	}
	if v, ok := ir.ConstValueOf(ir.OperandsOf(elemIV)[2]); !ok || v != 1 {
		t.Fatal("the element loop's step must be the const 1 (uniform walk)")
	}
	// The dissolved between loop is ERASED (its per-thread walk is gone).
	if !mInside.BasePtr().Erased() {
		t.Fatal("the dissolved between loop must be erased")
	}
	// The dissolved between dimvar: final-lower only erases it when nothing
	// still uses it (fix2 D-a — verb-death only); the orphaned ORIGINAL
	// coordinate spine keeps sea uses on it, so it is the TAIL dce's
	// garbage. Run the tail and pin the handoff: collected there.
	runTail(t, g)
	if !mInsideIV.BasePtr().Erased() {
		t.Fatal("the dissolved between dimvar must be collected by the tail dce (final-lower no longer sweeps it, fix2 D-a)")
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("the tailed graph must verify clean, got %v", vs)
	}

	// The Reduce call: inside home's block (home nested inside X, X inside
	// the element loop), right after the piped producer's (load's) item,
	// with the VALUE as its single argument.
	homeBlk := home.BasePtr().SchedBlock()
	homeItems := homeBlk.Items()
	callIdx := -1
	for i, it := range homeItems {
		if it.Node != nil && it.Node.BasePtr() == reduceCall.BasePtr() {
			callIdx = i
		}
	}
	if callIdx < 0 {
		t.Fatal("the Reduce call must be an item of the home block")
	}
	if callIdx == 0 || homeItems[callIdx-1].Node == nil ||
		homeItems[callIdx-1].Node.BasePtr() != rx.load.BasePtr() {
		t.Fatal("the Reduce call must sit immediately after the piped producer's (load's) item")
	}
	callOps := ir.OperandsOf(reduceCall)
	if len(callOps) != 2 || callOps[0].BasePtr() != construction.BasePtr() || callOps[1].BasePtr() != rx.load.BasePtr() {
		t.Fatalf("the Reduce call must carry (reducer, value) — R1's scalar protocol, the piped operand resolving at the load after Map Lower — got %d operands", len(callOps))
	}
	// Home is nested inside X, X inside the element loop, the element loop
	// inside X's ORIGINAL parental block (the CTA rounds block).
	if !blockNested(homeBlk, X.BasePtr().SchedBlock()) {
		t.Fatal("home must stay nested inside X")
	}
	if !blockNested(X.BasePtr().SchedBlock(), elemBlk) {
		t.Fatal("X must be nested inside the element loop")
	}

	// R-A mechanical pins (the 2026-10-24 bracket law, 1-between audit):
	// Init's containing block IS X's parental block (the element loop), Init
	// is the item IMMEDIATELY before X's block, Combine is the item
	// IMMEDIATELY after it in that same block, and Reduce's innermost
	// containing block is Y (home).
	xParent, xIdxRA := findParentOfBlock(k, X.BasePtr().SchedBlock())
	if xParent == nil || xParent != elemBlk {
		t.Fatalf("X's parental block must be the element loop hosting Init (R-A: the innermost output-element loop IS X's parent), got %v", xParent)
	}
	if xIdxRA != initIdx+1 {
		t.Fatalf("Init must sit IMMEDIATELY before X's block (R-A): Init at %d, X at %d", initIdx, xIdxRA)
	}
	afterX := elemBlk.Items()[xIdxRA+1]
	if afterX.Node == nil || afterX.Node.BasePtr() != combineCall.BasePtr() {
		t.Fatalf("Combine must sit IMMEDIATELY after X's block (R-A), got %v", afterX.Node)
	}
	if rBlk, _ := findItemIn(k, reduceCall); rBlk == nil || rBlk != homeBlk {
		t.Fatalf("Reduce's innermost containing block must be Y (home), got %v", rBlk)
	}
}

// blockNested reports whether inner is outer or nested inside it.
func blockNested(inner, outer *ir.ASTBlock) bool {
	if inner == outer {
		return true
	}
	for _, it := range outer.Items() {
		if it.Block != nil && blockNested(inner, it.Block) {
			return true
		}
	}
	return false
}

// TestReducerCombineRAUW pins 2.4: EVERY former use of the Reduce points at
// the Combine call (the store's value operand), the Reduce node and its
// KeyedAttribute are erased, and the combiner lambda stays LIVE through the
// construction's operand.
func TestReducerCombineRAUW(t *testing.T) {
	rx := newRedFx(t, false, false)
	runRedFx(t, rx)
	g := rx.f.g

	if !rx.red.BasePtr().Erased() {
		t.Fatal("the Reduce node must be erased")
	}
	combineCall := callOf(g, "Combine")
	if combineCall == nil {
		t.Fatal("the Combine call must exist")
	}
	if ops := ir.OperandsOf(rx.store); ops[0] == nil || ops[0].BasePtr() != combineCall.BasePtr() {
		t.Fatalf("the store's value must resolve at the Combine call, got %v", ops[0])
	}
	for _, u := range ir.UsesOf(combineCall) {
		_ = u // the Combine value is consumed (the store); uses exist
	}
	if len(ir.UsesOf(combineCall)) == 0 {
		t.Fatal("the Combine call must carry its moved consumers' uses")
	}
	if !rx.lambda.BasePtr().Erased() && len(ir.UsesOf(rx.lambda)) == 0 {
		t.Fatal("the combiner lambda must stay live through the construction's operand")
	}
	if rx.lambda.BasePtr().Erased() {
		t.Fatal("the combiner lambda must NEVER be erased (the runtime owns the application)")
	}
}

// TestReducerElementLoopConsumers pins R3's consumer half: the epilogue
// out-store MOVES into the fresh element loop AFTER Combine (the
// single-owner gated store — the emitter wraps it in
// ThreadMapping::is_first_in_cta()), leaving the epilogue with only the
// end node; the store's m coordinate references the FRESH element dimvar
// (rebuilt spine — the dissolved between IV is erased); and no moved
// consumer references a reduced-dim dimvar.
func TestReducerElementLoopConsumers(t *testing.T) {
	rx := newRedFx(t, false, false)
	cache := runTileLowerScheduleSplitK(t, rx.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	prov := provenanceOf(t, cache)
	mInsideIV := ir.OperandsOf(prov.Dims[rx.f.dims[0]].Inside)[0]
	if err := runFinalLower(t, rx.f.g, cache); err != nil {
		t.Fatalf("final-lower: %v", err)
	}
	g, k := rx.f.g, rx.f.g.AST()

	X, home := prov.Dims[rx.f.dims[1]].Between, prov.Dims[rx.f.dims[1]].Inside
	initCall, combineCall := callOf(g, "Init"), callOf(g, "Combine")
	if initCall == nil || combineCall == nil {
		t.Fatal("fixture setup: the bracket calls must exist")
	}
	elemBlk, _ := findItemIn(k, initCall)
	if elemBlk == nil || elemBlk.Kind() != ir.BlockSeqLoop {
		t.Fatal("fixture setup: the element loop block must host Init")
	}
	elemIV := ir.OperandsOf(elemBlk.Loop())[0]

	// Combine sits after X's block; the store sits after Combine, inside
	// the element loop; the end node did NOT move.
	items := elemBlk.Items()
	combineIdx := -1
	for i, it := range items {
		if it.Node != nil && it.Node.BasePtr() == combineCall.BasePtr() {
			combineIdx = i
		}
	}
	if combineIdx < 0 {
		t.Fatal("fixture setup: Combine must sit inside the element loop")
	}
	found := false
	for _, it := range items[combineIdx+1:] {
		if it.Node != nil && it.Node.BasePtr() == rx.store.BasePtr() {
			found = true
		}
	}
	if !found {
		t.Fatal("the epilogue out-store must MOVE into the element loop after Combine")
	}
	for _, lnt := range k.LNTs {
		for _, it := range k.Body.Items() {
			if it.Block != nil && it.Block.Kind() == ir.BlockGroup && it.Block.Nest() == lnt {
				for _, e := range it.Block.Items() {
					if e.Node != nil && e.Node.BasePtr() == rx.store.BasePtr() {
						t.Fatal("the store must no longer sit in the epilogue")
					}
				}
			}
		}
	}

	// The store's m coordinate references the FRESH element dimvar (the
	// rebuilt spine), never the dissolved between IV (erased).
	axis := ir.OperandsOf(rx.store)[2]
	cl := subtreeOf(axis)
	if !cl[elemIV] {
		t.Fatal("the store's m coordinate must reference the fresh element dimvar")
	}
	// The dissolved between dimvar is the tail's garbage now (fix2 D-a):
	// run the tail and pin it collected.
	runTail(t, g)
	if !mInsideIV.BasePtr().Erased() {
		t.Fatal("the dissolved between dimvar must be collected by the tail dce (R3 + fix2 D-a/D-e)")
	}
	// No moved consumer references a reduced-dim dimvar.
	for _, loop := range []ir.Node{X, home} {
		iv := ir.OperandsOf(loop)[0]
		if cl[iv] {
			t.Fatal("the moved store must not reference a reduced-dim dimvar")
		}
	}
	_ = g
}

// TestReducerNegativeProbes pins the fail-closed family: a reduced dim with
// no provenance entry, a piped type/init mismatch (doc/tca.md §4.2), and a
// consumer referencing a reduced-dim dimvar.
func TestReducerNegativeProbes(t *testing.T) {
	// (a) No provenance entry: the Map's as carries a foreign third dim.
	rx := newRedFx(t, true, false)
	cache := runTileLowerScheduleSplitK(t, rx.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	err := runFinalLower(t, rx.f.g, cache)
	if err == nil || !strings.Contains(err.Error(), "no tiling provenance entry") {
		t.Fatalf("a foreign reduced dim must fail closed naming the provenance, got %v", err)
	}

	// (b) Type mismatch: the init's type cell hand-corrupted to f16 against
	// the f32 piped element (the check probes FINAL-LOWER's own gate —
	// typeinfer itself would refuse a genuinely mismatched graph earlier).
	rx2 := newRedFx(t, false, false)
	mctx := &ir.MutContext{Pass: "test"}
	if err := mctx.SetType(rx2.init, ir.InternScalar(ir.TF16)); err != nil {
		t.Fatalf("fixture setup: %v", err)
	}
	cache2 := runTileLowerScheduleSplitK(t, rx2.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	err = runFinalLower(t, rx2.f.g, cache2)
	if err == nil || !strings.Contains(err.Error(), "homogeneous reducer contract") {
		t.Fatalf("an f16/f32 piped-init mismatch must fail closed naming the contract, got %v", err)
	}

	// (c) A moved consumer referencing a reduced-dim dimvar.
	rx3 := newRedFx(t, false, true)
	cache3 := runTileLowerScheduleSplitK(t, rx3.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	err = runFinalLower(t, rx3.f.g, cache3)
	if err == nil || !strings.Contains(err.Error(), "reduced-dim dimvar") {
		t.Fatalf("a consumer referencing a reduced-dim dimvar must fail closed, got %v", err)
	}
}

// TestQueryOutsideLNT pins P-11's hard diagnostic: a query whose SchedBlock
// resolves to NO LNT (here: hand-stamped into the kernel Body Group after
// scheduling) cannot be bound.
func TestQueryOutsideLNT(t *testing.T) {
	f, _, _, _ := mapOnlyFx(t)
	cache := runTileLowerSchedule(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
	})
	stray := f.g.NewRuntimeGridQuery("test", "index", f.g.NewConst("test", ir.InternScalar(ir.TU32), 0))
	stray.BasePtr().SetSched(0, f.g.AST().Body)
	err := runFinalLower(t, f.g, cache)
	if err == nil || !strings.Contains(err.Error(), "outside every LNT") {
		t.Fatalf("a query outside every LNT must fail closed, got %v", err)
	}
}

// TestTiledReduceEndToEnd pins the full post-pass shape over the tiled
// reduce kernel: zero Map/Reduce/query nodes (the producing Map AND its
// Reduce both lower in ONE run — the LR-2 order pin), exactly one
// KernelInfo first statement, one mapping pair in the prologue, the
// complete reducer statement set in the right blocks, ir.Verify green, and
// no AST item referencing an erased node.
func TestTiledReduceEndToEnd(t *testing.T) {
	rx := newRedFx(t, false, false)
	runRedFx(t, rx)
	g, k := rx.f.g, rx.f.g.AST()

	for _, kind := range []ir.Kind{ir.KindMap, ir.KindReduce,
		ir.KindRuntimeGridQuery, ir.KindRuntimeThreadQuery} {
		if got := liveOfKind(g, kind); len(got) != 0 {
			t.Fatalf("live %s nodes survive, got %d", kind, len(got))
		}
	}
	// The combiner lambda survives as the construction's operand (the
	// runtime owns the application). The producing Map's lambda is NO
	// LONGER swept (fix2 D-b): it is an AST item, so it may survive as
	// item-anchored debris (D-d's liveness truth; the emitter skips lambda
	// closures) — dead in the sea, never an operand of anything live.
	construction := liveOfKind(g, ir.KindRuntimeWarpReducer)
	if len(construction) != 1 {
		t.Fatalf("exactly one reducer construction expected, got %d", len(construction))
	}
	lamOps := ir.OperandsOf(construction[0])
	if len(lamOps) < 2 || lamOps[1].BasePtr() != rx.lambda.BasePtr() {
		t.Fatal("the combiner lambda must be the construction's operand and stay live")
	}
	for _, lam := range liveOfKind(g, ir.KindLambda) {
		if lam.BasePtr() == rx.lambda.BasePtr() {
			continue
		}
		if uses := ir.UsesOf(lam); len(uses) != 0 {
			t.Fatalf("a non-combiner lambda node %d still carries %d SSA use(s) — Map Lower's verb must leave the debris dead in the sea",
				lam.BasePtr().ID(), len(uses))
		}
	}
	kis := liveOfKind(g, ir.KindRuntimeKernelInfo)
	if len(kis) != 1 {
		t.Fatal("exactly one KernelInfo")
	}
	if items := k.Body.Items(); items[0].Node == nil || items[0].Node.KindOf() != ir.KindRuntimeKernelInfo {
		t.Fatal("the KernelInfo must be the kernel Body's first item")
	}
	// The complete reducer statement set exists and is stamped.
	for _, m := range []string{"Init", "Reduce", "Combine"} {
		if callOf(g, m) == nil {
			t.Fatalf("the %s call must exist", m)
		}
	}
	// Structural health: verify green, linkage consistent, no erased items.
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("the lowered kernel must verify clean, got %v", vs)
	}
	walkItems(k.Body, func(n ir.Node, containing *ir.ASTBlock) {
		if n.BasePtr().Erased() {
			t.Fatalf("AST item references erased node %d (%s)", n.BasePtr().ID(), n.KindOf())
		}
	})
	// Determinism.
	rx2 := newRedFx(t, false, false)
	runRedFx(t, rx2)
	if a, b := astShape(k), astShape(rx2.f.g.AST()); a != b {
		t.Fatalf("twin tiled-reduce runs differ:\n--- 1 ---\n%s\n--- 2 ---\n%s", a, b)
	}
}

// TestConstLiteralDedup pins the canonical-literal invariant: GVN (priority
// 110) dedups pure nodes long BEFORE the passes that mint most const atoms
// (tile-lower's rebuilt bounds, schedule-era expressions, final-lower's own
// constructions), so without a post-lowering canonicalization the same
// literal survives as duplicate nodes. After final-lower at most ONE live
// const node may exist per (type, value) pair, and the graph must still
// verify clean.
func TestConstLiteralDedup(t *testing.T) {
	rx := newRedFx(t, false, false)
	runRedFx(t, rx)
	g := rx.f.g
	// fix3 D-o: the dedup left final-lower (a lower never optimizes) — the
	// pipeline's TAIL owns it now, ahead of its dce instance.
	runTail(t, g)

	type key struct {
		ty   *ir.NumericType
		bits uint64
	}
	seen := make(map[key]ir.Node)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindConst {
			continue
		}
		v, ok := ir.ConstValueOf(n)
		if !ok {
			continue
		}
		// The DECLARED payload type (ConstTypeOf, set at construction) —
		// the typeinfer-computed cell is nil for most post-typeinfer
		// consts, which are exactly the duplicates this pins against.
		ty := ir.ConstTypeOf(n)
		if ty == nil {
			ty = ir.TypeOf(n)
		}
		nt, ok := ty.(*ir.NumericType)
		if !ok || nt == nil {
			continue
		}
		k := key{nt, math.Float64bits(v)}
		if first, dup := seen[k]; dup {
			t.Fatalf("duplicate literal const: node #%d and node #%d both carry %v of %v",
				first.BasePtr().ID(), n.BasePtr().ID(), v, nt)
		}
		seen[k] = n
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("verify after the canonicalization: %v", vs)
	}
}

// callOf returns the unique live RuntimeCallMethod carrying the method
// payload, or nil when not exactly one exists.
func callOf(g *ir.Graph, method string) ir.Node {
	var found ir.Node
	for _, n := range liveOfKind(g, ir.KindRuntimeCallMethod) {
		if ir.RuntimeMethodOf(n) != method {
			continue
		}
		if found != nil {
			return nil // not unique
		}
		found = n
	}
	return found
}

// findItemIn locates node's AST item (block + index) over the whole kernel.
func findItemIn(k *ir.ASTKernel, n ir.Node) (*ir.ASTBlock, int) {
	var search func(blk *ir.ASTBlock) (*ir.ASTBlock, int)
	search = func(blk *ir.ASTBlock) (*ir.ASTBlock, int) {
		for i, it := range blk.Items() {
			if it.Node != nil && it.Node.BasePtr() == n.BasePtr() {
				return blk, i
			}
			if it.Block != nil {
				if b, idx := search(it.Block); b != nil {
					return b, idx
				}
			}
		}
		return nil, -1
	}
	return search(k.Body)
}

// findParentOfBlock locates blk's parental block (holding blk as a Block
// item) and that item's index — the loop-bracket position.
func findParentOfBlock(k *ir.ASTKernel, blk *ir.ASTBlock) (*ir.ASTBlock, int) {
	var search func(parent *ir.ASTBlock) (*ir.ASTBlock, int)
	search = func(parent *ir.ASTBlock) (*ir.ASTBlock, int) {
		for i, it := range parent.Items() {
			if it.Block == blk {
				return parent, i
			}
			if it.Block != nil {
				if b, idx := search(it.Block); b != nil {
					return b, idx
				}
			}
		}
		return nil, -1
	}
	return search(k.Body)
}

// subtreeOf collects n's live data-subtree nodes (self included).
func subtreeOf(n ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || set[x] {
			return
		}
		set[x] = true
		for _, d := range ir.OperandsOf(x) {
			visit(d)
		}
	}
	visit(n)
	return set
}

// ---------------------------------------------------------------------------
// R-A bracket-law pins (2026-10-24): >=2-between-dims, the minimal-legal-tile
// one-between endpoint
// ---------------------------------------------------------------------------

// newRed2Fx builds the 2-dim last-axis reduce fixture (the golden redFx
// shape, decisions supplied by the caller): lhs[d0,d1] loaded over both
// IVs, piped through an INLINED Map as=[d0,d1], Reduce(0, +), stored into
// out[d0]. The minimal-legal-tile endpoint (both decisions at the floor
// tile 2, {CTA,100,2}/{WARP,8,2}) rides this builder. NOTE: the former
// T==1 sweep endpoint and the WARP-single refusal probe were deleted with
// the 2026-10-24 tile-disable ruling (tile 1 is not a legal schedule); the
// old rank-3 SEQ-T==1 not-buildable note died with the single form itself.
func newRed2Fx(t *testing.T, name string) (*fx, ir.Node) {
	t.Helper()
	f := newChain(name, 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0)}, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	return f, store
}

// newRed3Fx builds the rank-3 last-axis reduce fixture: lhs[d0,d1,d2]
// loaded over all three IVs (chained), piped through an INLINED-shape Map
// as=[d0,d1,d2], Reduce(0, +), stored into out[d0,d1] (the two-axis output
// tail the scheduler lands in the epilogue). typeinfer runs for the
// homogeneity types.
func newRed3Fx(t *testing.T, name string) (*fx, ir.Node) {
	t.Helper()
	f := newChain(name, 512, 256, 512)
	lhs := f.tensor("lhs", 0, 1, 2)
	out := f.tensor("out", 0, 1)
	env := ir.NewEnv()
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1), f.iv(2)})
	mp := f.g.NewMap("ctor", []ir.Node{load}, ir.NewLambda(nil, load), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	red := f.g.NewReduce("ctor", f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0), f.iv(1)}, red)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	return f, store
}

// TestReducerBracketMultiBetween pins the R-A bracket law on the
// >=2-between-dims shape (the HI-01 proof): with TWO non-reduced between
// dims the fresh element loops nest outer->inner in dim order and the whole
// bracket [Init immediately before X -> X{home{Reduce}} -> Combine
// immediately after -> the moved store] sits in the INNERMOST element loop
// (X's parent under R-A) — never in the outer element loop, never per
// X-round. The store's d0/d1 coordinates reference the two FRESH element
// dimvars, and no reduced-dim dimvar leaks into its closure.
func TestReducerBracketMultiBetween(t *testing.T) {
	f, store := newRed3Fx(t, "red3")
	cache := runTileLowerScheduleSplitK(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 64},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	prov := provenanceOf(t, cache)
	X, home := prov.Dims[f.dims[2]].Between, prov.Dims[f.dims[2]].Inside
	if X == nil || home == nil {
		t.Fatal("fixture setup: the reduced d2 must be tiled (Between+Inside)")
	}
	if err := runFinalLower(t, f.g, cache); err != nil {
		t.Fatalf("final-lower over the rank-3 reduce: %v", err)
	}
	g, k := f.g, f.g.AST()

	// Exactly one of each bracket call.
	initCall, combineCall, reduceCall := callOf(g, "Init"), callOf(g, "Combine"), callOf(g, "Reduce")
	if initCall == nil || combineCall == nil || reduceCall == nil {
		t.Fatal("exactly one Init, one Reduce, and one Combine call must exist")
	}

	// The innermost element block: Init's containing block under R-A.
	elemBlk, initIdx := findItemIn(k, initCall)
	if elemBlk == nil || initIdx < 0 {
		t.Fatalf("Init must be an item of the INNERMOST element loop, got index %d", initIdx)
	}
	// fix3 D-j: re-placed between items may precede Init; nothing between
	// Init and X's block.
	if elemBlk.Kind() != ir.BlockSeqLoop {
		t.Fatalf("the bracket host must be a SeqLoop (uniform), got %s", elemBlk.Kind())
	}
	// X's parent is the innermost element block; adjacency per R-A.
	xParent, xIdx := findParentOfBlock(k, X.BasePtr().SchedBlock())
	if xParent != elemBlk {
		t.Fatalf("X's parental block must be the INNERMOST element loop (R-A), got %v", xParent)
	}
	if xIdx != initIdx+1 {
		t.Fatalf("Init must sit IMMEDIATELY before X's block: Init at %d, X at %d", initIdx, xIdx)
	}
	items := elemBlk.Items()
	if items[xIdx+1].Node == nil || items[xIdx+1].Node.BasePtr() != combineCall.BasePtr() {
		t.Fatal("Combine must sit IMMEDIATELY after X's block (R-A)")
	}
	if len(items) < xIdx+2 || items[xIdx+2].Node == nil || items[xIdx+2].Node.BasePtr() != store.BasePtr() {
		t.Fatal("the moved store must follow Combine in the innermost element loop")
	}
	// Reduce's innermost containing block is Y (home).
	if rBlk, _ := findItemIn(k, reduceCall); rBlk != home.BasePtr().SchedBlock() {
		t.Fatalf("Reduce's innermost containing block must be Y (home), got %v", rBlk)
	}

	// The innermost element block nests TWO element loops deep: its parent
	// is the OUTER element loop (the outer's only Block item is the inner),
	// and Init is NOT an item of the outer.
	outer, outerIdx := findParentOfBlock(k, elemBlk)
	if outer == nil || outer.Kind() != ir.BlockSeqLoop {
		t.Fatalf("the innermost element loop must nest inside an outer element SeqLoop, got %v", outer)
	}
	nBlocks := 0
	for _, it := range outer.Items() {
		if it.Block != nil {
			nBlocks++
		}
	}
	if nBlocks != 1 || outer.Items()[outerIdx].Block != elemBlk {
		t.Fatal("the outer element loop's only Block item must be the inner element loop")
	}
	if ob, oi := findItemIn(k, initCall); ob == outer {
		t.Fatalf("Init must not be an item of the OUTER element loop (HI-01), found at %d", oi)
	}
	// The loops walk their own dims: inner over d1, outer over d0.
	if d := ir.OperandsOf(ir.OperandsOf(elemBlk.Loop())[0])[0]; d.BasePtr() != f.dims[1].BasePtr() {
		t.Fatal("the innermost element loop must walk d1 (dim order)")
	}
	if d := ir.OperandsOf(ir.OperandsOf(outer.Loop())[0])[0]; d.BasePtr() != f.dims[0].BasePtr() {
		t.Fatal("the outer element loop must walk d0 (dim order)")
	}

	// The store's d0/d1 coordinates reference the two FRESH element
	// dimvars; no reduced-dim dimvar leaks into the closure.
	freshOuter := ir.OperandsOf(outer.Loop())[0]
	freshInner := ir.OperandsOf(elemBlk.Loop())[0]
	axes := ir.OperandsOf(store)[2:]
	if len(axes) != 2 {
		t.Fatalf("the store must carry two axes, got %d", len(axes))
	}
	cl0, cl1 := subtreeOf(axes[0]), subtreeOf(axes[1])
	if !cl0[freshOuter] || cl0[freshInner] {
		t.Fatal("the store's d0 coordinate must reference the OUTER fresh element dimvar")
	}
	if !cl1[freshInner] || cl1[freshOuter] {
		t.Fatal("the store's d1 coordinate must reference the INNER fresh element dimvar")
	}
	for _, loop := range []ir.Node{X, home} {
		iv := ir.OperandsOf(loop)[0]
		if cl0[iv] || cl1[iv] {
			t.Fatal("the moved store must not reference a reduced-dim dimvar")
		}
	}

	// Twin-run determinism.
	f2, _ := newRed3Fx(t, "red3")
	cache2 := runTileLowerScheduleSplitK(t, f2, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 64},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	if err := runFinalLower(t, f2.g, cache2); err != nil {
		t.Fatalf("twin final-lower: %v", err)
	}
	if a, b := astShape(k), astShape(f2.g.AST()); a != b {
		t.Fatalf("twin rank-3 runs differ:\n--- 1 ---\n%s\n--- 2 ---\n%s", a, b)
	}
}

// TestReducerBracketMinTile pins R-A's ONE-between-dim law at the
// minimal-legal-tile endpoint ({CTA,100,2}/{WARP,8,2} — the 2026-10-24
// tile-disable ruling's floor; the former T==1 sweep endpoint is gone):
// the single non-reduced dim's walk is the FRESH ELEMENT loop (it takes
// the position X's single loop held under the old T==1 law), and the
// whole bracket [Init immediately before X -> X{home{Reduce}} -> Combine
// immediately after -> the moved store] sits in that element loop, with
// SchedBlock(combineCall) == SchedBlock(store) (the same-lexical-scope
// contract the emitted C++ needs: the Combine local and its consumer must
// share one C++ scope).
func TestReducerBracketMinTile(t *testing.T) {
	f, store := newRed2Fx(t, "redt1") // identifier kept (R6: fixture names stay)
	cache := runTileLowerScheduleSplitK(t, f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 2},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 2},
	})
	prov := provenanceOf(t, cache)
	X, home := prov.Dims[f.dims[1]].Between, prov.Dims[f.dims[1]].Inside
	if X == nil || home == nil {
		t.Fatal("fixture setup: the reduced d1 must be tiled (Between+Inside)")
	}
	if err := runFinalLower(t, f.g, cache); err != nil {
		t.Fatalf("final-lower over the minimal-tile reduce (the HI-02 consumer move): %v", err)
	}
	g, k := f.g, f.g.AST()

	initCall, combineCall, reduceCall := callOf(g, "Init"), callOf(g, "Combine"), callOf(g, "Reduce")
	if initCall == nil || combineCall == nil || reduceCall == nil {
		t.Fatal("exactly one Init, one Reduce, and one Combine call must exist")
	}
	// The single fresh element block: Init's containing block under R-A.
	elemBlk, initIdx := findItemIn(k, initCall)
	if elemBlk == nil || initIdx < 0 {
		t.Fatalf("Init must be an item of the element loop, got index %d", initIdx)
	}
	// fix3 D-j: re-placed between items may precede Init; nothing between
	// Init and X's block.
	if elemBlk.Kind() != ir.BlockSeqLoop {
		t.Fatalf("the bracket host must be a SeqLoop (uniform), got %s", elemBlk.Kind())
	}
	// X's parent is the element block; adjacency per R-A.
	xParent, xIdx := findParentOfBlock(k, X.BasePtr().SchedBlock())
	if xParent != elemBlk {
		t.Fatalf("X's parental block must be the element loop (R-A), got %v", xParent)
	}
	if xIdx != initIdx+1 {
		t.Fatalf("Init must sit IMMEDIATELY before X's block: Init at %d, X at %d", initIdx, xIdx)
	}
	items := elemBlk.Items()
	if items[xIdx].Block == nil || items[xIdx].Block.Loop().BasePtr() != X.BasePtr() {
		t.Fatal("X's block must be at the bracket center (R-A)")
	}
	if items[xIdx+1].Node == nil || items[xIdx+1].Node.BasePtr() != combineCall.BasePtr() {
		t.Fatal("Combine must sit IMMEDIATELY after X's block (R-A)")
	}
	if items[xIdx+2].Node == nil || items[xIdx+2].Node.BasePtr() != store.BasePtr() {
		t.Fatal("the store must sit IMMEDIATELY after Combine (the HI-02 same-scope move)")
	}
	if combineCall.BasePtr().SchedBlock() != elemBlk || store.BasePtr().SchedBlock() != elemBlk {
		t.Fatal("SchedBlock(combineCall) and SchedBlock(store) must both be the element block")
	}
	// Reduce's innermost containing block is Y (home).
	if rBlk, _ := findItemIn(k, reduceCall); rBlk != home.BasePtr().SchedBlock() {
		t.Fatalf("Reduce's innermost containing block must be Y (home), got %v", rBlk)
	}
	// The element loop walks d0, and the store's d0 coordinate references
	// the FRESH element dimvar; no reduced-dim dimvar leaks into the
	// closure.
	freshElem := ir.OperandsOf(elemBlk.Loop())[0]
	if d := ir.OperandsOf(freshElem)[0]; d.BasePtr() != f.dims[0].BasePtr() {
		t.Fatal("the element loop must walk d0 (the non-reduced dim)")
	}
	axes := ir.OperandsOf(store)[2:]
	if len(axes) != 1 {
		t.Fatalf("the store must carry one axis, got %d", len(axes))
	}
	if cl := subtreeOf(axes[0]); !cl[freshElem] {
		t.Fatal("the store's d0 coordinate must reference the FRESH element dimvar")
	}
	if ops := ir.OperandsOf(store); ops[0] == nil || ops[0].BasePtr() != combineCall.BasePtr() {
		t.Fatal("the store's value must resolve at the Combine call")
	}
}

// TestReducerBracketMinTile's companion refusal probe (the WARP-single
// consumer refusal) was DELETED with the 2026-10-24 tile-disable ruling:
// the single form it guarded no longer exists, and its gate
// (refuseThreadPartitionedConsumers + warpSingleIVs) was removed from
// reducer.go with the T==1 machinery.

// ---------------------------------------------------------------------------
// fix3 D-p battery: the reduce anchor (final-lower's final gate)
// ---------------------------------------------------------------------------

// newSeveredRedFx builds the HAND-SEVERED tiled-reduce fixture (fix3 D-p
// battery negative (b)): identical to newRedFx except the piped Map's
// producer is a loop-INVARIANT const. The lhs load still happens (chained
// through the same env — the effect chain and every structural invariant
// stay intact), but nothing in the Reduce's DATA closure references memory
// anymore: the 2026-10-24 "Combine returns the init, every output is 0"
// wall class, severed by hand at the exact edge the anchor exists to name.
func newSeveredRedFx(t *testing.T) *redFx {
	t.Helper()
	f := newChain("redsev", 1024, 512)
	lhs := f.tensor("lhs", 0, 1)
	out := f.tensor("out", 0)
	env := ir.NewEnv()
	// The severing: the piped producer is a pure REGISTER expression over
	// the home IVs — structurally sound (it references the loop IVs, so it
	// lowers inside home and passes every bracket-law check) but it reads
	// NO memory. The lhs load still happens and stays chained into the
	// epilogue consumer only: a memory access exists in the kernel, even
	// effect-chained, yet the Reduce's DATA closure reaches none of it —
	// the anchor must not count it (liveness never crosses chain edges).
	load := f.g.NewMemLoadChained("ctor", env, lhs, []ir.Node{f.iv(0), f.iv(1)})
	konst := f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 1)
	prod := f.g.NewBinOp("ctor", ir.BinOpAdd,
		f.g.NewCast("ctor", ir.InternScalar(ir.TF32), f.iv(0)),
		f.g.NewBinOp("ctor", ir.BinOpAdd,
			f.g.NewCast("ctor", ir.InternScalar(ir.TF32), f.iv(1)), konst))
	mp := f.g.NewMap("ctor", []ir.Node{prod}, ir.NewLambda(nil, prod), append([]ir.Node(nil), f.dims...))
	acc := f.g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := f.g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := f.g.NewBinOp("ctor", ir.BinOpAdd, f.g.NewParamRef("ctor", acc), f.g.NewParamRef("ctor", now))
	init := f.g.NewConst("ctor", ir.InternScalar(ir.TF32), 0)
	red := f.g.NewReduce("ctor", init, mp, ir.NewLambda([]ir.Node{acc, now}, comb))
	lambda := ir.LambdaOf(red)
	store := f.g.NewMemStoreChained("ctor", env, out, []ir.Node{f.iv(0)}, red)
	// A poison-style epilogue consumer (the newRedFx knob's pattern) keeps
	// both iteration chains sea-reachable — WITHOUT entering the Reduce's
	// value closure: the severing stays surgical.
	kf32 := f.g.NewCast("ctor", ir.InternScalar(ir.TF32), f.iv(1))
	keepAlive := f.g.NewBinOp("ctor", ir.BinOpAdd, kf32, load)
	f.g.NewReturnChained("ctor", env, []ir.Node{out, lhs}, store, keepAlive)
	if err := (typeinfer.Pass{}).Run(nil, f.g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	return &redFx{f: f, load: load, mp: mp, red: red, init: init, lambda: lambda, store: store}
}

// TestReduceAnchorSeveredChainFailsClosed pins fix3 D-p battery negative
// (b): the anchor fires ON THE LOWERING — a Reduce whose value closure
// reaches no memory access fails final-lower with its named error, never
// at numerical acceptance. The tile-lower → schedule prefix must still
// run clean (the severing is type-legal and chain-legal; only the
// anchor's semantic predicate rejects it).
func TestReduceAnchorSeveredChainFailsClosed(t *testing.T) {
	sx := newSeveredRedFx(t)
	cache := runTileLowerScheduleSplitK(t, sx.f, []tiling.IterDecision{
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
		{SoC: tiling.SoCWARP, LoC: 8, TileSize: 64},
	})
	err := runFinalLower(t, sx.f.g, cache)
	if err == nil {
		t.Fatal("final-lower must fail closed on the hand-severed Reduce chain — the anchor's named error — got nil")
	}
	if !strings.Contains(err.Error(), "reduce anchor") {
		t.Fatalf("want the reduce-anchor error on the severed chain, got: %v", err)
	}
}
