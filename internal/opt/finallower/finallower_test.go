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
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/finallower"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling"
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
// returning the cache that now holds the published provenance.
func runTileLowerSchedule(t *testing.T, f *fx, decisions []tiling.IterDecision) *opt.Cache {
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
	if err := (tiling.LowerPass{Params: tiling.DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, f.g); err != nil {
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
		{SoC: tiling.SoCTA, LoC: 100, TileSize: 1},
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

	// The interim reducer gate: a live KindReduce at this stage errors
	// naming the reducer task.
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
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 1},
	})
	if err := runFinalLower(t, f3.g, cache3); err == nil ||
		!strings.Contains(err.Error(), "reducer task") {
		t.Fatalf("a live Reduce must hit the interim gate naming the reducer task, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// Map Lower + objects + query rewrite (the map-only end-to-end)
// ---------------------------------------------------------------------------

// mapOnlyFx builds the map-only kernel: a 2-dim chain (CTA tiled + SEQ
// untiled), a GMEM load over both IVs, an INLINED-shape Map over it with
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
		{SoC: tiling.SoCSequential, LoC: 1, TileSize: 1},
	})
	if err := runFinalLower(t, f.g, cache); err != nil {
		t.Fatalf("final-lower over the map-only kernel: %v", err)
	}
	return f, f.g.AST(), load, mp, out
}

// TestMapLowerEliminatesMap pins LR-9: the Map's every use is RAUW'd onto
// its body expression (the store now reads the load), and the Map node, its
// trailing As, and its orphaned lambda are all erased — no dangling
// Map/As/lambda debris survives.
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
	// No dangling Map / As / lambda debris.
	for _, k := range []ir.Kind{ir.KindMap, ir.KindAs, ir.KindLambda} {
		if got := liveOfKind(f.g, k); len(got) != 0 {
			t.Fatalf("no live %s may survive Map Lower, got %d", k, len(got))
		}
	}
	if vs := ir.Verify(f.g); len(vs) != 0 {
		t.Fatalf("the lowered graph must verify clean, got %v", vs)
	}
}

// TestRuntimeObjectsInserted pins LR-11: exactly one RuntimeKernelInfo as
// item 0 of the kernel Body Group (SN 0, Prototype untouched — its items
// are exactly the kernel params), and every LNT's prologue Group starts
// with [RuntimeGridMapping, RuntimeThreadMapping] built from the provenance
// plan (grid [-1]; thread [AutoWarpPortion, -1] — the CTA-dim branch), both
// stamped SN 0/prologue.
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
		// Operands per P-3: [kernel_info, ndim, portions...].
		checkMappingOperands(t, grid, ki, []int{-1}, "grid")
		checkMappingOperands(t, thread, ki, []int{tiling.AutoWarpPortion, -1}, "thread")
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
	// (the queries' vocabulary), operand 0 is a mapping construction, and
	// exactly one const-u32 arg.
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
		if len(ops) != 2 || ops[0] == nil {
			t.Fatalf("index/dimension getattr must carry [mapping, DoSoC], got %v", ops)
		}
		if k := ops[0].KindOf(); k != ir.KindRuntimeGridMapping && k != ir.KindRuntimeThreadMapping {
			t.Fatalf("getattr operand 0 kind = %s, want a mapping construction", k)
		}
		if v, ok := ir.ConstValueOf(ops[1]); !ok || v < 0 {
			t.Fatalf("getattr DoSoC arg must be a const, got %v", ops[1])
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

// TestMapOnlyEndToEnd pins the map-only kernel's full post-pass shape:
// zero Map/lambda-debris/query nodes, exactly one KernelInfo as the first
// statement, one mapping pair per LNT prologue, every former query use
// resolved through getattrs, no AST item referencing an erased node, verify
// green, and determinism (twin runs produce identical AST item orders).
func TestMapOnlyEndToEnd(t *testing.T) {
	f, k, _, _, _ := runMapOnly(t)

	// No query / map / debris kinds anywhere.
	for _, kind := range []ir.Kind{ir.KindMap, ir.KindAs, ir.KindLambda,
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
