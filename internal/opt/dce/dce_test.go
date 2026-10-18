package dce_test

import (
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/dce"
)

// dce_test.go pins the mark-sweep DCE pass (quick-261018-l36). Task 2 covers
// the entry gates, the basic sweep, and mark hygiene; task 3 extends this
// file with the soundness-hinge suite (chain liveness through ImpactBy,
// load-chained stores, dead allocations, lambda bodies, idempotence, cyclic
// conservatism, Manager interop).

// nodeSet collects a graph's AllocatedNodes into a set keyed by the shared
// Base cell — the erased-ness and survivor assertion substrate.
func nodeSet(g *ir.Graph) map[*ir.Base]bool {
	set := make(map[*ir.Base]bool)
	for _, n := range g.AllocatedNodes() {
		set[n.BasePtr()] = true
	}
	return set
}

// sweepFixture builds the basic-sweep graph: a dim param, a tensor param,
// a live value tree (two consts + binop), an env-chained store to the param
// whose value is the tree, and the end node carrying the store as its
// returned value. deadAlloc (a fresh GMEM allocation nothing consumes) and
// deadExpr (a pure-expr subtree nothing references) are the dead code.
func sweepFixture() (g *ir.Graph, out, sum, st, end, deadAlloc, deadExpr ir.Node) {
	g = ir.NewGraph("sweep")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	c1 := g.NewConst("ctor", f32, 1)
	c2 := g.NewConst("ctor", f32, 2)
	sum = g.NewBinOp("ctor", ir.BinOpAdd, c1, c2)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, sum)
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	// Dead code: an in-kernel GMEM allocation nothing consumes (its extent
	// const dies with it), and an abandoned pure-expr subtree.
	deadExtent := g.NewConst("ctor", f32, 7)
	deadAlloc = g.NewMemNew("ctor", deadExtent)
	dc1 := g.NewConst("ctor", f32, 3)
	dc2 := g.NewConst("ctor", f32, 4)
	deadExpr = g.NewBinOp("ctor", ir.BinOpMul, dc1, dc2)
	return g, out, sum, st, end, deadAlloc, deadExpr
}

func TestDCEGateNilGraph(t *testing.T) {
	err := dce.Pass{}.Run(nil, nil)
	if err == nil {
		t.Fatal("Run(nil) must return a deterministic error")
	}
	if err.Error() != "dce: nil graph" {
		t.Fatalf("Run(nil) error = %q, want %q", err.Error(), "dce: nil graph")
	}
}

func TestDCEGateMaterializedRefused(t *testing.T) {
	g, _, _, _, _, _, _ := sweepFixture()
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := len(g.AllocatedNodes())

	err := dce.Pass{}.Run(nil, g)
	if err == nil {
		t.Fatal("Run on a materialized graph must fail")
	}
	if !strings.Contains(err.Error(), "materialized") {
		t.Fatalf("error %q must mention the materialized refusal", err.Error())
	}
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("a refused graph must be untouched: %d nodes before, %d after", before, after)
	}
}

func TestDCEBasicSweep(t *testing.T) {
	g, out, sum, st, end, deadAlloc, deadExpr := sweepFixture()

	if err := (dce.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}

	set := nodeSet(g)
	for name, dead := range map[string]ir.Node{
		"dead alloc": deadAlloc,
		"dead expr":  deadExpr,
	} {
		if set[dead.BasePtr()] {
			t.Fatalf("%s survived the sweep", name)
		}
	}
	for name, live := range map[string]ir.Node{
		"tensor param": out,
		"value tree":   sum,
		"store":        st,
		"end node":     end,
	} {
		if !set[live.BasePtr()] {
			t.Fatalf("%s was erased by the sweep", name)
		}
	}

	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the sweep: %v", vs)
	}
}

func TestDCEMarkHygiene(t *testing.T) {
	g, _, _, _, _, _, _ := sweepFixture()

	// A stale mark from an aborted prior run must not protect dead code:
	// mark the dead alloc, then confirm the pass still sweeps it (ResetMarks
	// on entry clears the stale state).
	if err := (dce.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, n := range g.AllocatedNodes() {
		if n.BasePtr().Alive() {
			t.Fatalf("node %d still alive after Run — marks must never leak out of the pass", n.BasePtr().ID())
		}
	}
}

// TestDCEStaleMarkCannotProtect pins T-l36-01 directly: pre-seeding a dead
// node's mark before Run does not save it, because Run resets marks first.
func TestDCEStaleMarkCannotProtect(t *testing.T) {
	g, _, _, _, _, deadAlloc, _ := sweepFixture()
	deadAlloc.BasePtr().MarkAlive()

	if err := (dce.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if nodeSet(g)[deadAlloc.BasePtr()] {
		t.Fatal("a stale alive mark must not protect a dead node (ResetMarks runs first)")
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the sweep: %v", vs)
	}
}

// TestDCEManagerInterop pins that the registry-era pass still interops with
// the untouched Manager/Instrumentation/legality machinery: a [dce] pipeline
// with VerifyEach runs, sweeps, and verifies — mirroring pkg/builder.go's
// usage shape.
func TestDCEManagerInterop(t *testing.T) {
	g, _, _, st, _, deadAlloc, _ := sweepFixture()

	pm := opt.NewManager([]opt.Pass{dce.Pass{}})
	pm.AddInstrumentation(opt.NewVerifyEach(nil))
	if err := pm.Run(g); err != nil {
		t.Fatalf("manager pipeline: %v", err)
	}

	set := nodeSet(g)
	if set[deadAlloc.BasePtr()] {
		t.Fatal("the sweep must have happened under the manager")
	}
	if !set[st.BasePtr()] {
		t.Fatal("the store must survive under the manager")
	}
}

// ---------------------------------------------------------------------------
// Soundness hinges (task 3): the exact liveness semantics of the two-root-set
// model, pinned graph by graph.
// ---------------------------------------------------------------------------

// runDCE runs the pass and fails the test on error.
func runDCE(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (dce.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("dce.Run: %v", err)
	}
}

// assertGone fails the test when any of the nodes is still allocated.
func assertGone(t *testing.T, g *ir.Graph, nodes ...ir.Node) {
	t.Helper()
	set := nodeSet(g)
	for _, n := range nodes {
		if n != nil && set[n.BasePtr()] {
			t.Fatalf("node %d (%s) must be swept", n.BasePtr().ID(), n.KindOf())
		}
	}
}

// assertKept fails the test when any of the nodes is missing.
func assertKept(t *testing.T, g *ir.Graph, nodes ...ir.Node) {
	t.Helper()
	set := nodeSet(g)
	for _, n := range nodes {
		if !set[n.BasePtr()] {
			t.Fatalf("node %d (%s) must survive", n.BasePtr().ID(), n.KindOf())
		}
	}
}

// Hinge 1 (requirement 4): a GMEM store stays alive purely through the end
// node's ImpactBy effect chain even when no data edge connects it to the
// return value. Two chained stores to one mem param: the end node's ImpactBy
// names only the chain HEAD (store2), so store1 is reachable exclusively via
// store2's ImpactBy edge — and must survive.
func TestDCEHingeChainLiveness(t *testing.T) {
	g := ir.NewGraph("hinge1")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	c1 := g.NewConst("ctor", f32, 1)
	c2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	store1 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	store2 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c2)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, store2)

	runDCE(t, g)
	assertKept(t, g, m, out, c1, c2, store1, store2, end)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// Hinge 2: loads chain too (soundness hinge). A store followed by a chained
// LOAD on the same param, the load's value returned via a data operand — the
// store survives through the LOAD's ImpactBy edge alone.
func TestDCEHingeLoadChainedStore(t *testing.T) {
	g := ir.NewGraph("hinge2")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	v := g.NewConst("ctor", f32, 1)
	env := ir.NewEnv()
	store := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	load := g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, load)

	runDCE(t, g)
	assertKept(t, g, m, out, v, store, load, end)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// Hinge 3 (requirement 5, the user's example): an in-kernel GMEM allocation
// with no contribution to the return value or any by-ref parameter is
// removed. The store's target tensor is the alloc NODE itself — the env keys
// chains by the tensor node, so the store chains on the alloc — and
// NewReturnChained is called with the REAL mem params only, so the end
// node's ImpactBy never reaches the alloc's chain. Nothing observable
// (no param, no return path) reaches the alloc, the store, or the value
// tree: all three are swept.
func TestDCEHingeDeadGMEMAlloc(t *testing.T) {
	g := ir.NewGraph("hinge3")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	extent := g.NewConst("ctor", f32, 64)
	alloc := g.NewMemNew("ctor", extent)
	vc1 := g.NewConst("ctor", f32, 1)
	vc2 := g.NewConst("ctor", f32, 2)
	val := g.NewBinOp("ctor", ir.BinOpAdd, vc1, vc2)
	env := ir.NewEnv()
	deadStore := g.NewMemStoreChained("ctor", env, alloc, []ir.Node{m}, val)
	// A LIVE store to the real param keeps the graph honest.
	liveVal := g.NewConst("ctor", f32, 3)
	liveStore := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, liveVal)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, liveStore)

	runDCE(t, g)
	assertGone(t, g, alloc, deadStore, val, vc1, vc2, extent)
	assertKept(t, g, m, out, liveVal, liveStore, end)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// Hinge 4 (in-program memory): a dead NewMemNewSMem + NewMemStoreSMem pair
// nothing reads is swept — SMEM is not observable kernel I/O (R3).
func TestDCEHingeDeadSMemPair(t *testing.T) {
	g := ir.NewGraph("hinge4")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	extent := g.NewConst("ctor", f32, 32)
	smem := g.NewMemNewSMem("ctor", extent)
	v := g.NewConst("ctor", f32, 1)
	deadStore := g.NewMemStoreSMem("ctor", v, smem)
	liveVal := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	liveStore := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, liveVal)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, liveStore)

	runDCE(t, g)
	assertGone(t, g, smem, deadStore, v, extent)
	assertKept(t, g, m, out, liveVal, liveStore, end)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// Hinge 5 (lambda bodies): an abandoned Map whose lambda body references a
// dedicated const, consumed by nothing — the map AND the lambda-body subtree
// AND the otherwise-unreferenced const are swept; same for an abandoned
// Reduce. Walk's child relation includes lambda bodies, so the mirror holds:
// nothing reaches them once the stage node is unreachable.
func TestDCEHingeLambdaBodies(t *testing.T) {
	g := ir.NewGraph("hinge5")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	// Abandoned Map: piped const + lambda(l){ body referencing l and a
	// dedicated const } + explicit as-order dim.
	piped := g.NewConst("ctor", f32, 1)
	lp := g.NewLambdaInputParam("ctor", "l", f32)
	dedicated := g.NewConst("ctor", f32, 2)
	body := g.NewBinOp("ctor", ir.BinOpAdd, lp, dedicated)
	lam := ir.NewLambda([]ir.Node{lp}, body)
	deadMap := g.NewMap("ctor", []ir.Node{piped}, lam, []ir.Node{m})

	// Abandoned Reduce: init + piped consts + combiner lambda with its own
	// dedicated-const body.
	ri := g.NewConst("ctor", f32, 3)
	rp := g.NewConst("ctor", f32, 4)
	rl := g.NewLambdaInputParam("ctor", "acc", f32)
	rdedicated := g.NewConst("ctor", f32, 5)
	rbody := g.NewBinOp("ctor", ir.BinOpMul, rl, rdedicated)
	rlam := ir.NewLambda([]ir.Node{rl}, rbody)
	deadReduce := g.NewReduce("ctor", ri, rp, rlam)

	liveVal := g.NewConst("ctor", f32, 6)
	env := ir.NewEnv()
	liveStore := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, liveVal)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, liveStore)

	runDCE(t, g)
	assertGone(t, g, deadMap, piped, lp, dedicated, body)
	assertGone(t, g, deadReduce, ri, rp, rl, rdedicated, rbody)
	assertKept(t, g, m, out, liveVal, liveStore, end)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// TestDCERootsSurvive: every Params() node and every operand of End()
// survives unconditionally — the two-root set is untouchable by the sweep.
func TestDCERootsSurvive(t *testing.T) {
	g := ir.NewGraph("roots")
	f32 := ir.InternScalar(ir.TF32)
	// The dim param node is asserted via the Params() loop below; only its
	// name feeds the tensor's DimRef here.
	g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	sc := g.NewInputParam("ctor", ir.RoleScalar, "alpha", f32)
	r1 := g.NewConst("ctor", f32, 1)
	r2 := g.NewBinOp("ctor", ir.BinOpMul, sc, r1)
	end := g.NewReturnChained("ctor", ir.NewEnv(), []ir.Node{out}, r2)
	dead := g.NewConst("ctor", f32, 9)

	runDCE(t, g)
	for _, p := range g.Params() {
		assertKept(t, g, p)
	}
	assertKept(t, g, r1, r2, end)
	assertGone(t, g, dead)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// TestDCEIdempotent: a second Run erases nothing and returns nil.
func TestDCEIdempotent(t *testing.T) {
	g, _, _, _, _, _, _ := sweepFixture()
	runDCE(t, g)
	count := len(g.AllocatedNodes())

	if err := (dce.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if got := len(g.AllocatedNodes()); got != count {
		t.Fatalf("second Run erased nodes: %d before, %d after", count, got)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
}

// TestDCECyclicClusterSurvives: two raw nodes made mutually-referencing via
// MutContext.AddUse, unreachable from the roots — both survive (conservative
// by construction: every member keeps one live-looking use) and the graph
// still verifies.
func TestDCECyclicClusterSurvives(t *testing.T) {
	g, _, _, _, _, _, _ := sweepFixture()
	a := g.NewNode(ir.KindBinOp, "t")
	b := g.NewNode(ir.KindBinOp, "t")
	mctx := &ir.MutContext{Pass: "t"}
	mctx.AddUse(a, b)
	mctx.AddUse(b, a)

	runDCE(t, g)
	assertKept(t, g, a, b)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify with the cyclic survivors: %v", vs)
	}
}
