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
