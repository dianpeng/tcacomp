package memopt_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/dce"
	"tcacomp/internal/opt/gvn"
	"tcacomp/internal/opt/memopt"
)

// memopt_test.go pins the memory-optimization pass and its verifier
// (quick-261020-ejf) as the review's ten property-test families: gates,
// forwarding (firing and blocked), chain splicing, store-store elimination,
// dead stores, the C1 eligibility guard, head removal, idempotence/Manager
// interop, and the verifier's own behavior. Fixtures are built ONLY through
// the ir constructors — chained constructors for well-formed regions, plain
// NewMemStore to plant C1 violations — exactly the review's expectation
// note that today's builder graphs rarely fire the rules.

// ---------------------------------------------------------------------------
// Shared helpers (dce_test.go's substrate).
// ---------------------------------------------------------------------------

// nodeSet collects a graph's AllocatedNodes into a set keyed by the shared
// Base cell — the erased-ness and survivor assertion substrate.
func nodeSet(g *ir.Graph) map[*ir.Base]bool {
	set := make(map[*ir.Base]bool)
	for _, n := range g.AllocatedNodes() {
		set[n.BasePtr()] = true
	}
	return set
}

// assertGone fails the test when any of the nodes is still allocated.
func assertGone(t *testing.T, g *ir.Graph, nodes ...ir.Node) {
	t.Helper()
	set := nodeSet(g)
	for _, n := range nodes {
		if n != nil && set[n.BasePtr()] {
			t.Fatalf("node %d (%s) must be erased", n.BasePtr().ID(), n.KindOf())
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

// assertImpactBy fails the test unless n's first ImpactBy effect edge points
// at want — the chain-integrity assertion (dce_test.go's helper).
func assertImpactBy(t *testing.T, n, want ir.Node) {
	t.Helper()
	imp := ir.ImpactBy(n)
	if len(imp) == 0 {
		t.Fatalf("node %d (%s) has no ImpactBy edge", n.BasePtr().ID(), n.KindOf())
	}
	if got := imp[0].Def(); got == nil || got.BasePtr() != want.BasePtr() {
		t.Fatalf("node %d (%s) ImpactBy points at %v, want node %d (%s)",
			n.BasePtr().ID(), n.KindOf(), got, want.BasePtr().ID(), want.KindOf())
	}
}

// assertNoImpactBy fails the test when n still carries an ordering edge.
func assertNoImpactBy(t *testing.T, n ir.Node) {
	t.Helper()
	if imp := ir.ImpactBy(n); len(imp) != 0 {
		t.Fatalf("node %d (%s) must carry no ImpactBy edge, got %d",
			n.BasePtr().ID(), n.KindOf(), len(imp))
	}
}

// runMemopt runs the pass and fails the test on error.
func runMemopt(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (memopt.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("memopt.Run: %v", err)
	}
}

// runMemVerify runs the verifier and fails the test on error.
func runMemVerify(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (memopt.VerifyPass{}).Run(nil, g); err != nil {
		t.Fatalf("memopt-verify.Run: %v", err)
	}
}

// assertClean pins INV-A: the graph verifies and the pass's mutations ride
// the one memopt MutContext (every touched memory node carries the stamp).
func assertClean(t *testing.T, g *ir.Graph, touched ...ir.Node) {
	t.Helper()
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	for _, n := range touched {
		if got := n.BasePtr().LastTouchedBy(); got != "" && got != "memopt" {
			t.Fatalf("node %d (%s) LastTouchedBy = %q, want \"\" or \"memopt\"",
				n.BasePtr().ID(), n.KindOf(), got)
		}
	}
}

// memGraph starts the standard fixture shape: one dim param m plus a second
// dim param m2 (the different-position axis), and one rank-1 tensor out.
func memGraph(name string) (g *ir.Graph, m, m2, out ir.Node) {
	g = ir.NewGraph(name)
	f32 := ir.InternScalar(ir.TF32)
	_ = f32
	m = g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	m2 = g.NewInputParam("ctor", ir.RoleDim, "m2", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	return g, m, m2, out
}

// ---------------------------------------------------------------------------
// Family 1 — gates: nil graph exact error; materialized graph refused
// untouched.
// ---------------------------------------------------------------------------

func TestMemoptGateNilGraph(t *testing.T) {
	err := memopt.Pass{}.Run(nil, nil)
	if err == nil {
		t.Fatal("Run(nil) must return a deterministic error")
	}
	if err.Error() != "memopt: nil graph" {
		t.Fatalf("Run(nil) error = %q, want %q", err.Error(), "memopt: nil graph")
	}
}

func TestMemoptGateMaterializedRefused(t *testing.T) {
	g, _, out, _, _, _, _ := forwardFixture()
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := len(g.AllocatedNodes())

	err := memopt.Pass{}.Run(nil, g)
	if err == nil {
		t.Fatal("Run on a materialized graph must fail")
	}
	if !strings.Contains(err.Error(), "materialized") {
		t.Fatalf("error %q must mention the materialized refusal", err.Error())
	}
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("a refused graph must be untouched: %d nodes before, %d after", before, after)
	}
	_ = out
}

// ---------------------------------------------------------------------------
// Fixtures (families 2-8). Every mutating fixture is also consumed by
// family 10's verifier sweep.
// ---------------------------------------------------------------------------

// forwardFixture (family 2): S(m, v) → L(m) with the load returned — the
// canonical forwarding graph.
func forwardFixture() (g *ir.Graph, m, out, v, st, load, end ir.Node) {
	g = ir.NewGraph("fwd")
	f32 := ir.InternScalar(ir.TF32)
	m = g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	v = g.NewConst("ctor", f32, 42)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, load)
	return g, m, out, v, st, load, end
}

// forwardBlockedPosFixture (family 3): the load subscribes a DIFFERENT dim
// param than the store — the paths are not proven equal, no forward.
func forwardBlockedPosFixture() (g *ir.Graph, m, m2, out, st, load, end ir.Node) {
	g, m, m2, out = memGraph("fwd-pos")
	f32 := ir.InternScalar(ir.TF32)
	v := g.NewConst("ctor", f32, 1)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m2})
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, load)
	return g, m, m2, out, st, load, end
}

// forwardBlockedUnequalFixture (family 3): an unequal-position store sits
// between the same-position store and the load — IT is the first store
// reached, so the most-recent-store discipline blocks the forward.
func forwardBlockedUnequalFixture() (g *ir.Graph, m, m2, out, s1, s2, load, end ir.Node) {
	g, m, m2, out = memGraph("fwd-uneq")
	f32 := ir.InternScalar(ir.TF32)
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m2}, v2)
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, load)
	return g, m, m2, out, s1, s2, load, end
}

// forwardNoStoreFixture (family 3): a lone chained load — no predecessor
// store at all, nothing to forward.
func forwardNoStoreFixture() (g *ir.Graph, m, out, load, end ir.Node) {
	g, m, _, out = memGraph("fwd-nostore")
	env := ir.NewEnv()
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, load)
	return g, m, out, load, end
}

// twoLoadsFixture (family 4a): S(m, v) → L1(m) → L2(m), both loads
// returned — both forward against the same store in one run.
func twoLoadsFixture() (g *ir.Graph, m, out, v, st, l1, l2, end ir.Node) {
	g = ir.NewGraph("two-loads")
	f32 := ir.InternScalar(ir.TF32)
	m = g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	v = g.NewConst("ctor", f32, 7)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	l1 = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	l2 = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, l1, l2)
	return g, m, out, v, st, l1, l2, end
}

// hingeSpliceFixture (family 4b, the hinge-6 shape): s1(m) → L(m) → s2(m2)
// — the middle load forwards against s1 and s2's ImpactBy edge is SPLICE-
// RELINKED to s1 (the exact chain splice DCE documented it does not do).
// s2 holds a different position, so no store-store follows: the splice is
// observable in isolation.
func hingeSpliceFixture() (g *ir.Graph, m, m2, out, s1, load, s2 ir.Node) {
	g, m, m2, out = memGraph("hinge-splice")
	f32 := ir.InternScalar(ir.TF32)
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m})
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m2}, v2)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	return g, m, m2, out, s1, load, s2
}

// storeStoreFixture (family 5 / family 8): s1(m, v1) → s2(m, v2), end wired
// at s2 — the same-position shadow with no read between.
func storeStoreFixture() (g *ir.Graph, m, out, v1, v2, s1, s2, end ir.Node) {
	g = ir.NewGraph("s-s")
	f32 := ir.InternScalar(ir.TF32)
	m = g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	v1 = g.NewConst("ctor", f32, 1)
	v2 = g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v2)
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	return g, m, out, v1, v2, s1, s2, end
}

// storeStoreBlockedByLoadFixture (family 5): s1(m) → L(m2) → s2(m) — the
// interleaved load (itself at an unforwardable position) stops the
// store-store walk: s1 may have been read.
func storeStoreBlockedByLoadFixture() (g *ir.Graph, m, m2, out, s1, load, s2 ir.Node) {
	g, m, m2, out = memGraph("s-l-s")
	f32 := ir.InternScalar(ir.TF32)
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	load = g.NewMemLoadChained("ctor", env, out, []ir.Node{m2})
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v2)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	return g, m, m2, out, s1, load, s2
}

// storeStoreDiffPosFixture (family 5): s1(m) → s2(m2) — different
// positions, both stores stay.
func storeStoreDiffPosFixture() (g *ir.Graph, m, m2, out, s1, s2 ir.Node) {
	g, m, m2, out = memGraph("s-s-diff")
	f32 := ir.InternScalar(ir.TF32)
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m2}, v2)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	return g, m, m2, out, s1, s2
}

// walkPastFixture (family 5): s0(m) → s1(m2) → s2(m) — the walk passes the
// unequal middle store and kills s0; s1 stays.
func walkPastFixture() (g *ir.Graph, m, m2, out, s0, s1, s2 ir.Node) {
	g, m, m2, out = memGraph("walk-past")
	f32 := ir.InternScalar(ir.TF32)
	v0 := g.NewConst("ctor", f32, 0)
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s0 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v0)
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m2}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v2)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	return g, m, m2, out, s0, s1, s2
}

// cascadeFixture (family 5): s1(m) → s2(m) → s3(m) — a three-store
// same-position cascade collapses to the head-observed s3 in one walk.
func cascadeFixture() (g *ir.Graph, m, out, s1, s2, s3 ir.Node) {
	g, m, _, out = memGraph("cascade")
	f32 := ir.InternScalar(ir.TF32)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 1))
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 2))
	s3 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 3))
	g.NewReturnChained("ctor", env, []ir.Node{out}, g.NewConst("ctor", f32, 9))
	return g, m, out, s1, s2, s3
}

// deadTailFixture (family 6): tensor a carries an unwired s1 → s2 tail
// nothing reads (the doc's post-loop-fusion intermediate-GMEM shape);
// tensor b keeps a live wired store so the end node exists honestly.
func deadTailFixture() (g *ir.Graph, m, out, s1, s2 ir.Node) {
	g, m, _, out = memGraph("dead-tail")
	f32 := ir.InternScalar(ir.TF32)
	env := ir.NewEnv()
	v1 := g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v2)
	// A second, wired tensor keeps the return honest.
	b := g.NewInputParam("ctor", ir.RoleTensor, "b", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	vb := g.NewConst("ctor", f32, 3)
	sb := g.NewMemStoreChained("ctor", env, b, []ir.Node{m}, vb)
	g.NewReturnChained("ctor", env, []ir.Node{b}, vb)
	_ = sb
	return g, m, out, s1, s2
}

// dataUseStoreFixture (family 6): an unwired dead store whose VALUE is
// consumed by a binop — the sweep RAUWs the data use onto the value
// operand, then erases the store; the pure consumer survives untouched.
func dataUseStoreFixture() (g *ir.Graph, m, out, v, st, r ir.Node) {
	g, m, _, out = memGraph("data-use")
	f32 := ir.InternScalar(ir.TF32)
	v = g.NewConst("ctor", f32, 5)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	c := g.NewConst("ctor", f32, 1)
	r = g.NewBinOp("ctor", ir.BinOpAdd, st, c)
	g.NewReturnChained("ctor", ir.NewEnv(), nil, r)
	return g, m, out, v, st, r
}

// wiredTailFixture (family 6, INV-D): a single Return-wired store — the
// end node's ImpactBy keeps its payload use, so it is NEVER erased even
// though nothing else reads it.
func wiredTailFixture() (g *ir.Graph, m, out, v, st ir.Node) {
	g, m, _, out = memGraph("wired-tail")
	f32 := ir.InternScalar(ir.TF32)
	v = g.NewConst("ctor", f32, 8)
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v)
	return g, m, out, v, st
}

// c1SkipFixture (family 7): s1 → s2 chained, PLUS a third same-tensor
// store built through the plain un-chained NewMemStore — the region's
// access set no longer forms one linear chain, so C1 skips the whole
// region: no error, no mutation.
func c1SkipFixture() (g *ir.Graph, m, out, v1, s1, s2, su ir.Node) {
	g, m, _, out = memGraph("c1-skip")
	f32 := ir.InternScalar(ir.TF32)
	v1 = g.NewConst("ctor", f32, 1)
	v2 := g.NewConst("ctor", f32, 2)
	env := ir.NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v1)
	s2 = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, v2)
	g.NewReturnChained("ctor", env, []ir.Node{out}, v2)
	// The C1 violation: a same-tensor, same-position store OUTSIDE the
	// chain (built after the return so nothing else observes it either).
	su = g.NewMemStore("ctor", out, []ir.Node{m}, v1)
	return g, m, out, v1, s1, s2, su
}

// ---------------------------------------------------------------------------
// Family 2 — forwarding fires.
// ---------------------------------------------------------------------------

func TestMemoptForwardingFires(t *testing.T) {
	g, m, out, v, st, load, end := forwardFixture()

	runMemopt(t, g)
	assertGone(t, g, load)
	assertKept(t, g, m, out, v, st, end)

	// The return's data operand now sees the stored value (INV-D: same
	// VALUE, via the RAUW-to-value), and its ImpactBy still observes the
	// surviving chain head — the store.
	ops := ir.OperandsOf(end)
	if len(ops) != 1 || ops[0] != v {
		t.Fatalf("return data operand must be the stored value, got %v", ops)
	}
	assertImpactBy(t, end, st)
	assertClean(t, g, load, end, st)
}

// ---------------------------------------------------------------------------
// Family 3 — forwarding blocked.
// ---------------------------------------------------------------------------

func TestMemoptForwardingBlockedByPosition(t *testing.T) {
	g, _, _, _, st, load, _ := forwardBlockedPosFixture()
	runMemopt(t, g)
	assertKept(t, g, st, load)
	assertImpactBy(t, load, st)
	assertClean(t, g, st, load)
}

func TestMemoptForwardingBlockedByInterveningStore(t *testing.T) {
	g, _, _, _, s1, s2, load, _ := forwardBlockedUnequalFixture()
	runMemopt(t, g)
	// The most-recent store discipline: the unequal s2 IS the first store
	// behind the load, so nothing forwards — and s1 is not eliminable
	// either (the walk from s2 stops at nothing same-path).
	assertKept(t, g, s1, s2, load)
	assertImpactBy(t, load, s2)
	assertImpactBy(t, s2, s1)
	assertClean(t, g, s1, s2, load)
}

func TestMemoptForwardingBlockedNoStore(t *testing.T) {
	g, _, _, load, _ := forwardNoStoreFixture()
	runMemopt(t, g)
	assertKept(t, g, load)
	assertNoImpactBy(t, load)
	assertClean(t, g, load)
}

// ---------------------------------------------------------------------------
// Family 4 — two same-position loads forward; hinge-6 middle load spliced.
// ---------------------------------------------------------------------------

func TestMemoptTwoLoadsBothForwarded(t *testing.T) {
	g, _, _, v, st, l1, l2, end := twoLoadsFixture()

	runMemopt(t, g)
	assertGone(t, g, l1, l2)
	assertKept(t, g, st, end)
	ops := ir.OperandsOf(end)
	if len(ops) != 2 || ops[0] != v || ops[1] != v {
		t.Fatalf("both return data operands must see the stored value, got %v", ops)
	}
	assertImpactBy(t, end, st)
	assertClean(t, g, l1, l2, end)
}

func TestMemoptHingeMiddleLoadSpliced(t *testing.T) {
	g, _, _, _, s1, load, s2 := hingeSpliceFixture()

	runMemopt(t, g)
	assertGone(t, g, load)
	assertKept(t, g, s1, s2)
	// The splice hinge-6 documented as missing: s2's ordering edge now
	// names s1 directly — the load's edge was relinked, not dangled.
	assertImpactBy(t, s2, s1)
	assertNoImpactBy(t, s1)
	assertClean(t, g, load, s2)
}

// ---------------------------------------------------------------------------
// Family 5 — store-store elimination.
// ---------------------------------------------------------------------------

func TestMemoptStoreStoreSamePosition(t *testing.T) {
	g, _, _, v1, v2, s1, s2, end := storeStoreFixture()

	runMemopt(t, g)
	assertGone(t, g, s1)
	assertKept(t, g, v1, v2, s2, end)
	assertImpactBy(t, end, s2)
	assertClean(t, g, s1, s2)
}

func TestMemoptStoreStoreBlockedByLoad(t *testing.T) {
	g, _, _, _, s1, load, s2 := storeStoreBlockedByLoadFixture()
	runMemopt(t, g)
	assertKept(t, g, s1, load, s2)
	assertImpactBy(t, load, s1)
	assertImpactBy(t, s2, load)
	assertClean(t, g, s1, load, s2)
}

func TestMemoptStoreStoreDifferentPositionsKept(t *testing.T) {
	g, _, _, _, s1, s2 := storeStoreDiffPosFixture()
	runMemopt(t, g)
	assertKept(t, g, s1, s2)
	assertImpactBy(t, s2, s1)
	assertClean(t, g, s1, s2)
}

func TestMemoptStoreStoreWalkPastUnequal(t *testing.T) {
	g, _, _, _, s0, s1, s2 := walkPastFixture()

	runMemopt(t, g)
	assertGone(t, g, s0)
	assertKept(t, g, s1, s2)
	// s2's edge was spliced past the erased s0 onto the walked-past s1.
	assertImpactBy(t, s2, s1)
	assertNoImpactBy(t, s1)
	assertClean(t, g, s0, s2)
}

func TestMemoptStoreStoreCascade(t *testing.T) {
	g, _, _, s1, s2, s3 := cascadeFixture()

	runMemopt(t, g)
	assertGone(t, g, s1, s2)
	assertKept(t, g, s3)
	assertNoImpactBy(t, s3)
	assertClean(t, g, s1, s2, s3)
}

// ---------------------------------------------------------------------------
// Family 6 — dead stores.
// ---------------------------------------------------------------------------

func TestMemoptDeadTailCascadeErased(t *testing.T) {
	g, _, _, s1, s2 := deadTailFixture()

	runMemopt(t, g)
	assertGone(t, g, s1, s2)
	assertClean(t, g, s1, s2)
}

func TestMemoptDeadStoreDataUseRAUW(t *testing.T) {
	g, _, _, v, st, r := dataUseStoreFixture()

	runMemopt(t, g)
	assertGone(t, g, st)
	assertKept(t, g, v, r)
	if ops := ir.OperandsOf(r); len(ops) != 2 || ops[0] != v {
		t.Fatalf("the data use must see the store's value operand, got %v", ops)
	}
	assertClean(t, g, st, r)
}

func TestMemoptWiredTailNeverErased(t *testing.T) {
	g, _, _, _, st := wiredTailFixture()
	runMemopt(t, g)
	assertKept(t, g, st)
	if uses := ir.UsesOf(st); len(uses) != 1 {
		t.Fatalf("the Return-wired store must keep exactly its payload use, got %d", len(uses))
	}
	assertClean(t, g, st)
}

// ---------------------------------------------------------------------------
// Family 7 — the C1 eligibility guard: skip, never error.
// ---------------------------------------------------------------------------

func TestMemoptC1SkipsNonChainRegions(t *testing.T) {
	g, _, _, _, s1, s2, su := c1SkipFixture()
	before := len(g.AllocatedNodes())

	if err := (memopt.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("an ineligible region is legal-but-unoptimized: Run must return nil, got %v", err)
	}
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("a skipped region must be untouched: %d nodes before, %d after", before, after)
	}
	assertKept(t, g, s1, s2, su)
	assertImpactBy(t, s2, s1)
	assertClean(t, g, s1, s2, su)
}

// ---------------------------------------------------------------------------
// Family 8 — head removal: the chain-first erased store's successor edge is
// DROPPED via RemoveImpactBy (the case C3 exists for).
// ---------------------------------------------------------------------------

func TestMemoptHeadRemovalDropsEdge(t *testing.T) {
	g, _, _, _, _, _, s2, end := storeStoreFixture()

	runMemopt(t, g)
	// s1 was chain-first: s2's ordering edge must be GONE (not re-pointed
	// at a nil def, not left killed-in-place) — ir.Verify green proves the
	// drop rode the choke point.
	assertNoImpactBy(t, s2)
	assertImpactBy(t, end, s2)
	if uses := ir.UsesOf(s2); len(uses) != 1 {
		t.Fatalf("s2 must keep exactly the end node's payload use, got %d use(s)", len(uses))
	}
	assertClean(t, g, s2)
}

// ---------------------------------------------------------------------------
// Family 9 — idempotence, Manager interop, registry wiring.
// ---------------------------------------------------------------------------

func TestMemoptIdempotent(t *testing.T) {
	g, _, _, _, _, _, _ := forwardFixture()
	runMemopt(t, g)
	count := len(g.AllocatedNodes())

	if err := (memopt.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if got := len(g.AllocatedNodes()); got != count {
		t.Fatalf("second Run changed the graph: %d nodes before, %d after", count, got)
	}
	runMemVerify(t, g)
	assertClean(t, g)
}

func TestMemoptManagerInterop(t *testing.T) {
	g, _, _, v, st, load, _ := forwardFixture()
	dead := g.NewConst("ctor", ir.InternScalar(ir.TF32), 99) // DCE's business

	pm := opt.NewManager([]opt.Pass{dce.Pass{}, gvn.Pass{}, memopt.Pass{}, memopt.VerifyPass{}})
	pm.AddInstrumentation(opt.NewVerifyEach(nil))
	if err := pm.Run(g); err != nil {
		t.Fatalf("manager [dce, gvn, memopt, memopt-verify] pipeline: %v", err)
	}

	assertGone(t, g, load, dead)
	assertKept(t, g, v, st)
	assertClean(t, g)
}

func TestMemoptRegistryWiring(t *testing.T) {
	spec, ok := opt.Lookup("memopt")
	if !ok {
		t.Fatal("Lookup(memopt) not found — the import must self-register")
	}
	if spec.Priority != 112 {
		t.Errorf("memopt priority = %d, want 112", spec.Priority)
	}
	vspec, ok := opt.Lookup("memopt-verify")
	if !ok {
		t.Fatal("Lookup(memopt-verify) not found")
	}
	if vspec.Priority != 113 {
		t.Errorf("memopt-verify priority = %d, want 113", vspec.Priority)
	}
	if len(vspec.Requires) != 1 || vspec.Requires[0] != "memopt" {
		t.Errorf("memopt-verify Requires = %v, want [memopt]", vspec.Requires)
	}
	if req := (memopt.Pass{}).Requires(); len(req) != 1 || req[0] != "gvn" {
		t.Errorf("memopt Requires = %v, want [gvn]", req)
	}

	// Default pipeline order after the registrations: gvn-verify < memopt
	// < memopt-verify.
	pipe := opt.BuildPipeline()
	idx := make(map[string]int)
	for i, p := range pipe {
		idx[p.ID()] = i
	}
	for _, id := range []string{"gvn-verify", "memopt", "memopt-verify"} {
		if _, ok := idx[id]; !ok {
			t.Fatalf("BuildPipeline missing %q (got IDs %v)", id, idx)
		}
	}
	if !(idx["gvn-verify"] < idx["memopt"] && idx["memopt"] < idx["memopt-verify"]) {
		t.Fatalf("pipeline order wrong: gvn-verify=%d memopt=%d memopt-verify=%d",
			idx["gvn-verify"], idx["memopt"], idx["memopt-verify"])
	}
}

// ---------------------------------------------------------------------------
// Family 10 — the memopt-verify pass.
// ---------------------------------------------------------------------------

// TestMemoptVerifyGreenAfterRun: the verifier accepts every cleanly
// optimized fixture of families 2-8 (and the idempotence re-run).
func TestMemoptVerifyGreenAfterRun(t *testing.T) {
	fixtures := []struct {
		name string
		build func() *ir.Graph
	}{
		{"forward", func() *ir.Graph { g, _, _, _, _, _, _ := forwardFixture(); return g }},
		{"forward-blocked-pos", func() *ir.Graph { g, _, _, _, _, _, _ := forwardBlockedPosFixture(); return g }},
		{"forward-blocked-unequal", func() *ir.Graph { g, _, _, _, _, _, _, _ := forwardBlockedUnequalFixture(); return g }},
		{"forward-no-store", func() *ir.Graph { g, _, _, _, _ := forwardNoStoreFixture(); return g }},
		{"two-loads", func() *ir.Graph { g, _, _, _, _, _, _, _ := twoLoadsFixture(); return g }},
		{"hinge-splice", func() *ir.Graph { g, _, _, _, _, _, _ := hingeSpliceFixture(); return g }},
		{"store-store", func() *ir.Graph { g, _, _, _, _, _, _, _ := storeStoreFixture(); return g }},
		{"store-store-load-blocked", func() *ir.Graph { g, _, _, _, _, _, _ := storeStoreBlockedByLoadFixture(); return g }},
		{"store-store-diff-pos", func() *ir.Graph { g, _, _, _, _, _ := storeStoreDiffPosFixture(); return g }},
		{"walk-past", func() *ir.Graph { g, _, _, _, _, _, _ := walkPastFixture(); return g }},
		{"cascade", func() *ir.Graph { g, _, _, _, _, _ := cascadeFixture(); return g }},
		{"dead-tail", func() *ir.Graph { g, _, _, _, _ := deadTailFixture(); return g }},
		{"data-use", func() *ir.Graph { g, _, _, _, _, _ := dataUseStoreFixture(); return g }},
		{"wired-tail", func() *ir.Graph { g, _, _, _, _ := wiredTailFixture(); return g }},
		{"c1-skip", func() *ir.Graph { g, _, _, _, _, _, _ := c1SkipFixture(); return g }},
	}
	for _, f := range fixtures {
		g := f.build()
		runMemopt(t, g)
		runMemVerify(t, g)
		if vs := ir.Verify(g); len(vs) != 0 {
			t.Fatalf("%s: graph must verify: %v", f.name, vs)
		}
	}
}

// TestMemoptVerifyFlagsForwardableLoad: run ONLY the verifier over the
// un-optimized forwarding fixture — the planted forwardable load must be
// reported by node ID and kind.
func TestMemoptVerifyFlagsForwardableLoad(t *testing.T) {
	g, _, _, _, _, load, _ := forwardFixture()

	err := (memopt.VerifyPass{}).Run(nil, g)
	if err == nil {
		t.Fatal("the verifier must flag a planted forwardable load")
	}
	for _, want := range []string{
		fmt.Sprintf("node %d", load.BasePtr().ID()),
		load.KindOf().String(),
		"forwardable",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q must mention %q", err.Error(), want)
		}
	}
}

// TestMemoptVerifyRequiresMemoptFirst: the Manager rejects a reversed
// [memopt-verify, memopt] pipeline on Requires.
func TestMemoptVerifyRequiresMemoptFirst(t *testing.T) {
	g, _, _, _, _, _, _ := forwardFixture()
	pm := opt.NewManager([]opt.Pass{memopt.VerifyPass{}, memopt.Pass{}})
	err := pm.Run(g)
	if err == nil || !strings.Contains(err.Error(), "requires") {
		t.Fatalf("reversed pipeline must fail on Requires, got: %v", err)
	}
}

func TestMemoptVerifyGateNilGraph(t *testing.T) {
	err := memopt.VerifyPass{}.Run(nil, nil)
	if err == nil || err.Error() != "memopt-verify: nil graph" {
		t.Fatalf("Run(nil) error = %v, want %q", err, "memopt-verify: nil graph")
	}
}

func TestMemoptVerifyGateMaterializedRefused(t *testing.T) {
	g, _, _, _, _, _, _ := forwardFixture()
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := len(g.AllocatedNodes())

	err := memopt.VerifyPass{}.Run(nil, g)
	if err == nil || !strings.Contains(err.Error(), "materialized") {
		t.Fatalf("Run on a materialized graph must refuse, got: %v", err)
	}
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("a refused graph must be untouched: %d nodes before, %d after", before, after)
	}
}
