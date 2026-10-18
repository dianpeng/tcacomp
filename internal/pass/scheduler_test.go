package pass_test

import (
	"io"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
)

// The Scheduler pins (quick-261018-bl0 task 3, R5): Materialize's default
// schedule (D-7) is allocation order; the optimization-pass scheduler
// re-establishes dependency order with a STABLE topological sort (ties keep
// the pre-existing list order), then runs LICM over lowered for-loops and
// re-sorts. All reasoning is statement-level, built from the exported ir
// primitives only (StmtNodes/OperandsOf/PosetDeps/LambdaBodies) — the pass
// package never reaches into ir internals.

// stmtIDs collects the block's statement IDs in list order.
func stmtIDs(g *ir.Graph) []uint64 {
	var ids []uint64
	for s := g.Block().First(); s != nil; s = s.Next() {
		ids = append(ids, s.ID())
	}
	return ids
}

func equalUint64s(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestSchedulerRequiresMaterialized pins the pass's entry guard: scheduling
// is a Level-2 operation and refuses an unmaterialized graph with a
// deterministic message, never a partial schedule.
func TestSchedulerRequiresMaterialized(t *testing.T) {
	g := ir.NewGraph("unmaterialized")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 1))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	s := &pass.Scheduler{}
	if err := s.Run(nil, g); err == nil || !strings.Contains(err.Error(), "schedule: graph not materialized") {
		t.Fatalf("Run on an unmaterialized graph = %v, want the not-materialized guard", err)
	}
	if g.Block() != nil {
		t.Fatal("the refused run must not create a block")
	}
}

// TestSchedulerRestoresDependencyOrder pins the stable topological sort: the
// statement list is deliberately scrambled (the intrusive list supports O(1)
// relinking, so the test does the violence directly), and the scheduler must
// restore dependency order — store1 before store2 (the env chain ImpactBy
// edge), the end node last — while keeping every pinned statement ID stable.
func TestSchedulerRestoresDependencyOrder(t *testing.T) {
	g := ir.NewGraph("scrambled")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	env := ir.NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 1))
	st2 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 2))
	ret := g.NewReturnChained("ctor", env, []ir.Node{out}, st2)

	g.PinStatementID(st1, 100)
	g.PinStatementID(st2, 200)
	g.PinStatementID(ret, 300)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	bb := g.Block()
	s1 := bb.First() // [st1(100), st2(200), ret(300)]
	s2, s3 := s1.Next(), s1.Next().Next()
	if s1.ID() != 100 || s2.ID() != 200 || s3.ID() != 300 {
		t.Fatalf("precondition: materialized order = %v, want [100 200 300]", stmtIDs(g))
	}

	// Scramble: move the first statement to the tail → [200, 300, 100].
	bb.Remove(s1)
	bb.Append(s1)
	if !equalUint64s(stmtIDs(g), []uint64{200, 300, 100}) {
		t.Fatalf("precondition: scrambled order = %v, want [200 300 100]", stmtIDs(g))
	}

	s := &pass.Scheduler{}
	if err := s.Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := stmtIDs(g); !equalUint64s(got, []uint64{100, 200, 300}) {
		t.Fatalf("scheduler must restore dependency order with stable IDs, got %v", got)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("rescheduled graph must verify clean: %v", vs)
	}
}

// TestSchedulerCycleFailsClosed pins the fail-closed remainder: a statement
// dependency cycle (here a white-box self-loop wired through the exported
// MutContext.AddUse choke point) can never be topologically ordered, and the
// pass reports a cycle error instead of emitting a partial schedule.
func TestSchedulerCycleFailsClosed(t *testing.T) {
	g := ir.NewGraph("cycle")
	f32 := ir.InternScalar(ir.TF32)
	a := g.NewConst("ctor", f32, 1)
	b := g.NewConst("ctor", f32, 2)
	sum := g.NewBinOp("ctor", ir.BinOpAdd, a, b)
	g.NewReturn("ctor", sum)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := stmtIDs(g)

	ctx := &ir.MutContext{Pass: "test-cycle"}
	ctx.AddUse(sum, sum) // self-loop: sum now data-depends on itself

	s := &pass.Scheduler{}
	err := s.Run(nil, g)
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("Run with a dependency cycle = %v, want a cycle error", err)
	}
	if got := stmtIDs(g); !equalUint64s(got, before) {
		t.Fatalf("a failed run must leave the block untouched, got %v want %v", got, before)
	}
}

// licmFixture builds the shared LICM graph: one loop for(@x/64.0) whose body
// combines a loop-invariant const binop chain (c1 = 1+2, c2 = c1*3) with a
// varying biv-dependent add. Materialized, the block holds exactly the end
// node's statement (no GMEM/SMEM effect nodes exist).
func licmFixture(t *testing.T) (g *ir.Graph, c1, c2, v1 ir.Node) {
	t.Helper()
	g = ir.NewGraph("licm")
	f32 := ir.InternScalar(ir.TF32)
	x := g.NewInputParam("ctor", ir.RoleDim, "x", nil)

	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	sixtyfour := g.NewConst("ctor", f32, 64)
	iv := g.NewDimVar("ctor", x, zero, one)
	biv := g.NewBoundedDimVar("ctor", iv, sixtyfour)

	k1 := g.NewConst("ctor", f32, 1)
	k2 := g.NewConst("ctor", f32, 2)
	k3 := g.NewConst("ctor", f32, 3)
	c1 = g.NewBinOp("ctor", ir.BinOpAdd, k1, k2)
	c2 = g.NewBinOp("ctor", ir.BinOpMul, c1, k3)
	v1 = g.NewBinOp("ctor", ir.BinOpAdd, biv, c2) // varies: biv transitively depends on iv

	loop := g.NewForLoop("ctor", iv, sixtyfour, v1)
	g.NewReturn("ctor", loop)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if g.Block().First() == nil || g.Block().First().Next() != nil {
		t.Fatal("precondition: the fixture block holds exactly the end node's statement")
	}
	return g, c1, c2, v1
}

// TestSchedulerLICHappyPath pins the LICM core: the invariant chain c1, c2
// leaves the loop body and becomes statements immediately preceding the loop's
// statement (dependency order c1 before c2), the body now references them as
// %N, and the varying v1 stays inside.
func TestSchedulerLICHappyPath(t *testing.T) {
	g, c1, c2, v1 := licmFixture(t)

	s := &pass.Scheduler{}
	if err := s.Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}

	sC1, sC2 := c1.BasePtr().Statement(), c2.BasePtr().Statement()
	if sC1 == nil || sC2 == nil {
		t.Fatal("the invariant chain c1, c2 must gain statements of their own")
	}
	if sC1.Next() != sC2 {
		t.Fatal("c1's statement must immediately precede c2's (dependency order)")
	}
	if sC2.Next() != g.Block().Last() || g.Block().First() != sC1 {
		t.Fatal("the hoisted statements must sit immediately before the loop's statement")
	}
	if v1.BasePtr().Statement() != nil {
		t.Fatal("the varying v1 must not be hoisted")
	}
	if n := 0; true {
		for st := g.Block().First(); st != nil; st = st.Next() {
			n++
		}
		if n != 3 {
			t.Fatalf("block holds %d statements, want 3 (c1, c2, the loop's statement)", n)
		}
	}

	out := ir.Print(g)
	if !strings.Contains(out, "%0") {
		t.Fatalf("the body must reference the hoisted c1 as %%0, got:\n%s", out)
	}
	// The hoisted chain renders producer-first: c2 references c1 by number.
	if !strings.Contains(out, "%0 * 3.0") {
		t.Fatalf("c2's hoisted statement must reference c1 as %%0, got:\n%s", out)
	}
	if !strings.Contains(out, "for(") {
		t.Fatalf("the loop statement must survive scheduling, got:\n%s", out)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("scheduled graph must verify clean: %v", vs)
	}
}

// TestSchedulerLICMNegativeMemoryOps pins the never-hoist set: lowered SMEM
// stores and loads — and anything whose operand closure touches them — stay
// inside the loop even when their indices and co-operands look invariant.
func TestSchedulerLICMNegativeMemoryOps(t *testing.T) {
	t.Run("smem store stays", func(t *testing.T) {
		g := ir.NewGraph("licm-store")
		f32 := ir.InternScalar(ir.TF32)
		x := g.NewInputParam("ctor", ir.RoleDim, "x", nil)
		zero := g.NewConst("ctor", f32, 0)
		one := g.NewConst("ctor", f32, 1)
		sixtyfour := g.NewConst("ctor", f32, 64)
		iv := g.NewDimVar("ctor", x, zero, one)
		biv := g.NewBoundedDimVar("ctor", iv, sixtyfour)
		buf := g.NewMemNewSMem("ctor", sixtyfour)
		st := g.NewMemStoreSMem("ctor", biv, buf, biv) // value and index both vary
		loop := g.NewForLoop("ctor", iv, sixtyfour, st)
		g.NewReturn("ctor", loop)
		if err := g.Materialize(); err != nil {
			t.Fatalf("Materialize: %v", err)
		}
		before := stmtIDs(g)
		storeStmt := st.BasePtr().Statement()
		if storeStmt == nil {
			t.Fatal("precondition: the smem store roots its own materialized statement")
		}

		if err := (&pass.Scheduler{}).Run(nil, g); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if st.BasePtr().Statement() != storeStmt {
			t.Fatal("the smem store must keep its original statement — never hoisted")
		}
		if got := stmtIDs(g); !equalUint64s(got, before) {
			t.Fatalf("nothing may hoist out of an all-varying/all-memory body, got %v want %v", got, before)
		}
	})

	t.Run("smem load stays", func(t *testing.T) {
		g := ir.NewGraph("licm-load")
		f32 := ir.InternScalar(ir.TF32)
		x := g.NewInputParam("ctor", ir.RoleDim, "x", nil)
		zero := g.NewConst("ctor", f32, 0)
		one := g.NewConst("ctor", f32, 1)
		sixtyfour := g.NewConst("ctor", f32, 64)
		iv := g.NewDimVar("ctor", x, zero, one)
		biv := g.NewBoundedDimVar("ctor", iv, sixtyfour)
		buf := g.NewMemNewSMem("ctor", sixtyfour)
		load := g.NewMemLoadSMem("ctor", buf, biv) // index varies
		body := g.NewBinOp("ctor", ir.BinOpMul, load, g.NewConst("ctor", f32, 2))
		loop := g.NewForLoop("ctor", iv, sixtyfour, body)
		g.NewReturn("ctor", loop)
		if err := g.Materialize(); err != nil {
			t.Fatalf("Materialize: %v", err)
		}
		before := stmtIDs(g)

		if err := (&pass.Scheduler{}).Run(nil, g); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if load.BasePtr().Statement() != nil {
			t.Fatal("the smem load must not gain a statement — never hoisted")
		}
		if got := stmtIDs(g); !equalUint64s(got, before) {
			t.Fatalf("nothing may hoist: the load's body chain varies, got %v want %v", got, before)
		}
	})
}

// TestSchedulerLICMNestedHoistsToOutermost pins the nest rule: a candidate
// inside an inner loop is hoisted before the outermost enclosing loop's
// statement only when it is invariant with respect to EVERY loop on the
// enclosing path; anything varying in an outer loop stays put.
func TestSchedulerLICMNestedHoistsToOutermost(t *testing.T) {
	g := ir.NewGraph("licm-nested")
	f32 := ir.InternScalar(ir.TF32)
	x := g.NewInputParam("ctor", ir.RoleDim, "x", nil)
	y := g.NewInputParam("ctor", ir.RoleDim, "y", nil)

	ivx := g.NewDimVar("ctor", x, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
	bvx := g.NewBoundedDimVar("ctor", ivx, g.NewConst("ctor", f32, 64))
	ivy := g.NewDimVar("ctor", y, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
	bvy := g.NewBoundedDimVar("ctor", ivy, g.NewConst("ctor", f32, 32))

	c1 := g.NewBinOp("ctor", ir.BinOpAdd, g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 2)) // invariant in both
	ov := g.NewBinOp("ctor", ir.BinOpAdd, bvx, c1)                                                // varies in the OUTER loop
	ivv := g.NewBinOp("ctor", ir.BinOpAdd, bvy, c1)                                               // varies in the INNER loop
	innerBody := g.NewBinOp("ctor", ir.BinOpAdd, ivv, ov)
	inner := g.NewForLoop("ctor", ivy, g.NewConst("ctor", f32, 32), innerBody)
	outer := g.NewForLoop("ctor", ivx, g.NewConst("ctor", f32, 64), inner)
	g.NewReturn("ctor", outer)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	loopStmt := g.Block().First()
	if loopStmt == nil || loopStmt.Next() != nil {
		t.Fatal("precondition: one statement (the end node's) holds the whole nest")
	}

	if err := (&pass.Scheduler{}).Run(nil, g); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sC1 := c1.BasePtr().Statement()
	if sC1 == nil {
		t.Fatal("the fully-invariant c1 must hoist")
	}
	if g.Block().First() != sC1 || sC1.Next() != loopStmt {
		t.Fatal("c1 must hoist to immediately before the outermost loop's statement")
	}
	if ov.BasePtr().Statement() != nil || ivv.BasePtr().Statement() != nil {
		t.Fatal("nodes varying in the outer/inner loops must stay inside the nest")
	}
}

// TestSchedulerIdempotent pins the pass's fixpoint: a second Run must change
// nothing — no new statements, byte-identical print. Passes in the pipeline
// must be safe under manager-driven re-runs.
func TestSchedulerIdempotent(t *testing.T) {
	g, _, _, _ := licmFixture(t)
	s := &pass.Scheduler{}
	if err := s.Run(nil, g); err != nil {
		t.Fatalf("first Run: %v", err)
	}
	first := ir.Print(g)
	count := 0
	for st := g.Block().First(); st != nil; st = st.Next() {
		count++
	}

	if err := s.Run(nil, g); err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if again := ir.Print(g); again != first {
		t.Fatalf("second Run changed the print:\nfirst:\n%s\nagain:\n%s", first, again)
	}
	n := 0
	for st := g.Block().First(); st != nil; st = st.Next() {
		n++
	}
	if n != count {
		t.Fatalf("second Run created statements: %d, want %d", n, count)
	}
}

// TestSchedulerViaManager pins pipeline integration (PASS-01/IR-07): the
// scheduler is an ordinary pass — the manager checks its produced stage's
// legality window after Run, and VerifyEach re-verifies the graph after the
// pass without a single violation.
func TestSchedulerViaManager(t *testing.T) {
	g, c1, _, _ := licmFixture(t)
	pm := pass.NewManager([]pass.Pass{&pass.Scheduler{}})
	pm.AddInstrumentation(pass.NewVerifyEach(io.Discard))
	if err := pm.Run(g); err != nil {
		t.Fatalf("pipeline Run: %v", err)
	}
	if c1.BasePtr().Statement() == nil {
		t.Fatal("the pipeline run must perform the LICM hoist")
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("post-pipeline graph must verify clean: %v", vs)
	}
}
