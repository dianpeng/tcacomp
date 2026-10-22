package sideeffect_test

import (
	"testing"

	// The top-level analysis package stays imported for the Analysis-
	// interface satisfaction pin in TestName (opt.Cache compatibility).
	"tcacomp/internal/analysis"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
)

// sideeffect_test.go pins the accumulated side-effect analysis
// (quick-261018-nga): base verdicts as amended by decision A (every
// FamilyMemory node effectful, nil ImpactBy included; implicit control flow
// per [2.3]) and the order-sensitive-only taint closure as amended by C
// (pure kinds NEVER propagate — address/index math over iter values stays
// GVN-eligible).

// fixture builds a kernel-shaped scaffold: dim m, tensor out, scalar alpha,
// and a const.
func fixture() (g *ir.Graph, m, out, alpha, c1 ir.Node) {
	g = ir.NewGraph("se")
	f32 := ir.InternScalar(ir.TF32)
	m = g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	alpha = g.NewInputParam("ctor", ir.RoleScalar, "alpha", f32)
	c1 = g.NewConst("ctor", f32, 1)
	return g, m, out, alpha, c1
}

// iterOver builds a normalized iter over dim bounded by max (optionally
// parented): DimVar(dim,0,1) -> BoundedDimVar(dv,max) -> Iter(biv,parent).
func iterOver(g *ir.Graph, dim, max, parent ir.Node) ir.Node {
	f32 := ir.InternScalar(ir.TF32)
	dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
	return g.NewIter("ctor", g.NewBoundedDimVar("ctor", dv, max), parent)
}

// TestName pins the Analysis-name contract (opt.Cache compatibility).
func TestName(t *testing.T) {
	var zero sideeffect.SideEffects
	if got := zero.Name(); got != "side-effect" {
		t.Fatalf("Name() = %q, want %q", got, "side-effect")
	}
	var a analysis.Analysis = mustAnalyze(t, ir.NewGraph("empty"))
	if a.Name() != "side-effect" {
		t.Fatalf("SideEffects must satisfy analysis.Analysis, Name() = %q", a.Name())
	}
}

// mustAnalyze runs Analyze and fails the test on a nil graph panic guard.
func mustAnalyze(t *testing.T, g *ir.Graph) sideeffect.SideEffects {
	t.Helper()
	se := sideeffect.Analyze(g)
	if !se.HasSideEffect(nil) {
		t.Fatal("a nil node must query effectful (conservative)")
	}
	return se
}

// TestMemoryFamilyEffectfulWithNilImpactBy pins amendment A: every one of
// the NINE FamilyMemory kinds — loads, stores, and allocations alike, all
// built UNCHAINED (nil ImpactBy) — is effectful.
func TestMemoryFamilyEffectfulWithNilImpactBy(t *testing.T) {
	g, m, out, _, c1 := fixture()
	f32 := ir.InternScalar(ir.TF32)

	smem := g.NewMemNewSMem("ctor", c1)
	frag := g.NewFragNew("ctor", c1)
	nodes := map[string]ir.Node{
		"memload":      g.NewMemLoad("ctor", out, []ir.Node{m}),
		"memstore":     g.NewMemStore("ctor", out, []ir.Node{m}, c1),
		"memnew":       g.NewMemNew("ctor", c1),
		"memloadsmem":  g.NewMemLoadSMem("ctor", smem),
		"memstoresmem": g.NewMemStoreSMem("ctor", c1, smem),
		"memnewsmem":   smem,
		"fragload":     g.NewFragLoad("ctor", frag),
		"fragstore":    g.NewFragStore("ctor", c1, frag),
		"fragnew":      frag,
	}
	se := mustAnalyze(t, g)
	for name, n := range nodes {
		if got := len(ir.ImpactBy(n)); got != 0 {
			t.Fatalf("%s fixture must be unchained (nil ImpactBy), got %d edges", name, got)
		}
		if !se.HasSideEffect(n) {
			t.Fatalf("%s must be effectful with nil ImpactBy (amendment A)", name)
		}
	}
	_ = f32
}

// TestChainedAccessEffectful pins rule [2.1]: an env-chained access —
// ImpactBy present — is effectful (subsumed by A for today's bearers; the
// hook stays for future effectBearers).
func TestChainedAccessEffectful(t *testing.T) {
	g, m, out, _, c1 := fixture()
	env := ir.NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	st2 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	if len(ir.ImpactBy(st2)) == 0 {
		t.Fatal("fixture setup: the second chained store must carry ImpactBy")
	}
	se := mustAnalyze(t, g)
	if !se.HasSideEffect(st1) || !se.HasSideEffect(st2) {
		t.Fatal("env-chained accesses must be effectful ([2.1])")
	}
}

// TestImplicitControlFlowEffectful pins rule [2.3] minus the Iter:
// IterCluster, ForLoop, and Return stay effectful (the conservative
// closure of implicit control flow). The bare Iter was REMOVED from the
// base verdict per doc/optana/loop_fusion.md [1] ("The Iter node is
// itself side effect free") — TestIterIsPure pins that inversion.
func TestImplicitControlFlowEffectful(t *testing.T) {
	g, m, _, _, c1 := fixture()
	f32 := ir.InternScalar(ir.TF32)
	step := g.NewConst("ctor", f32, 1)
	cluster := g.NewIterCluster("ctor", []ir.Node{m}, c1)
	dv := g.NewDimVar("ctor", m, c1, step)
	loop := g.NewForLoop("ctor", dv, c1, c1)
	ret := g.NewReturn("ctor", c1)

	se := mustAnalyze(t, g)
	for name, n := range map[string]ir.Node{
		"itercluster": cluster,
		"forloop":     loop,
		"return":      ret,
	} {
		if !se.HasSideEffect(n) {
			t.Fatalf("%s must be effectful ([2.3] implicit control flow)", name)
		}
	}
}

// TestRuntimeFamilyBaseVerdicts pins the P-2 lockstep mirror
// (quick-261022-l1o): baseVerdict marks every RuntimeCallMethod effectful
// (the ir.EffectOf arm's exact twin — an ordered method call on a stateful
// reducer object is never a GVN candidate), while every other Runtime/
// Collective kind (the constructions, the getattr, the renamed queries)
// stays pure — the objects are immutable, and no pass runs after
// final-lower anyway, so value-identical constructions can never be
// CSE-merged.
func TestRuntimeFamilyBaseVerdicts(t *testing.T) {
	g, _, _, _, _ := fixture()
	ki := g.NewRuntimeKernelInfo("ctor")
	gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
	init := g.NewConst("ctor", ir.InternScalar(ir.TF32), 0)
	acc := g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := g.NewBinOp("ctor", ir.BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	lam := g.NewLambdaNode("ctor", []ir.Node{acc, now}, comb)
	red := g.NewRuntimeWarpReducer("ctor", init, lam)
	getattr := g.NewRuntimeGetAttribute("ctor", "index", gm, g.NewConst("ctor", ir.InternScalar(ir.TU32), 0))
	gq := g.NewRuntimeGridQuery("ctor", "index", g.NewConst("ctor", ir.InternScalar(ir.TU32), 0))
	callInit := g.NewRuntimeCallMethod("ctor", "Init", red)
	callReduce := g.NewRuntimeCallMethod("ctor", "Reduce", red, init)
	callCombine := g.NewRuntimeCallMethod("ctor", "Combine", red)

	se := mustAnalyze(t, g)
	for name, n := range map[string]ir.Node{
		"kernelinfo": ki, "gridmapping": gm, "reducer": red,
		"getattr": getattr, "gridquery": gq,
	} {
		if se.HasSideEffect(n) {
			t.Fatalf("%s must stay PURE in the accumulated analysis (P-2 mirror)", name)
		}
	}
	for name, n := range map[string]ir.Node{
		"init call":    callInit,
		"reduce call":  callReduce,
		"combine call": callCombine,
	} {
		if !se.HasSideEffect(n) {
			t.Fatalf("%s must be EFFECTFUL in the accumulated analysis (the baseVerdict mirror of ir.EffectOf's arm)", name)
		}
	}
}

// TestIterIsPure pins doc/optana/loop_fusion.md [1] ("The Iter node is
// itself side effect free, though it is a control flow node"): in the
// (biv, parent?) formation both a nest ROOT and a PARENTED iter query
// PURE — their operands are BIV value objects and (at most) another pure
// iter — which is what makes identical loop headers GVN-fusable.
func TestIterIsPure(t *testing.T) {
	g, m, _, _, c1 := fixture()
	bare := iterOver(g, m, c1, nil)
	child := iterOver(g, m, c1, bare)

	se := mustAnalyze(t, g)
	if se.HasSideEffect(bare) {
		t.Fatal("a bare root Iter must be side-effect free (doc/optana/loop_fusion.md [1])")
	}
	if se.HasSideEffect(child) {
		t.Fatal("a parented Iter must be side-effect free (doc/optana/loop_fusion.md [1])")
	}
}

// TestIterStaysPureOverValueObjects pins the value-object taint quarantine
// of the (biv, parent?) formation: a well-formed iter is
// DimVar -> BoundedDimVar -> Iter, all FamilyAbstract VALUE OBJECTS — pure
// and non-propagating — so even a BIV whose max is an accumulated-effectful
// MemLoad keeps the iter (and the biv and the dv) PURE. The taint cannot
// cross the value-object boundary, and identity keying keeps BIVs over
// distinct load defs distinct, so this is GVN-sound. The OLD taint path —
// an iter tainting through an effectful init operand — is structurally
// unreachable now: no raw value operand of an iter exists anymore.
func TestIterStaysPureOverValueObjects(t *testing.T) {
	g, m, out, _, c1 := fixture()
	f32 := ir.InternScalar(ir.TF32)
	step := g.NewConst("ctor", f32, 1)
	load := g.NewMemLoad("ctor", out, []ir.Node{m})
	dv := g.NewDimVar("ctor", m, c1, step)
	biv := g.NewBoundedDimVar("ctor", dv, load) // max = the load — structurally legal; the analysis does not type-check
	tainted := g.NewIter("ctor", biv, nil)

	se := mustAnalyze(t, g)
	if !se.HasSideEffect(load) {
		t.Fatal("fixture setup: the MemLoad must be effectful")
	}
	if se.HasSideEffect(dv) || se.HasSideEffect(biv) || se.HasSideEffect(tainted) {
		t.Fatal("the value-object chain (dv/biv/iter) must stay PURE over an effectful max (FamilyAbstract quarantine)")
	}
}

// TestPureKindsNeverPropagate pins amendment C, the GVN-eligibility hinge:
// PURE kinds never taint — a binop over an ITER value stays pure; a
// cast/select/call chain over such math stays pure; a Map piping a
// MemLoad operand stays pure; a Map whose lambda body references iter
// values stays pure (body children do not taint a pure bearer — and since
// the doc/optana/loop_fusion.md reclassification the iter is itself
// pure, so the effectful-content pin rides the piped load).
func TestPureKindsNeverPropagate(t *testing.T) {
	g, m, out, alpha, c1 := fixture()
	f32 := ir.InternScalar(ir.TF32)

	iter := iterOver(g, m, c1, nil)
	load := g.NewMemLoad("ctor", out, []ir.Node{m})

	bin := g.NewBinOp("ctor", ir.BinOpMul, iter, c1) // i*2 over the (pure) iter
	cast := g.NewCast("ctor", f32, bin)              // cast over the chain
	sel := g.NewSelect("ctor", bin, cast, c1)        // select over the chain
	call := g.NewCall("ctor", "exp", sel)            // call over the chain
	lp := g.NewLambdaInputParam("ctor", "l", f32)
	body := g.NewBinOp("ctor", ir.BinOpAdd, lp, iter) // body references the iter value (pure since the reclassification)
	mapOverLoad := g.NewMap("ctor", []ir.Node{load}, ir.NewLambda([]ir.Node{lp}, body), []ir.Node{m})
	mapPureBody := g.NewMap("ctor", []ir.Node{c1}, ir.NewLambda([]ir.Node{lp}, body), []ir.Node{m})

	se := mustAnalyze(t, g)
	if se.HasSideEffect(iter) || !se.HasSideEffect(load) {
		t.Fatal("fixture setup: the bare iter must be pure, the load effectful")
	}
	for name, n := range map[string]ir.Node{
		"binop over iter":    bin,
		"cast chain":         cast,
		"select chain":       sel,
		"call chain":         call,
		"map piping load":    mapOverLoad,
		"map with iter body": mapPureBody,
	} {
		if se.HasSideEffect(n) {
			t.Fatalf("%s must stay PURE (amendment C: pure kinds never propagate)", name)
		}
	}
	// The scalar param and consts stay pure, of course.
	if se.HasSideEffect(alpha) || se.HasSideEffect(c1) {
		t.Fatal("params and consts must stay pure")
	}
}

// TestOrderSensitivePropagation pins the propagation MACHINERY: an
// order-sensitive kind with an effectful child reports effectful, and the
// result is a deterministic fixpoint. Since the doc/optana/loop_fusion.md
// reclassification the order-sensitive kinds (FamilyLoop) are NOT
// base-effectful, so the clause is LIVE for FamilyLoop-with-effectful-
// child — but in the (biv, parent?) formation no WELL-FORMED iter feeds
// it anymore (its operands are pure value objects), so accumulated ==
// base-set still holds on this fixture.
func TestOrderSensitivePropagation(t *testing.T) {
	g, m, out, _, c1 := fixture()

	iter := iterOver(g, m, c1, nil)
	dv := g.NewDimVar("ctor", m, c1, c1)
	loop := g.NewForLoop("ctor", dv, c1, c1) // loop kind with effectful-child machinery in scope
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	_ = iter

	se1, se2 := sideeffect.Analyze(g), sideeffect.Analyze(g)
	nodes := g.AllocatedNodes()
	for _, n := range nodes {
		if se1.HasSideEffect(n) != se2.HasSideEffect(n) {
			t.Fatalf("two Analyze runs disagree on node %d (%s)", n.BasePtr().ID(), n.KindOf())
		}
	}
	if !se1.HasSideEffect(loop) || !se1.HasSideEffect(end) || !se1.HasSideEffect(st) {
		t.Fatal("loop/end/store must be effectful")
	}
	// Determinism + base-set equality on the fixture: no Iter here has an
	// effectful child, so the (now live) propagation clause adds nothing
	// and accumulated == base set.
	for _, n := range nodes {
		want := len(ir.ImpactBy(n)) > 0 ||
			n.KindOf().Family() == ir.FamilyMemory ||
			n.KindOf() == ir.KindIterCluster ||
			n.KindOf() == ir.KindForLoop || n.KindOf() == ir.KindReturn
		if se1.HasSideEffect(n) != want {
			t.Fatalf("node %d (%s): accumulated verdict %v differs from the base set %v (no iter on this fixture has an effectful child)",
				n.BasePtr().ID(), n.KindOf(), se1.HasSideEffect(n), want)
		}
	}
}

// TestConservativeQueries pins the fail-closed query face: erased nodes and
// nil nodes report effectful; Analyze(nil) yields an all-conservative
// result (every query true).
func TestConservativeQueries(t *testing.T) {
	g, m, out, _, c1 := fixture()
	dead := g.NewBinOp("ctor", ir.BinOpAdd, c1, c1)
	dead2 := g.NewConst("ctor", ir.InternScalar(ir.TF32), 9)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	_ = end

	se := mustAnalyze(t, g)
	if se.HasSideEffect(dead) {
		t.Fatal("setup guard: a live pure binop must query pure")
	}
	mctx := &ir.MutContext{Pass: "t"}
	if err := mctx.Erase(dead); err != nil {
		t.Fatalf("Erase: %v", err)
	}
	if !se.HasSideEffect(dead) {
		t.Fatal("an ERASED node must query effectful (conservative)")
	}
	if !se.HasSideEffect(nil) {
		t.Fatal("a nil node must query effectful (conservative)")
	}
	_ = dead2

	nilSE := sideeffect.Analyze(nil)
	for _, n := range []ir.Node{c1, st, m} {
		if !nilSE.HasSideEffect(n) {
			t.Fatal("Analyze(nil) must be all-conservative (every query true)")
		}
	}
}

// TestAnalyzeDoesNotMutate pins the analysis-package contract: Analyze is
// read-only — the node set and use counts are identical before and after.
func TestAnalyzeDoesNotMutate(t *testing.T) {
	g, m, out, _, c1 := fixture()
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, c1)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	before := len(g.AllocatedNodes())
	usesBefore := len(ir.UsesOf(c1))
	sideeffect.Analyze(g)
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("Analyze must not erase/allocate: %d before, %d after", before, after)
	}
	if got := len(ir.UsesOf(c1)); got != usesBefore {
		t.Fatalf("Analyze must not rewire uses: %d before, %d after", usesBefore, got)
	}
	for _, n := range g.AllocatedNodes() {
		if n.BasePtr().Erased() {
			t.Fatal("Analyze must not tombstone nodes")
		}
	}
}
