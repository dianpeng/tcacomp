package pipeline_test

// pipeline_test.go pins the orchestrator's composition (fix2 §4 step 1): the
// explicit windows reproduce — order for order — the registry's default
// pipeline over the same pass set, resolved through the registered factories
// (fail-closed on unknown IDs), so the orchestration refactor is behavior-
// identical to opt.BuildPipeline at this commit.

import (
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/pipeline"
)

// passIDs renders a pass list's IDs in order.
func passIDs(passes []opt.Pass) []string {
	ids := make([]string, len(passes))
	for i, p := range passes {
		ids[i] = p.ID()
	}
	return ids
}

// TestCompositionMatchesRegistry pins fix2 §4 step 1: HIR ++ LOWER is
// order-for-order IDENTICAL to opt.BuildPipeline()'s ID sequence — the
// orchestration refactor adds wiring only, zero composition drift.
func TestCompositionMatchesRegistry(t *testing.T) {
	got := passIDs(append(pipeline.HIRPasses(), pipeline.LOWERPasses()...))
	want := passIDs(opt.BuildPipeline())
	if len(got) != len(want) {
		t.Fatalf("pipeline composition has %d passes, registry default has %d:\npipeline: %v\nregistry: %v",
			len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("composition diverges at position %d: pipeline %q, registry %q\npipeline: %v\nregistry: %v",
				i, got[i], want[i], got, want)
		}
	}
}

// TestWindowsAreFreshInstances pins R-E's mechanism at the wiring level: the
// factories ARE re-invokable — two calls yield independent instances of the
// same pass (the registry hands out fresh instances, never shared state).
func TestWindowsAreFreshInstances(t *testing.T) {
	a, b := pipeline.HIRPasses(), pipeline.HIRPasses()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("twin HIR windows differ in shape: %d vs %d", len(a), len(b))
	}
	for i := range a {
		// Interface values over distinct pointers differ; the registry's
		// Factory contract guarantees fresh construction per call.
		if &a[i] == &b[i] {
			t.Fatalf("HIR window %d shares storage across calls", i)
		}
	}
}

// TestRunRefusesNilGraph pins the entry's nil-gate delegation.
func TestRunRefusesNilGraph(t *testing.T) {
	if err := pipeline.Run(nil); err == nil {
		t.Fatal("Run(nil) must fail closed")
	}
}

// TestTailFillLivenessWiring pins fix2 D-e's tail composition at the WIRING
// level: the fill-liveness fixture shape (lifted from memopt's direct-
// invocation family-11 fixture — a FragStore with no live buffer loads plus
// its source GMEM load) run through the REAL tail composition
// (TailPasses() under one opt.Manager, with the registry's gvn instance in
// front mirroring how Run's HIR window satisfies the tail memopt's
// Requires ["gvn"]) erases the unconsumed fill AND its source load
// in-pipeline — memopt rule (d) reachable through the pipeline for the
// first time. The consumed control fill survives, and the AST-item-roots
// dce keeps the buffer ALLOCATIONS (they are items — D-d's liveness truth)
// while sweeping the erased nodes' items.
func TestTailFillLivenessWiring(t *testing.T) {
	g := ir.NewGraph("tail-fill-wiring")
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	lhs2 := g.NewInputParam("ctor", ir.RoleTensor, "lhs2", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	// The unconsumed fill: buffer B's only store, no buffer load anywhere.
	buf := g.NewFragNew("ctor", m)
	env := ir.NewEnv()
	srcLoad := g.NewMemLoadChained("ctor", env, lhs, []ir.Node{m})
	fillStore := g.NewFragStore("ctor", srcLoad, buf, m)

	// The tensor chain's successor of srcLoad — the removal's splice work.
	env2 := ir.NewEnv()
	env2.NoteStore(lhs, srcLoad)
	succLoad := g.NewMemLoadChained("ctor", env2, lhs, []ir.Node{m})

	// The control fill: buffer C holds a store AND a live reader.
	ctlBuf := g.NewFragNew("ctor", m)
	ctlLoad := g.NewMemLoadChained("ctor", env, lhs2, []ir.Node{m})
	ctlStore := g.NewFragStore("ctor", ctlLoad, ctlBuf, m)
	bufEnv := ir.NewEnv()
	bufEnv.NoteStore(ctlBuf, ctlStore)
	ctlAccess := g.NewFragLoadChained("ctor", bufEnv, ctlBuf, m)
	end := g.NewReturnChained("ctor", env, []ir.Node{out}, succLoad, ctlAccess)

	// The scheduled shape the tail consumes: a Prototype with the params, a
	// Body Group holding the statements (allocations and accesses are SN-0
	// in-kernel declarations/statements — the schedule's D-4 discipline).
	proto := ir.NewASTPrototype()
	for _, p := range g.Params() {
		proto.AppendItem(ir.ASTItem{Node: p})
		p.BasePtr().SetSched(ir.PrototypeSN, proto)
	}
	body := ir.NewASTGroup()
	for _, n := range []ir.Node{buf, ctlBuf, srcLoad, fillStore, succLoad, ctlLoad, ctlStore, ctlAccess, end} {
		body.AppendItem(ir.ASTItem{Node: n})
		n.BasePtr().SetSched(0, body)
	}
	if err := g.AttachAST(&ir.ASTKernel{Prototype: proto, Body: body}); err != nil {
		t.Fatalf("fixture: attach AST: %v", err)
	}

	// The REAL tail composition under one manager: [gvn (the registry
	// instance, mirroring Run's HIR window satisfying the tail memopt's
	// Requires), memopt (factory reuse), dce (AST-item roots)].
	gvnSpec, ok := opt.Lookup("gvn")
	if !ok {
		t.Fatal("fixture: gvn must be registered (pipeline blank-imports opt/all)")
	}
	pm := opt.NewManager(append([]opt.Pass{gvnSpec.Factory()}, pipeline.TailPasses()...))
	if err := pm.Run(g); err != nil {
		t.Fatalf("tail pipeline: %v", err)
	}

	// Rule (d) fired in-pipeline: the fill AND its source load are erased.
	for name, n := range map[string]ir.Node{"fill store": fillStore, "source load": srcLoad} {
		if !n.BasePtr().Erased() {
			t.Fatalf("the %s must be erased by the tail composition", name)
		}
	}
	// The control fill and the allocations survive (items + live readers).
	for name, n := range map[string]ir.Node{
		"buffer B alloc": buf, "buffer C alloc": ctlBuf, "control fill": ctlStore,
		"control source": ctlLoad, "control reader": ctlAccess, "chain successor": succLoad,
	} {
		if n.BasePtr().Erased() {
			t.Fatalf("the %s must survive the tail composition", name)
		}
	}
	// The dce item sweep collected the erased nodes' items.
	for _, it := range g.AST().Body.Items() {
		if it.Node != nil && it.Node.BasePtr().Erased() {
			t.Fatalf("AST item still references the erased node %d (%s)", it.Node.BasePtr().ID(), it.Node.KindOf())
		}
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the tail: %v", vs)
	}
}
