package loopforest_test

// loopforest_lir_test.go pins the generic analysis over the LIR forloop
// dialect (quick-261022-f5o D-03): a forloop chain links by BODY-OPERAND
// CONTAINMENT (the BodyRoot rule), a mixed iter+forloop graph links
// coherently (iters via ParentEdge, forloops via containment), and — the
// D-04 probe — an ImpactBy completion edge to a FOREIGN nest's root loop
// can never create a forest parent link (repointChains' shape: completion
// edges order, membership edges nest).

import (
	"testing"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// forOver builds a forloop over a fresh normalized dimvar on dim, bounded
// by max, with body as ops[2] — the scheduler tests' construction idiom.
func forOver(g *ir.Graph, dim, max, body ir.Node) ir.Node {
	u32 := ir.InternScalar(ir.TU32)
	dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	biv := g.NewBoundedDimVar("ctor", dv, max)
	if body == nil {
		body = g.NewConst("ctor", ir.InternScalar(ir.TF32), 0)
	}
	return g.NewForLoop("ctor", dv, biv, body)
}

// TestLIRForLoopChain pins the containment link: an inner forloop inside
// the outer's ops[2] subtree parents at the outer; depths are 0-based
// (D-02); an unrelated second forloop stays its own root in allocation
// order.
func TestLIRForLoopChain(t *testing.T) {
	g, dims, out := dimFixture("lir-chain", 2)
	u32 := ir.InternScalar(ir.TU32)
	inner := forOver(g, dims[1], g.NewConst("ctor", u32, 8), nil)
	outer := forOver(g, dims[0], g.NewConst("ctor", u32, 64), inner) // body CONTAINS inner
	other := forOver(g, dims[1], g.NewConst("ctor", u32, 8), nil)    // separate nest
	f32 := ir.InternScalar(ir.TF32)
	scaffold(g, dims[0], out, f32zero(g, f32))
	_ = other

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("the contained chain plus the separate forloop must yield 2 roots, got %d", len(roots))
	}
	if roots[0].Iter != outer || roots[1].Iter != other {
		t.Fatalf("roots must be [outer, other] in allocation order, got [%d, %d]",
			roots[0].Iter.BasePtr().ID(), roots[1].Iter.BasePtr().ID())
	}
	if roots[0].Depth != 0 {
		t.Fatalf("outer root depth = %d, want 0 (the 0-based convention, D-02)", roots[0].Depth)
	}
	if len(roots[0].Children) != 1 || roots[0].Children[0].Iter != inner {
		t.Fatal("inner must be outer's only child (body-operand containment)")
	}
	if roots[0].Children[0].Depth != 1 {
		t.Fatalf("inner depth = %d, want 1", roots[0].Children[0].Depth)
	}
	if roots[0].Children[0].Parent != roots[0] {
		t.Fatal("inner's Parent must be outer's tree")
	}
}

// TestMixedDialectLinking pins coherence on a hypothetical mixed graph: an
// iter inside a forloop's body subtree links under the forloop via
// containment, while a free iter stays a root and links nothing.
func TestMixedDialectLinking(t *testing.T) {
	g, dims, out := dimFixture("mixed", 2)
	u32 := ir.InternScalar(ir.TU32)
	inside := iterOver(g, dims[1], g.NewConst("ctor", u32, 8), nil) // an iter IN the forloop's body
	loop := forOver(g, dims[0], g.NewConst("ctor", u32, 64), inside)
	free := iterOver(g, dims[1], g.NewConst("ctor", u32, 8), nil) // root iter, no edges
	scaffold(g, dims[0], out, f32zero(g, ir.InternScalar(ir.TF32)))

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("the forloop and the free iter must yield 2 roots, got %d", len(roots))
	}
	if roots[0].Iter != loop || roots[1].Iter != free {
		t.Fatal("roots must be [loop, freeIter] in allocation order")
	}
	if len(roots[0].Children) != 1 || roots[0].Children[0].Iter != inside {
		t.Fatal("the body-contained iter must link under the forloop (containment serves both dialects)")
	}
	if it, ok := f.NodeOf(inside); !ok || it.Parent != roots[0] {
		t.Fatal("the contained iter's tree must parent at the forloop's tree")
	}
}

// TestImpactByNeverNests pins D-04 — THE reason the spec interface exists:
// an effect edge (ImpactBy) from inside nest B's body at nest A's ROOT
// loop node is a COMPLETION edge ("after the whole loop"), never a
// membership edge. The forest must stay two-rooted even though loop A is
// uniform-reachable from loop B's body.
func TestImpactByNeverNests(t *testing.T) {
	g, dims, out := dimFixture("impactby", 2)
	f32 := ir.InternScalar(ir.TF32)
	u32 := ir.InternScalar(ir.TU32)

	// Nest A: a plain root loop.
	loopA := forOver(g, dims[0], g.NewConst("ctor", u32, 64), nil)

	// A store whose EFFECT-def is loop A (the repointChains shape: env
	// chain primed at the foreign root — "execute after A completes").
	envB := ir.NewEnv()
	envB.NoteStore(out, loopA)
	stB := g.NewMemStoreChained("ctor", envB, out, []ir.Node{dims[1]}, f32zero(g, f32))

	// Nest B: its BODY is that store — so loop A is uniform-reachable
	// (data+impact closure) from B's body via stB's chain edge.
	loopB := forOver(g, dims[1], g.NewConst("ctor", u32, 8), stB)

	// Sanity: the impact edge really exists and points at loop A.
	found := false
	for _, u := range ir.ImpactBy(stB) {
		if d := u.Def(); d != nil && d.BasePtr() == loopA.BasePtr() {
			found = true
		}
	}
	if !found {
		t.Fatal("fixture bug: stB must carry an ImpactBy edge at loop A")
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("completion edges must never nest: want 2 roots [A, B], got %d", len(roots))
	}
	if roots[0].Iter != loopA || roots[1].Iter != loopB {
		t.Fatal("roots must be [loopA, loopB] in allocation order")
	}
	for _, r := range roots {
		if len(r.Children) != 0 {
			t.Fatalf("loop %d must have no children — effect edges are not membership edges",
				r.Iter.BasePtr().ID())
		}
	}
}

// f32zero allocates one f32 zero const (scaffold value).
func f32zero(g *ir.Graph, f32 ir.Type) ir.Node {
	return g.NewConst("ctor", f32, 0)
}
