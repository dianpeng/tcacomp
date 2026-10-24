package loopforest_test

// loopforest_lir_test.go pins the generic analysis over the LIR forloop
// dialect (quick-261022-f5o D-03, rewritten by fix1.md D-b): a forloop
// chain links by IV DATA-DEP REFERENCE DERIVATION (loop L_in is nested in
// L_out iff L_out's IV dimvar is in L_in's data-dep operand closure), a
// mixed iter+forloop graph links coherently (iters via ParentEdge,
// forloops via IV-closure), and — the D-04 probe — an ImpactBy completion
// edge to a FOREIGN nest's root loop can never create a forest parent link
// (repointChains' shape: completion edges order, membership edges nest).

import (
	"testing"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// forOver builds a forloop over a fresh normalized dimvar on dim, bounded
// by max. Sea-of-nodes (fix1.md §0): ForLoop has no body operand; the body
// arg is accepted for signature compatibility but unused. To pin nesting,
// callers wire IV references explicitly (e.g. forOverNested).
func forOver(g *ir.Graph, dim, max, _ ir.Node) ir.Node {
	u32 := ir.InternScalar(ir.TU32)
	dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	biv := g.NewBoundedDimVar("ctor", dv, max)
	return g.NewForLoop("ctor", dv, biv)
}

// forOverNested builds an inner forloop whose bound references outerIV in
// its data-dep operand closure — the fix1.md D-b nesting relation: the
// inner loop's closure contains the outer loop's IV dimvar, so the forest
// parents inner under outer.
func forOverNested(g *ir.Graph, dim ir.Node, outerIV ir.Node, max float64) ir.Node {
	u32 := ir.InternScalar(ir.TU32)
	dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	// The bound references the outer loop's IV: BinOpAdd(outerIV, max-const).
	// This is the round-base / promoted-site shape (fix1.md §1.4): the
	// bound's operand closure contains the enclosing loop's IV.
	bound := g.NewBinOp("ctor", ir.BinOpAdd, outerIV, g.NewConst("ctor", u32, max))
	biv := g.NewBoundedDimVar("ctor", dv, bound)
	return g.NewForLoop("ctor", dv, biv)
}

// loopIV returns the loop's IV dimvar (operand 0).
func loopIV(loop ir.Node) ir.Node {
	ops := ir.OperandsOf(loop)
	if len(ops) == 0 {
		return nil
	}
	return ops[0]
}

// TestLIRForLoopChain pins the IV-closure link (fix1.md D-b): an inner
// forloop whose bound references the outer loop's IV in its data-dep
// closure parents at the outer; depths are 0-based (D-02); an unrelated
// second forloop stays its own root in allocation order.
func TestLIRForLoopChain(t *testing.T) {
	g, dims, out := dimFixture("lir-chain", 2)
	u32 := ir.InternScalar(ir.TU32)
	// Outer loop first: its IV dimvar must exist before inner references it.
	outerDV := g.NewDimVar("ctor", dims[0], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	outerBIV := g.NewBoundedDimVar("ctor", outerDV, g.NewConst("ctor", u32, 64))
	outer := g.NewForLoop("ctor", outerDV, outerBIV)
	// Inner loop references outer's IV in its bound (D-b nesting).
	inner := forOverNested(g, dims[1], outerDV, 8)
	// A separate, unrelated nest over dims[1].
	other := forOver(g, dims[1], g.NewConst("ctor", u32, 8), nil)
	f32 := ir.InternScalar(ir.TF32)
	scaffold(g, dims[0], out, f32zero(g, f32))
	_ = other

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("the IV-linked chain plus the separate forloop must yield 2 roots, got %d", len(roots))
	}
	if roots[0].Iter != outer || roots[1].Iter != other {
		t.Fatalf("roots must be [outer, other] in allocation order, got [%d, %d]",
			roots[0].Iter.BasePtr().ID(), roots[1].Iter.BasePtr().ID())
	}
	if roots[0].Depth != 0 {
		t.Fatalf("outer root depth = %d, want 0 (the 0-based convention, D-02)", roots[0].Depth)
	}
	if len(roots[0].Children) != 1 || roots[0].Children[0].Iter != inner {
		t.Fatal("inner must be outer's only child (IV data-dep closure nesting, fix1.md D-b)")
	}
	if roots[0].Children[0].Depth != 1 {
		t.Fatalf("inner depth = %d, want 1", roots[0].Children[0].Depth)
	}
	if roots[0].Children[0].Parent != roots[0] {
		t.Fatal("inner's Parent must be outer's tree")
	}
}

// TestMixedDialectLinking pins coherence on a hypothetical mixed graph: an
// iter whose dimvar's step references the forloop's IV links under the
// forloop via IV-closure derivation, while a free iter stays a root and
// links nothing. (Iter ParentEdge is checked first; the IV-closure
// derivation applies to KindForLoop only, so the mixed case wires the
// iter's step to reference the forloop's IV dimvar.)
func TestMixedDialectLinking(t *testing.T) {
	g, dims, out := dimFixture("mixed", 2)
	u32 := ir.InternScalar(ir.TU32)
	f32 := ir.InternScalar(ir.TF32)
	// The forloop root.
	loopDV := g.NewDimVar("ctor", dims[0], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	loopBIV := g.NewBoundedDimVar("ctor", loopDV, g.NewConst("ctor", u32, 64))
	loop := g.NewForLoop("ctor", loopDV, loopBIV)
	// An iter over dims[1] whose step references the forloop's IV: the
	// iter's data-dep closure contains loopDV. NOTE: ParentEdge is checked
	// FIRST in Analyze, and the iter's ParentEdge is nil (no loop operand
	// in DataDeps), so the iter remains a root. IV-closure derivation is
	// KindForLoop-only. This test now pins that the iter does NOT nest
	// under the forloop (mixed-dialect IV-closure linking is out of scope
	// for the LIR-only rule).
	_ = loopDV
	inside := iterOver(g, dims[1], g.NewConst("ctor", u32, 8), nil)
	free := iterOver(g, dims[1], g.NewConst("ctor", u32, 8), nil) // root iter, no edges
	scaffold(g, dims[0], out, f32zero(g, f32))

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 3 {
		t.Fatalf("the forloop and both iters must yield 3 roots (IV-closure is forloop-only), got %d", len(roots))
	}
	if roots[0].Iter != loop || roots[1].Iter != inside || roots[2].Iter != free {
		t.Fatal("roots must be [loop, insideIter, freeIter] in allocation order")
	}
	for _, r := range roots {
		if len(r.Children) != 0 {
			t.Fatalf("loop %d must have no children (mixed-dialect IV-closure linking is forloop-only)",
				r.Iter.BasePtr().ID())
		}
	}
}

// TestImpactByNeverNests pins D-04 — THE reason the spec interface exists:
// an effect edge (ImpactBy) from inside nest B's body at nest A's ROOT
// loop node is a COMPLETION edge ("after the whole loop"), never a
// membership edge. The forest must stay two-rooted even though loop A is
// uniform-reachable from loop B's body. Post-fix1.md, the BODY is gone;
// the ImpactBy edge rides the store's chain slot directly. Loop B's
// operand closure does NOT reference loop A's IV (no data-dep path), so
// IV-closure derivation correctly keeps the nests separate.
func TestImpactByNeverNests(t *testing.T) {
	g, dims, out := dimFixture("impactby", 2)
	f32 := ir.InternScalar(ir.TF32)
	u32 := ir.InternScalar(ir.TU32)

	// Nest A: a plain root loop over dims[0].
	loopA := forOver(g, dims[0], g.NewConst("ctor", u32, 64), nil)

	// A store whose EFFECT-def is loop A (the repointChains shape: env
	// chain primed at the foreign root — "execute after A completes").
	envB := ir.NewEnv()
	envB.NoteStore(out, loopA)
	stB := g.NewMemStoreChained("ctor", envB, out, []ir.Node{dims[1]}, f32zero(g, f32))

	// Nest B: a root loop over dims[1]; its operand closure does NOT
	// reference loop A's IV (the ImpactBy edge is a CHAIN edge, not a
	// data-dep operand — D-04: completion edges never nest).
	_ = stB
	loopB := forOver(g, dims[1], g.NewConst("ctor", u32, 8), nil)

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

	// Sanity: loop B's operand closure does NOT contain loop A's IV.
	bClosure := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || bClosure[x] {
			return
		}
		bClosure[x] = true
		for _, d := range ir.OperandsOf(x) {
			visit(d)
		}
	}
	visit(loopB)
	if bClosure[loopIV(loopA)] {
		t.Fatal("fixture bug: loop B's operand closure must NOT contain loop A's IV")
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

// TestLIRIVDerivedNestingPositive pins the fix1.md D-b positive case
// explicitly: an inner forloop whose bound references the outer's IV nests
// under the outer — the exact relation buildNest produces post-Step 3.
func TestLIRIVDerivedNestingPositive(t *testing.T) {
	g, dims, out := dimFixture("iv-nesting", 2)
	u32 := ir.InternScalar(ir.TU32)
	// Outer loop over dims[0].
	outerDV := g.NewDimVar("ctor", dims[0], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
	outer := g.NewForLoop("ctor", outerDV, g.NewBoundedDimVar("ctor", outerDV, g.NewConst("ctor", u32, 64)))
	// Middle loop references outer's IV.
	middle := forOverNested(g, dims[1], outerDV, 8)
	// Innermost loop references middle's IV (transitive closure).
	middleIV := loopIV(middle)
	inner := forOverNested(g, dims[0], middleIV, 4)
	f32 := ir.InternScalar(ir.TF32)
	scaffold(g, dims[0], out, f32zero(g, f32))

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 {
		t.Fatalf("the 3-deep IV chain must yield 1 root, got %d", len(roots))
	}
	if roots[0].Iter != outer {
		t.Fatalf("root must be outer, got node %d", roots[0].Iter.BasePtr().ID())
	}
	if roots[0].Depth != 0 {
		t.Fatalf("outer depth = %d, want 0", roots[0].Depth)
	}
	if len(roots[0].Children) != 1 || roots[0].Children[0].Iter != middle {
		t.Fatal("middle must be outer's only child")
	}
	if roots[0].Children[0].Depth != 1 {
		t.Fatalf("middle depth = %d, want 1", roots[0].Children[0].Depth)
	}
	if len(roots[0].Children[0].Children) != 1 || roots[0].Children[0].Children[0].Iter != inner {
		t.Fatal("inner must be middle's only child (transitive IV closure)")
	}
	if roots[0].Children[0].Children[0].Depth != 2 {
		t.Fatalf("inner depth = %d, want 2", roots[0].Children[0].Children[0].Depth)
	}
}

// f32zero allocates one f32 zero const (scaffold value).
func f32zero(g *ir.Graph, f32 ir.Type) ir.Node {
	return g.NewConst("ctor", f32, 0)
}
