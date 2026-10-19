package loopforest_test

import (
	"fmt"
	"strings"
	"testing"

	// The top-level analysis package stays imported for the Analysis-
	// interface satisfaction pin in TestName (opt.Cache compatibility).
	"tcacomp/internal/analysis"
	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"

	// The post-GVN integration case runs the real dedup pass.
	"tcacomp/internal/opt/gvn"
)

// loopforest_test.go pins the loop-nest forest analysis per
// doc/optana/loop_fusion.md ("Loop Nests/Forrest analysis" — the live
// spec): live-Iter-only membership, reversed parent edges (an IR edge
// iter(x) -> iter(y) becomes forest edge y parent-of x and x leaves the
// root set), allocation-order determinism, depths from roots, the
// separate-tree rule, the read-only contract, and the post-GVN
// single-survivor invariant. Future consumer: loop tiling
// (doc/optana/loop_tiling.md).

// scaffold builds the observable spine every fixture shares (mirrors
// gvn_test.go's helper) so the graphs are well-formed for ir.Verify.
func scaffold(g *ir.Graph, m, out, val ir.Node) (st, end ir.Node) {
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, val)
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return st, end
}

// dimFixture builds a graph with count dim params named d0..dN-1 and the
// standard tensor out over the first dim.
func dimFixture(name string, count int) (g *ir.Graph, dims []ir.Node, out ir.Node) {
	g = ir.NewGraph(name)
	for i := 0; i < count; i++ {
		dims = append(dims, g.NewInputParam("ctor", ir.RoleDim, fmt.Sprintf("d%d", i), nil))
	}
	out = g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "d0"}}))
	return g, dims, out
}

// TestName pins the Analysis-name contract (opt.Cache compatibility) and
// the registry cross-check shape: the zero Forest and every Analyze
// product answer "loop-forest".
func TestName(t *testing.T) {
	var zero loopforest.Forest
	if got := zero.Name(); got != "loop-forest" {
		t.Fatalf("Name() = %q, want %q", got, "loop-forest")
	}
	var a analysis.Analysis = loopforest.Analyze(ir.NewGraph("empty"))
	if a.Name() != "loop-forest" {
		t.Fatalf("Forest must satisfy analysis.Analysis, Name() = %q", a.Name())
	}
}

// TestFlatGraphAllSingletons pins today's only builder shape: no
// iter→iter operand edges, so every live iter is its own depth-0 root in
// allocation order — and only KindIter is collected (IterCluster, DimVar,
// ForLoop are not forest members).
func TestFlatGraphAllSingletons(t *testing.T) {
	g, dims, out := dimFixture("flat", 2)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	it1 := g.NewIter("ctor", dims[0], nil, c0, c1)
	it2 := g.NewIter("ctor", dims[1], nil, c0, c1) // same consts, distinct dim: still flat
	cluster := g.NewIterCluster("ctor", []ir.Node{dims[0]}, c0)
	dv := g.NewDimVar("ctor", dims[0], c0, c1)
	loop := g.NewForLoop("ctor", dv, c1, c1)
	scaffold(g, dims[0], out, c0)

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("two unrelated iters must yield 2 roots, got %d", len(roots))
	}
	if roots[0].Iter != it1 || roots[1].Iter != it2 {
		t.Fatalf("roots must be in allocation order [it1, it2]")
	}
	for _, r := range roots {
		if r.Depth != 0 {
			t.Fatalf("a flat root must have Depth 0, got %d", r.Depth)
		}
		if len(r.Children) != 0 {
			t.Fatalf("a flat root must have no children, got %d", len(r.Children))
		}
		if r.Parent != nil {
			t.Fatal("a flat root must have no parent")
		}
	}
	for _, it := range []ir.Node{it1, it2} {
		if _, ok := f.NodeOf(it); !ok {
			t.Fatalf("NodeOf must hold for live iter %d", it.BasePtr().ID())
		}
	}
	for name, node := range map[string]ir.Node{
		"itercluster": cluster,
		"dimvar":      dv,
		"forloop":     loop,
	} {
		if _, ok := f.NodeOf(node); ok {
			t.Fatalf("%s must not be a forest member (only KindIter is collected)", name)
		}
	}
}

// TestNestedChain pins the reversed-edge core: parent-wired iters carry
// the parent Iter as their FINAL operand (the layout convention shared
// with the gvn fusion tests), so the forest parents inner at outer and
// innermost at inner, with depths 0/1/2 counting from the single root.
func TestNestedChain(t *testing.T) {
	g, dims, out := dimFixture("chain", 3)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	outer := g.NewIter("ctor", dims[0], nil, c0, c1)
	inner := g.NewNode(ir.KindIter, "ctor", dims[1], c0, c1, outer)     // parent appended LAST
	innermost := g.NewNode(ir.KindIter, "ctor", dims[2], c0, c1, inner) // ...and again
	scaffold(g, dims[0], out, innermost)

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != outer {
		t.Fatalf("the chain must have exactly one root (outer), got %d roots", len(roots))
	}
	to, tok := f.NodeOf(outer)
	ti, tik := f.NodeOf(inner)
	tm, tmk := f.NodeOf(innermost)
	if !tok || !tik || !tmk {
		t.Fatal("all three chain iters must be forest members")
	}
	if to.Depth != 0 || ti.Depth != 1 || tm.Depth != 2 {
		t.Fatalf("depths must be 0/1/2 from the root, got %d/%d/%d", to.Depth, ti.Depth, tm.Depth)
	}
	if ti.Parent != to || tm.Parent != ti {
		t.Fatal("the reversed edges must parent inner at outer, innermost at inner")
	}
	if len(to.Children) != 1 || to.Children[0] != ti {
		t.Fatal("outer's children must be exactly [inner]")
	}
	if len(ti.Children) != 1 || ti.Children[0] != tm {
		t.Fatal("inner's children must be exactly [innermost]")
	}
	if len(tm.Children) != 0 {
		t.Fatal("innermost must be a leaf")
	}
}

// TestTwoNestsTwoRoots pins two disjoint outer/inner chains: two roots,
// each with exactly one child, children parented at their own root.
func TestTwoNestsTwoRoots(t *testing.T) {
	g, dims, out := dimFixture("twonests", 4)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	o1 := g.NewIter("ctor", dims[0], nil, c0, c1)
	i1 := g.NewNode(ir.KindIter, "ctor", dims[1], c0, c1, o1)
	o2 := g.NewIter("ctor", dims[2], nil, c0, c1)
	i2 := g.NewNode(ir.KindIter, "ctor", dims[3], c0, c1, o2)
	scaffold(g, dims[0], out, i1)

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 2 {
		t.Fatalf("two disjoint nests must yield 2 roots, got %d", len(roots))
	}
	if roots[0].Iter != o1 || roots[1].Iter != o2 {
		t.Fatal("roots must be in allocation order [o1, o2]")
	}
	for _, r := range roots {
		if len(r.Children) != 1 {
			t.Fatalf("each nest root must have exactly one child, got %d", len(r.Children))
		}
		child := r.Children[0]
		if child.Parent != r {
			t.Fatal("the child must be parented at its own root")
		}
		if r.Depth != 0 || child.Depth != 1 {
			t.Fatalf("nest depths must be 0/1, got %d/%d", r.Depth, child.Depth)
		}
	}
	if _, ok := f.NodeOf(i2); !ok {
		t.Fatal("every live iter must be a forest member")
	}
}

// TestErasedIterSkipped pins the live-membership rule: an iter erased via
// ir.MutContext before Analyze is not collected — the forest holds
// exactly the survivor.
func TestErasedIterSkipped(t *testing.T) {
	g, dims, out := dimFixture("erased", 1)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	it1 := g.NewIter("ctor", dims[0], nil, c0, c1)
	it2 := g.NewIter("ctor", dims[0], nil, c0, c1)
	scaffold(g, dims[0], out, c0) // iters stay user-free so Erase is legal

	mctx := &ir.MutContext{Pass: "t"}
	if err := mctx.Erase(it2); err != nil {
		t.Fatalf("Erase: %v", err)
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != it1 {
		t.Fatalf("the forest must hold exactly the survivor, got %d roots", len(roots))
	}
	if _, ok := f.NodeOf(it2); ok {
		t.Fatal("the erased iter must not be a forest member")
	}
}

// TestNilGraphEmpty pins the nil-graph contract: an empty forest (no
// roots, NodeOf false for nil and foreign nodes) whose Name() still
// answers "loop-forest" (Register cross-checks Factory(nil).Name()).
func TestNilGraphEmpty(t *testing.T) {
	f := loopforest.Analyze(nil)
	if got := f.Name(); got != "loop-forest" {
		t.Fatalf("Name() = %q, want %q", got, "loop-forest")
	}
	if roots := f.Roots(); len(roots) != 0 {
		t.Fatalf("Analyze(nil) must yield an empty forest, got %d roots", len(roots))
	}
	if _, ok := f.NodeOf(nil); ok {
		t.Fatal("NodeOf(nil) must be false")
	}
	other := ir.NewGraph("other")
	n := other.NewConst("ctor", ir.InternScalar(ir.TF32), 1)
	if _, ok := f.NodeOf(n); ok {
		t.Fatal("NodeOf of a foreign node must be false")
	}
}

// TestDeterministicAndReadOnly pins the two analysis-package contracts:
// two Analyze runs produce structurally identical forests (root order,
// per-tree children order, depths — rendered to a fingerprint), and
// Analyze provably does not mutate — the node set, use counts, and
// erased-flags are identical before and after.
func TestDeterministicAndReadOnly(t *testing.T) {
	g, dims, out := dimFixture("determinism", 3)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	flat := g.NewIter("ctor", dims[2], nil, c0, c1)
	outer := g.NewIter("ctor", dims[0], nil, c0, c1)
	inner := g.NewNode(ir.KindIter, "ctor", dims[1], c0, c1, outer)
	scaffold(g, dims[0], out, inner)
	_ = flat

	render := func(f loopforest.Forest) string {
		var b strings.Builder
		var walk func(t *loopforest.Tree)
		walk = func(t *loopforest.Tree) {
			fmt.Fprintf(&b, "(%d:%d", t.Iter.BasePtr().ID(), t.Depth)
			for _, c := range t.Children {
				walk(c)
			}
			b.WriteString(")")
		}
		for _, r := range f.Roots() {
			walk(r)
		}
		return b.String()
	}

	count := len(g.AllocatedNodes())
	usesBefore := make(map[uint64]int)
	for _, n := range g.AllocatedNodes() {
		usesBefore[n.BasePtr().ID()] = len(ir.UsesOf(n))
	}

	f1, f2 := loopforest.Analyze(g), loopforest.Analyze(g)
	if r1, r2 := render(f1), render(f2); r1 != r2 {
		t.Fatalf("two Analyze runs must produce identical forests:\n%s\n%s", r1, r2)
	}

	if got := len(g.AllocatedNodes()); got != count {
		t.Fatalf("Analyze must not erase/allocate: %d before, %d after", count, got)
	}
	for _, n := range g.AllocatedNodes() {
		if got := len(ir.UsesOf(n)); got != usesBefore[n.BasePtr().ID()] {
			t.Fatalf("Analyze must not rewire uses of node %d: %d before, %d after",
				n.BasePtr().ID(), usesBefore[n.BasePtr().ID()], got)
		}
		if n.BasePtr().Erased() {
			t.Fatal("Analyze must not tombstone nodes")
		}
	}
}

// TestForestAfterGVNFusion is the bridge pin tying fusion to the forest
// invariant: after gvn fuses two identical iters (doc/optana/
// loop_fusion.md), the forest holds exactly one survivor tree and the
// erased duplicate is not a member.
func TestForestAfterGVNFusion(t *testing.T) {
	g, dims, out := dimFixture("fuseforest", 1)
	f32 := ir.InternScalar(ir.TF32)
	one := g.NewConst("ctor", f32, 1)
	it1 := g.NewIter("ctor", dims[0], nil, one, one)
	it2 := g.NewIter("ctor", dims[0], nil, one, one)
	consumer := g.NewBinOp("ctor", ir.BinOpAdd, it1, it2)
	scaffold(g, dims[0], out, consumer)

	if err := (gvn.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("gvn.Run: %v", err)
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != it1 {
		t.Fatalf("after fusion the forest must hold exactly the survivor, got %d roots", len(roots))
	}
	if _, ok := f.NodeOf(it2); ok {
		t.Fatal("the erased duplicate must not be a forest member")
	}
}
