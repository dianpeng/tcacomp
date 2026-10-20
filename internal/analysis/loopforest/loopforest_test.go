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

// iterOver builds a normalized iter over dim bounded by max (optionally
// parented): DimVar(dim,0,1) -> BoundedDimVar(dv,max) -> Iter(biv,parent).
func iterOver(g *ir.Graph, dim, max, parent ir.Node) ir.Node {
	f32 := ir.InternScalar(ir.TF32)
	dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
	return g.NewIter("ctor", g.NewBoundedDimVar("ctor", dv, max), parent)
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

// TestFlatGraphAllSingletons pins the parentless shape: no iter→iter
// operand edges, so every live root iter is its own depth-0 root in
// allocation order — and only KindIter is collected (IterCluster, DimVar,
// ForLoop are not forest members).
func TestFlatGraphAllSingletons(t *testing.T) {
	g, dims, out := dimFixture("flat", 2)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 1)
	it1 := iterOver(g, dims[0], c0, nil)
	it2 := iterOver(g, dims[1], c0, nil) // distinct dim: still flat
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

// TestNestedChain pins the reversed-edge core: nested iters carry the
// parent Iter as operand 1 (the (biv, parent?) formation), so the forest
// parents inner at outer and innermost at inner, with depths 0/1/2
// counting from the single root.
func TestNestedChain(t *testing.T) {
	g, dims, out := dimFixture("chain", 3)
	f32 := ir.InternScalar(ir.TF32)
	c0 := g.NewConst("ctor", f32, 0) // the shared max of all three levels
	outer := iterOver(g, dims[0], c0, nil)
	inner := iterOver(g, dims[1], c0, outer)
	innermost := iterOver(g, dims[2], c0, inner)
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
	o1 := iterOver(g, dims[0], c0, nil)
	i1 := iterOver(g, dims[1], c0, o1)
	o2 := iterOver(g, dims[2], c0, nil)
	i2 := iterOver(g, dims[3], c0, o2)
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
	it1 := iterOver(g, dims[0], c0, nil)
	it2 := iterOver(g, dims[0], c0, nil) // fresh value objects per call
	scaffold(g, dims[0], out, c0)        // iters stay user-free so Erase is legal

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
	flat := iterOver(g, dims[2], c0, nil)
	outer := iterOver(g, dims[0], c0, nil)
	inner := iterOver(g, dims[1], c0, outer)
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
	it1 := iterOver(g, dims[0], one, nil)
	it2 := iterOver(g, dims[0], one, nil) // fresh value objects per call: gvn has duplicates to fuse
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

// TestForestAfterGVNPrefixFusion pins the forest consequence of PREFIX
// fusion (doc/optana/loop_fusion.md): gvn fuses nest2's common prefix
// into nest1 and re-parents the deeper level D2 onto C1, so the forest
// holds ONE root with a 0/1/2/3 chain — D2 is a member (re-parented),
// the fused-away A2 is not.
func TestForestAfterGVNPrefixFusion(t *testing.T) {
	g := ir.NewGraph("prefixforest")
	f32 := ir.InternScalar(ir.TF32)
	x := g.NewInputParam("ctor", ir.RoleDim, "x", nil)
	y := g.NewInputParam("ctor", ir.RoleDim, "y", nil)
	z := g.NewInputParam("ctor", ir.RoleDim, "z", nil)
	w := g.NewInputParam("ctor", ir.RoleDim, "w", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "x"}}))

	// nest 1: A1 <- B1 <- C1 (consts z1/s1; per-level maxes 64/128/256).
	z1, s1 := g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1)
	mA1, mB1, mC1 := g.NewConst("ctor", f32, 64), g.NewConst("ctor", f32, 128), g.NewConst("ctor", f32, 256)
	dvA1 := g.NewDimVar("ctor", x, z1, s1)
	A1 := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dvA1, mA1), nil)
	dvB1 := g.NewDimVar("ctor", y, z1, s1)
	B1 := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dvB1, mB1), A1)
	dvC1 := g.NewDimVar("ctor", z, z1, s1)
	bivC1 := g.NewBoundedDimVar("ctor", dvC1, mC1)
	C1 := g.NewIter("ctor", bivC1, B1)

	// nest 2: A2 <- B2 <- C2 <- D2 — separately-built equal value objects.
	z2, s2 := g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1)
	mA2, mB2 := g.NewConst("ctor", f32, 64), g.NewConst("ctor", f32, 128)
	mC2, mD2 := g.NewConst("ctor", f32, 256), g.NewConst("ctor", f32, 512)
	dvA2 := g.NewDimVar("ctor", x, z2, s2)
	A2 := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dvA2, mA2), nil)
	dvB2 := g.NewDimVar("ctor", y, z2, s2)
	B2 := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dvB2, mB2), A2)
	dvC2 := g.NewDimVar("ctor", z, z2, s2)
	C2 := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dvC2, mC2), B2)
	dvD2 := g.NewDimVar("ctor", w, z2, s2)
	bivD2 := g.NewBoundedDimVar("ctor", dvD2, mD2)
	D2 := g.NewIter("ctor", bivD2, C2)

	consumer := g.NewBinOp("ctor", ir.BinOpAdd, C1, D2)
	scaffold(g, x, out, consumer)

	if err := (gvn.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("gvn.Run: %v", err)
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != A1 {
		t.Fatalf("the prefix-fused forest must have exactly one root (A1), got %d", len(roots))
	}
	ta, _ := f.NodeOf(A1)
	tb, _ := f.NodeOf(B1)
	tc, _ := f.NodeOf(C1)
	td, okd := f.NodeOf(D2)
	if !okd {
		t.Fatal("D2 must survive as a forest member")
	}
	if ta.Depth != 0 || tb.Depth != 1 || tc.Depth != 2 || td.Depth != 3 {
		t.Fatalf("the chain depths must be 0/1/2/3, got %d/%d/%d/%d",
			ta.Depth, tb.Depth, tc.Depth, td.Depth)
	}
	if tb.Parent != ta || tc.Parent != tb || td.Parent != tc {
		t.Fatal("the chain must link A1 <- B1 <- C1 <- D2 (D2 re-parented onto C1)")
	}
	if _, ok := f.NodeOf(A2); ok {
		t.Fatal("the fused-away prefix iter must not be a forest member")
	}
}
