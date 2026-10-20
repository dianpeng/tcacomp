package simplify_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/simplify"

	// The composition root: firing every pass package's self-registration
	// is what makes the pipeline-wiring assertions below observe the
	// default pipeline.
	_ "tcacomp/internal/opt/all"
)

// simplify_test.go pins the Simplification passes (quick-261019-ktm):
// simplify-lower (IterCluster format → Iter format), simplify-inline
// (lambda parameter inlining), and simplify-verify (post-simplify
// invariants), plus their 102/103/104 pipeline wiring between dce-verify
// and gvn.

// docExample builds the doc/optana/lower_itercluster.md myfoo shape IN THE
// ITERCLUSTER FORMAT, directly through the ir constructors (the established
// pass-test pattern — pkg's builder emits the legacy wrapper today):
//
//	dims a/b/c; tensors lhs=[a,b], rhs=[a,c] (f32)
//	cluster = IterCluster(dims=[a,b,c], inputs=[MemLoad(lhs,[a,b]),
//	                                         MemLoad(rhs,[a,c])])
//	map = Map(inputs=[Project(cluster,0), Project(cluster,1)],
//	          lambda(l, r) = l * r, as=[a,b,c])
//
// plus node handles for the assertions.
func docExample(t *testing.T) (g *ir.Graph, lhs, rhs, a, b, c, loadL, loadR, cluster, mp ir.Node) {
	t.Helper()
	g = ir.NewGraph("myfoo")
	f32 := ir.InternScalar(ir.TF32)
	a = g.NewInputParam("ctor", ir.RoleDim, "a", nil)
	b = g.NewInputParam("ctor", ir.RoleDim, "b", nil)
	c = g.NewInputParam("ctor", ir.RoleDim, "c", nil)
	lhs = g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))
	rhs = g.NewInputParam("ctor", ir.RoleTensor, "rhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "c"}}))

	loadL = g.NewMemLoad("ctor", lhs, []ir.Node{a, b})
	loadR = g.NewMemLoad("ctor", rhs, []ir.Node{a, c})
	cluster = g.NewIterCluster("ctor", []ir.Node{a, b, c}, loadL, loadR)

	lp := g.NewLambdaInputParam("ctor", "l", f32)
	rp := g.NewLambdaInputParam("ctor", "r", f32)
	body := g.NewBinOp("ctor", ir.BinOpMul, g.NewParamRef("ctor", lp), g.NewParamRef("ctor", rp))
	p0 := g.NewIterClusterProject("ctor", cluster, 0)
	p1 := g.NewIterClusterProject("ctor", cluster, 1)
	mp = g.NewMap("ctor", []ir.Node{p0, p1}, ir.NewLambda([]ir.Node{lp, rp}, body), []ir.Node{a, b, c})
	g.NewReturn("ctor", mp)

	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("IterCluster-format fixture must verify clean, got: %v", vs)
	}
	return g, lhs, rhs, a, b, c, loadL, loadR, cluster, mp
}

// runLower / runInline / runSimplifyVerify run one pass directly.
func runLower(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (simplify.LowerPass{}).Run(nil, g); err != nil {
		t.Fatalf("simplify-lower: %v", err)
	}
}

func runInline(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (simplify.InlinePass{}).Run(nil, g); err != nil {
		t.Fatalf("simplify-inline: %v", err)
	}
}

// nodeSet/fingerprint: the liveness and stability substrate (gvn_test's
// shape — AllocatedNodes keyed by the shared Base, and a structural
// fingerprint for idempotence probes).
func nodeSet(g *ir.Graph) map[*ir.Base]bool {
	set := make(map[*ir.Base]bool)
	for _, n := range g.AllocatedNodes() {
		set[n.BasePtr()] = true
	}
	return set
}

func fingerprint(g *ir.Graph) string {
	var b strings.Builder
	for _, n := range g.AllocatedNodes() {
		fmt.Fprintf(&b, "%d:%s(", n.BasePtr().ID(), n.KindOf())
		for _, d := range ir.OperandsOf(n) {
			if d == nil {
				b.WriteString("nil,")
			} else {
				fmt.Fprintf(&b, "%d,", d.BasePtr().ID())
			}
		}
		b.WriteString(")[")
		for _, u := range ir.UsesOf(n) {
			if user := u.User(); user != nil {
				fmt.Fprintf(&b, "%d,", user.BasePtr().ID())
			} else {
				b.WriteString("nil,")
			}
		}
		b.WriteString("]\n")
	}
	return b.String()
}

// nodesOfKind collects the graph's live nodes of kind k in allocation order.
func nodesOfKind(g *ir.Graph, k ir.Kind) []ir.Node {
	var out []ir.Node
	for _, n := range g.AllocatedNodes() {
		if n.KindOf() == k {
			out = append(out, n)
		}
	}
	return out
}

// assertCleanVerify fails unless ir.Verify reports zero violations.
func assertCleanVerify(t *testing.T, g *ir.Graph, what string) {
	t.Helper()
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("%s must verify clean, got: %v", what, vs)
	}
}

// ---------------------------------------------------------------------------
// simplify-lower
// ---------------------------------------------------------------------------

// TestLowerDocExample pins the doc's worked example: the IterCluster format
// lowers to the Iter format — exactly 3 iters keyed by dim in AS ORDER
// (a←TensorDimOf(lhs,0), b←TensorDimOf(lhs,1), c←TensorDimOf(rhs,1)), each
// load axis rewired to an InductionVariable of the dim's iter, the map's
// input slots pointing at the loads directly, and project+cluster erasure.
func TestLowerDocExample(t *testing.T) {
	g, lhsT, rhsT, aP, bP, cP, loadL, loadR, cluster, mp := docExample(t)

	runLower(t, g)

	// Exactly one iter per as-dim, in AS ORDER, each bound by the right
	// TensorDimOf (first pairing wins: a,b from lhs; c from rhs axis 1).
	iters := nodesOfKind(g, ir.KindIter)
	if len(iters) != 3 {
		t.Fatalf("the lowered graph must hold exactly 3 iters, got %d", len(iters))
	}
	wantBound := []struct {
		dim    ir.Node
		tensor ir.Node
		axis   int
	}{{aP, lhsT, 0}, {bP, lhsT, 1}, {cP, rhsT, 1}}
	for i, it := range iters {
		w := wantBound[i]
		if it.KindOf() != ir.KindIter {
			t.Fatalf("node %d is %s, want iter", it.BasePtr().ID(), it.KindOf())
		}
		// The iter binds its dim through the BIV chain: operand 0 (biv) ->
		// operand 0 (dv) -> operand 0 (the dim param).
		if got := iterDimOf(t, it); got.BasePtr() != w.dim.BasePtr() {
			t.Fatalf("iter %d must bind dim node %d in as-order position %d", it.BasePtr().ID(), w.dim.BasePtr().ID(), i)
		}
		td := boundTensorDimOf(t, g, it)
		if td == nil {
			t.Fatalf("iter %d must be bounded by a TensorDimOf", it.BasePtr().ID())
		}
		ops := ir.OperandsOf(td)
		if len(ops) != 2 || ops[0] != w.tensor {
			t.Fatalf("iter %d's bound must be TensorDimOf(tensor %d), got ops %v", it.BasePtr().ID(), w.tensor.BasePtr().ID(), ops)
		}
		if got := ir.Label(ops[1], nil); got != fmt.Sprint(w.axis) {
			t.Fatalf("iter %d's TensorDimOf axis = %s, want %d", it.BasePtr().ID(), got, w.axis)
		}
	}

	// Each MemLoad axis slot resolves to an InductionVariable whose iter
	// binds the ORIGINAL axis dim.
	for _, ld := range []struct {
		load ir.Node
		dims []ir.Node
	}{
		{loadL, []ir.Node{aP, bP}},
		{loadR, []ir.Node{aP, cP}},
	} {
		ops := ir.OperandsOf(ld.load)
		if len(ops) != 1+len(ld.dims) {
			t.Fatalf("load %d axes changed shape: %v", ld.load.BasePtr().ID(), ops)
		}
		for i, dim := range ld.dims {
			iv := ops[i+1]
			if iv.KindOf() != ir.KindInductionVariable {
				t.Fatalf("load axis %d must resolve to an InductionVariable, got %s", i, iv.KindOf())
			}
			iter := ir.OperandsOf(iv)[0]
			if iter.KindOf() != ir.KindIter || iterDimOf(t, iter).BasePtr() != dim.BasePtr() {
				t.Fatalf("load axis %d's IV must ride an iter binding dim %d", i, dim.BasePtr().ID())
			}
		}
	}

	// The map's operand slots 0/1 resolve to the two loads; the trailing As
	// survives as the last VALUE operand, before the lambda node (D15).
	ops := ir.OperandsOf(mp)
	if len(ops) != 4 || ops[0] != loadL || ops[1] != loadR ||
		ops[2].KindOf() != ir.KindAs || ops[3].KindOf() != ir.KindLambda {
		t.Fatalf("map inputs must resolve to the loads directly, got %v", ops)
	}
	if as := ir.AsOf(mp); as == nil || len(ir.OperandsOf(as)) != 3 {
		t.Fatal("the map's trailing As must survive lowering")
	}

	// No live project or cluster remains; both are tombstoned.
	for _, k := range []ir.Kind{ir.KindIterClusterProject, ir.KindIterCluster} {
		if live := nodesOfKind(g, k); len(live) != 0 {
			t.Fatalf("no live %s may remain after lowering, got %d", k, len(live))
		}
	}
	if alive := nodeSet(g); alive[cluster.BasePtr()] {
		t.Fatal("the cluster must be erased (tombstoned)")
	}
	assertCleanVerify(t, g, "the lowered doc example")
}

// useWithUser reports whether def holds a live use whose user is n.
func useWithUser(def, n ir.Node) bool {
	for _, u := range ir.UsesOf(def) {
		if user := u.User(); user != nil && user.BasePtr() == n.BasePtr() {
			return true
		}
	}
	return false
}

// iterDimOf chases an iter's BIV chain to its dim InputParam:
// iter -> operand 0 (biv) -> operand 0 (dv) -> operand 0 (the dim param).
// Fail-fatally on a wrong-kind link — a lowered iter is well-formed by
// construction.
func iterDimOf(t *testing.T, iter ir.Node) ir.Node {
	t.Helper()
	if iter.KindOf() != ir.KindIter {
		t.Fatalf("iterDimOf: node %d is %s, want iter", iter.BasePtr().ID(), iter.KindOf())
	}
	biv := ir.OperandsOf(iter)[0]
	if biv == nil || biv.KindOf() != ir.KindBoundedDimVar {
		t.Fatalf("iterDimOf: iter %d operand 0 must be a bounded dimvar, got %v", iter.BasePtr().ID(), biv)
	}
	dv := ir.OperandsOf(biv)[0]
	if dv == nil || dv.KindOf() != ir.KindDimVar {
		t.Fatalf("iterDimOf: biv %d operand 0 must be a dimvar, got %v", biv.BasePtr().ID(), dv)
	}
	dim := ir.OperandsOf(dv)[0]
	if dim == nil {
		t.Fatalf("iterDimOf: dimvar %d carries no dim operand", dv.BasePtr().ID())
	}
	return dim
}

// boundTensorDimOf chases the iter's BIV to its max (operand 1), returning
// it only when it is a TensorDimOf node (nil otherwise — the max accepts
// any value node, D5).
func boundTensorDimOf(t *testing.T, g *ir.Graph, iter ir.Node) ir.Node {
	t.Helper()
	if iter.KindOf() != ir.KindIter {
		t.Fatalf("boundTensorDimOf: node %d is %s, want iter", iter.BasePtr().ID(), iter.KindOf())
	}
	biv := ir.OperandsOf(iter)[0]
	if biv == nil || biv.KindOf() != ir.KindBoundedDimVar {
		t.Fatalf("boundTensorDimOf: iter %d operand 0 must be a bounded dimvar, got %v", iter.BasePtr().ID(), biv)
	}
	if mx := ir.OperandsOf(biv)[1]; mx != nil && mx.KindOf() == ir.KindTensorDimOf {
		return mx
	}
	return nil
}

// TestLowerSharedDimFirstWins pins the deterministic shared-dim rule: a dim
// bound by two loads yields ONE iter bound by the FIRST pairing.
func TestLowerSharedDimFirstWins(t *testing.T) {
	g := ir.NewGraph("shared")
	a := g.NewInputParam("ctor", ir.RoleDim, "a", nil)
	b := g.NewInputParam("ctor", ir.RoleDim, "b", nil)
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))
	rhs := g.NewInputParam("ctor", ir.RoleTensor, "rhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))

	loadL := g.NewMemLoad("ctor", lhs, []ir.Node{a, b})
	loadR := g.NewMemLoad("ctor", rhs, []ir.Node{a, b})
	cluster := g.NewIterCluster("ctor", []ir.Node{a, b}, loadL, loadR)
	lp := g.NewLambdaInputParam("ctor", "l", f32(t))
	rp := g.NewLambdaInputParam("ctor", "r", f32(t))
	body := g.NewBinOp("ctor", ir.BinOpMul, g.NewParamRef("ctor", lp), g.NewParamRef("ctor", rp))
	mp := g.NewMap("ctor",
		[]ir.Node{g.NewIterClusterProject("ctor", cluster, 0), g.NewIterClusterProject("ctor", cluster, 1)},
		ir.NewLambda([]ir.Node{lp, rp}, body), []ir.Node{a, b})
	g.NewReturn("ctor", mp)

	runLower(t, g)

	iters := nodesOfKind(g, ir.KindIter)
	if len(iters) != 2 {
		t.Fatalf("shared dims must yield ONE iter per dim, got %d", len(iters))
	}
	// Dim a's iter is bound by the FIRST pairing (lhs, axis 0).
	td := boundTensorDimOf(t, g, iters[0])
	if td == nil || ir.OperandsOf(td)[0] != lhs {
		t.Fatal("dim a's iter must be bound by TensorDimOf(lhs, 0) — first-wins")
	}
	// Both loads' axis-0 slots resolve to the SAME InductionVariable.
	ivL := ir.OperandsOf(loadL)[1]
	ivR := ir.OperandsOf(loadR)[1]
	if ivL.BasePtr() != ivR.BasePtr() {
		t.Fatal("a dim shared by two loads must share one InductionVariable")
	}
	assertCleanVerify(t, g, "the shared-dim lowering")
}

// TestLowerWiresNestChain pins that single-cluster lowering builds a real
// NEST, not a flat iter set: the as-order IS the nesting order, so the doc
// example's as [a,b,c] lowers to a 0/1/2 chain — ia the 1-operand root, ib
// parented at ia, ic parented at ib — and loopforest sees exactly that
// tree.
func TestLowerWiresNestChain(t *testing.T) {
	g, _, _, aP, bP, cP, _, _, _, _ := docExample(t)

	runLower(t, g)

	iters := nodesOfKind(g, ir.KindIter)
	if len(iters) != 3 {
		t.Fatalf("the lowered graph must hold exactly 3 iters, got %d", len(iters))
	}
	ia, ib, ic := iters[0], iters[1], iters[2]
	if got := iterDimOf(t, ia); got.BasePtr() != aP.BasePtr() {
		t.Fatalf("as-order head must bind dim a, got node %d", got.BasePtr().ID())
	}
	if got := iterDimOf(t, ib); got.BasePtr() != bP.BasePtr() {
		t.Fatalf("as-order position 1 must bind dim b, got node %d", got.BasePtr().ID())
	}
	if got := iterDimOf(t, ic); got.BasePtr() != cP.BasePtr() {
		t.Fatalf("as-order position 2 must bind dim c, got node %d", got.BasePtr().ID())
	}
	if ops := ir.OperandsOf(ia); len(ops) != 1 {
		t.Fatalf("ia must be the 1-operand nest root, got %v", ops)
	}
	opsB := ir.OperandsOf(ib)
	if len(opsB) != 2 || opsB[1] != ia {
		t.Fatalf("ib must be parented at ia (ops [biv, ia]), got %v", opsB)
	}
	opsC := ir.OperandsOf(ic)
	if len(opsC) != 2 || opsC[1] != ib {
		t.Fatalf("ic must be parented at ib (ops [biv, ib]), got %v", opsC)
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != ia {
		t.Fatalf("the wired nest must have exactly one root (ia), got %d", len(roots))
	}
	ta, _ := f.NodeOf(ia)
	tb, _ := f.NodeOf(ib)
	tc, _ := f.NodeOf(ic)
	if ta.Depth != 0 || tb.Depth != 1 || tc.Depth != 2 {
		t.Fatalf("the chain depths must be 0/1/2, got %d/%d/%d", ta.Depth, tb.Depth, tc.Depth)
	}
	if tb.Parent != ta || tc.Parent != tb {
		t.Fatal("the forest must chain ia <- ib <- ic")
	}
	assertCleanVerify(t, g, "the wired nest chain")
}

// TestLowerSharedPrefixNotFlat pins cross-cluster prefix interning: two
// clusters whose chains share dim a ([a,b] and [a,c]) yield ONE interned
// a-root with SEPARATE b/c children hanging off THAT SAME iter — a real
// tree, not a flat kernel — and the forest shows one root with children
// [ib, ic] in allocation order.
func TestLowerSharedPrefixNotFlat(t *testing.T) {
	g := ir.NewGraph("sharedprefix")
	ft := f32(t)
	a := g.NewInputParam("ctor", ir.RoleDim, "a", nil)
	b := g.NewInputParam("ctor", ir.RoleDim, "b", nil)
	c := g.NewInputParam("ctor", ir.RoleDim, "c", nil)
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))
	rhs := g.NewInputParam("ctor", ir.RoleTensor, "rhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "c"}}))

	loadL := g.NewMemLoad("ctor", lhs, []ir.Node{a, b})
	cluster1 := g.NewIterCluster("ctor", []ir.Node{a, b}, loadL)
	lp := g.NewLambdaInputParam("ctor", "l", ft)
	mp1 := g.NewMap("ctor", []ir.Node{g.NewIterClusterProject("ctor", cluster1, 0)},
		ir.NewLambda([]ir.Node{lp}, g.NewParamRef("ctor", lp)), []ir.Node{a, b})

	loadR := g.NewMemLoad("ctor", rhs, []ir.Node{a, c})
	cluster2 := g.NewIterCluster("ctor", []ir.Node{a, c}, loadR)
	rp := g.NewLambdaInputParam("ctor", "r", ft)
	mp2 := g.NewMap("ctor", []ir.Node{g.NewIterClusterProject("ctor", cluster2, 0)},
		ir.NewLambda([]ir.Node{rp}, g.NewParamRef("ctor", rp)), []ir.Node{a, c})

	g.NewReturn("ctor", mp1, mp2)
	assertCleanVerify(t, g, "the shared-prefix fixture")

	runLower(t, g)

	iters := nodesOfKind(g, ir.KindIter)
	if len(iters) != 3 {
		t.Fatalf("clusters as [a,b]/[a,c] must yield exactly 3 iters, got %d", len(iters))
	}
	// Exactly ONE iter binds dim a — the interned shared prefix.
	var ia ir.Node
	bindA := 0
	for _, it := range iters {
		if iterDimOf(t, it).BasePtr() == a.BasePtr() {
			ia = it
			bindA++
		}
	}
	if bindA != 1 {
		t.Fatalf("exactly ONE iter may bind dim a, got %d", bindA)
	}
	if ops := ir.OperandsOf(ia); len(ops) != 1 {
		t.Fatalf("ia must be the 1-operand nest root, got %v", ops)
	}
	// ib and ic BOTH parented at THAT iter — c hangs off the SAME interned
	// a-iter, not a flat second root.
	var ib, ic ir.Node
	for _, it := range iters {
		if it.BasePtr() == ia.BasePtr() {
			continue
		}
		switch iterDimOf(t, it).BasePtr() {
		case b.BasePtr():
			ib = it
		case c.BasePtr():
			ic = it
		}
	}
	if ib == nil || ic == nil {
		t.Fatal("the lowering must bind one iter each for dims b and c")
	}
	if ops := ir.OperandsOf(ib); len(ops) != 2 || ops[1] != ia {
		t.Fatalf("ib must be parented at the interned ia, got %v", ops)
	}
	if ops := ir.OperandsOf(ic); len(ops) != 2 || ops[1] != ia {
		t.Fatalf("ic must be parented at the interned ia, got %v", ops)
	}

	f := loopforest.Analyze(g)
	roots := f.Roots()
	if len(roots) != 1 || roots[0].Iter != ia {
		t.Fatalf("the forest must have exactly one root (ia), got %d", len(roots))
	}
	ta, _ := f.NodeOf(ia)
	tb, _ := f.NodeOf(ib)
	tc, _ := f.NodeOf(ic)
	if len(ta.Children) != 2 || ta.Children[0] != tb || ta.Children[1] != tc {
		t.Fatal("ia's children must be exactly [ib, ic] in allocation order")
	}
	if tb.Depth != 1 || tc.Depth != 1 {
		t.Fatalf("the shared-prefix depths must be 0/1/1, got %d/%d", tb.Depth, tc.Depth)
	}
	assertCleanVerify(t, g, "the shared-prefix lowering")
}

// TestLowerAsOrderRespected pins that the NESTING order is the AS order,
// never the dim-declaration order: a cluster as [b,a] over load axes
// [a,b] lowers to root b with child a, and the bound pairings stay per DIM
// (b: TensorDimOf(lhs,1), a: TensorDimOf(lhs,0)).
func TestLowerAsOrderRespected(t *testing.T) {
	g := ir.NewGraph("asorder")
	ft := f32(t)
	a := g.NewInputParam("ctor", ir.RoleDim, "a", nil)
	b := g.NewInputParam("ctor", ir.RoleDim, "b", nil)
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))

	load := g.NewMemLoad("ctor", lhs, []ir.Node{a, b})
	cluster := g.NewIterCluster("ctor", []ir.Node{b, a}, load) // cluster dims as [b, a]
	lp := g.NewLambdaInputParam("ctor", "l", ft)
	mp := g.NewMap("ctor", []ir.Node{g.NewIterClusterProject("ctor", cluster, 0)},
		ir.NewLambda([]ir.Node{lp}, g.NewParamRef("ctor", lp)), []ir.Node{b, a}) // as [b, a]
	g.NewReturn("ctor", mp)
	assertCleanVerify(t, g, "the as-order fixture")

	runLower(t, g)

	iters := nodesOfKind(g, ir.KindIter)
	if len(iters) != 2 {
		t.Fatalf("as [b,a] must yield exactly 2 iters, got %d", len(iters))
	}
	itB, itA := iters[0], iters[1]
	if got := iterDimOf(t, itB); got.BasePtr() != b.BasePtr() {
		t.Fatalf("the FIRST as-dim (b) must head the chain, got node %d", got.BasePtr().ID())
	}
	if ops := ir.OperandsOf(itB); len(ops) != 1 {
		t.Fatalf("b's iter must be the 1-operand nest root, got %v", ops)
	}
	if got := iterDimOf(t, itA); got.BasePtr() != a.BasePtr() {
		t.Fatalf("the SECOND as-dim (a) must follow, got node %d", got.BasePtr().ID())
	}
	if ops := ir.OperandsOf(itA); len(ops) != 2 || ops[1] != itB {
		t.Fatalf("a's iter must be parented at b's iter, got %v", ops)
	}
	// Pairings per DIM — not per chain position.
	tdB := boundTensorDimOf(t, g, itB)
	if tdB == nil || ir.OperandsOf(tdB)[0] != lhs {
		t.Fatal("b's iter must be bound by TensorDimOf(lhs, ...)")
	}
	if got := ir.Label(ir.OperandsOf(tdB)[1], nil); got != fmt.Sprint(1) {
		t.Fatalf("b's bound axis must be 1 (TensorDimOf(lhs, 1)), got %s", got)
	}
	tdA := boundTensorDimOf(t, g, itA)
	if tdA == nil || ir.OperandsOf(tdA)[0] != lhs {
		t.Fatal("a's iter must be bound by TensorDimOf(lhs, ...)")
	}
	if got := ir.Label(ir.OperandsOf(tdA)[1], nil); got != fmt.Sprint(0) {
		t.Fatalf("a's bound axis must be 0 (TensorDimOf(lhs, 0)), got %s", got)
	}

	f := loopforest.Analyze(g)
	if roots := f.Roots(); len(roots) != 1 || roots[0].Iter != itB {
		t.Fatalf("the forest root must be b's iter, got %d roots", len(roots))
	}
	assertCleanVerify(t, g, "the as-order lowering")
}

// TestLowerLegacyNoOp pins D11: a builder-shaped graph (cluster wrapping a
// Map, no projects) passes through untouched — alive-set identical, Verify
// clean.
func TestLowerLegacyNoOp(t *testing.T) {
	g := ir.NewGraph("legacy")
	f := f32(t)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	v := g.NewLambdaInputParam("ctor", "v", f)
	val := g.NewConst("ctor", f, 1)
	body := g.NewBinOp("ctor", ir.BinOpMul, g.NewParamRef("ctor", v), val)
	mp := g.NewMap("ctor", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, body), []ir.Node{m})
	ic := g.NewIterCluster("ctor", []ir.Node{m}, mp)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, ic)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	before := nodeSet(g)
	printBefore := fingerprint(g)
	runLower(t, g)
	after := nodeSet(g)
	if len(before) != len(after) {
		t.Fatalf("legacy no-op changed the alive set: %d before, %d after", len(before), len(after))
	}
	for b := range before {
		if !after[b] {
			t.Fatalf("legacy no-op erased node %d (%s)", b.ID(), b.KindOf())
		}
	}
	if got := fingerprint(g); got != printBefore {
		t.Fatalf("legacy no-op changed the graph:\n--- before ---\n%s\n--- after ---\n%s", printBefore, got)
	}
	assertCleanVerify(t, g, "the legacy no-op")
}

// TestLowerFailClosed pins the level gates and the uncovered-dim refusal:
// nil and materialized graphs error without mutation; a MemLoad axis dim no
// projecting map's As covers fails with an error naming the dim.
func TestLowerFailClosed(t *testing.T) {
	if err := (simplify.LowerPass{}).Run(nil, nil); err == nil || err.Error() != "simplify-lower: nil graph" {
		t.Fatalf("Run(nil) error = %v, want %q", err, "simplify-lower: nil graph")
	}

	g2, _, _, _, _, _, _, _, _, _ := docExample(t)
	if err := g2.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := len(g2.AllocatedNodes())
	if err := (simplify.LowerPass{}).Run(nil, g2); err == nil || !strings.Contains(err.Error(), "materialized") {
		t.Fatalf("a materialized graph must be refused, got: %v", err)
	}
	if after := len(g2.AllocatedNodes()); after != before {
		t.Fatalf("a refused graph must be untouched: %d before, %d after", before, after)
	}

	// Uncovered dim: the map's as lists only @a, but loadL binds @a,@b —
	// @b has no covering axis pairing.
	g3 := ir.NewGraph("uncovered")
	f := f32(t)
	a := g3.NewInputParam("ctor", ir.RoleDim, "a", nil)
	b := g3.NewInputParam("ctor", ir.RoleDim, "b", nil)
	lhs := g3.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "a"}, {Name: "b"}}))
	load := g3.NewMemLoad("ctor", lhs, []ir.Node{a, b})
	cluster := g3.NewIterCluster("ctor", []ir.Node{a, b}, load)
	lp := g3.NewLambdaInputParam("ctor", "l", f)
	mp := g3.NewMap("ctor", []ir.Node{g3.NewIterClusterProject("ctor", cluster, 0)},
		ir.NewLambda([]ir.Node{lp}, g3.NewParamRef("ctor", lp)), []ir.Node{a}) // as=[a] only
	g3.NewReturn("ctor", mp)

	err := (simplify.LowerPass{}).Run(nil, g3)
	if err == nil {
		t.Fatal("an uncovered MemLoad axis dim must fail closed")
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("dim node %d", b.BasePtr().ID())) {
		t.Fatalf("the error must name the uncovered dim node %d, got: %v", b.BasePtr().ID(), err)
	}
}

// f32 is the interned f32 scalar type shortcut.
func f32(t *testing.T) ir.Type {
	t.Helper()
	return ir.InternScalar(ir.TF32)
}

// ---------------------------------------------------------------------------
// simplify-inline
// ---------------------------------------------------------------------------

// TestInlineMap pins the doc example's inline: every paramRef of the map's
// lambda is gone (erased), the body's binop operands resolve directly to
// the two MemLoads, the lambda InputParams are erased, and the graph stays
// clean.
func TestInlineMap(t *testing.T) {
	g, _, _, _, _, _, loadL, loadR, _, mp := docExample(t)
	runLower(t, g)

	// The lambda's two InputParams, captured before the pass.
	params := ir.LambdaParams(mp)
	if len(params) != 1 || len(params[0]) != 2 {
		t.Fatalf("the map's lambda must declare 2 params, got %v", params)
	}
	lp, rp := params[0][0], params[0][1]

	runInline(t, g)

	// No live paramRef remains anywhere.
	if refs := nodesOfKind(g, ir.KindParam); len(refs) != 0 {
		t.Fatalf("all paramRefs must be erased by inline, got %d", len(refs))
	}
	// The body binop's operands resolve directly to the loads: the body is
	// the map's single lambda body (a binop rooted at the paramRefs).
	bodies := ir.LambdaBodies(mp)
	if len(bodies) != 1 || bodies[0].KindOf() != ir.KindBinOp {
		t.Fatalf("the map's body must remain a binop, got %v", bodies)
	}
	if ops := ir.OperandsOf(bodies[0]); len(ops) != 2 || ops[0] != loadL || ops[1] != loadR {
		t.Fatalf("the inlined body must resolve to the two loads, got %v", ops)
	}
	// The lambda InputParams are erased (tombstoned) and use-free.
	alive := nodeSet(g)
	for _, p := range []ir.Node{lp, rp} {
		if alive[p.BasePtr()] {
			t.Fatalf("lambda param %d must be erased", p.BasePtr().ID())
		}
	}
	// The map's lambda record now declares nothing.
	if params := ir.LambdaParams(mp); len(params) != 1 || len(params[0]) != 0 {
		t.Fatalf("the inlined map's lambda must declare no params, got %v", params)
	}
	assertCleanVerify(t, g, "the inlined doc example")
}

// TestInlineReduce pins D9's positional rule on a reduce: [piped,
// attr(init)] binds [acc, now] as acc↔piped, now↔init-VALUE; a param still
// referenced through a non-paramRef use stays declared (ClearLambdaParams
// keeps live ones), and a body-root paramRef (identity map) rewires through
// the body choke point.
func TestInlineReduce(t *testing.T) {
	g := ir.NewGraph("redinline")
	f := f32(t)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	tn := g.NewInputParam("ctor", ir.RoleTensor, "t", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	load := g.NewMemLoad("ctor", tn, []ir.Node{m})

	// Piped map with an IDENTITY body (body root IS the paramRef): inline
	// must rewire the body itself, not just the ref's users.
	vp := g.NewLambdaInputParam("ctor", "v", f)
	mp := g.NewMap("ctor", []ir.Node{load}, ir.NewLambda([]ir.Node{vp}, g.NewParamRef("ctor", vp)), nil)

	init := g.NewConst("ctor", f, 0)
	acc := g.NewLambdaInputParam("ctor", "acc", f)
	now := g.NewLambdaInputParam("ctor", "now", f)
	comb := g.NewBinOp("ctor", ir.BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	red := g.NewReduce("ctor", init, mp, ir.NewLambda([]ir.Node{acc, now}, comb))

	// A non-paramRef use of acc keeps the param alive: ClearLambdaParams
	// must keep it declared while erasing now.
	keeper := g.NewBinOp("ctor", ir.BinOpMul, acc, init)
	g.NewReturn("ctor", red, keeper)

	runInline(t, g)

	// acc↔piped (the map), now↔init VALUE (attribute unwrapped).
	if ops := ir.OperandsOf(comb); len(ops) != 2 || ops[0] != mp || ops[1] != init {
		t.Fatalf("combiner body must resolve to [piped, init-value], got %v", ops)
	}
	// The piped map's identity body rewired to the load.
	if bodies := ir.LambdaBodies(mp); len(bodies) != 1 || bodies[0] != load {
		t.Fatalf("identity body must rewire to the load, got %v", bodies)
	}
	// now is erased; acc stays declared (still used by keeper).
	alive := nodeSet(g)
	if alive[now.BasePtr()] {
		t.Fatal("the fully-inlined param `now` must be erased")
	}
	if !alive[acc.BasePtr()] {
		t.Fatal("acc must stay live (a non-paramRef use remains)")
	}
	if !useWithUser(acc, keeper) {
		t.Fatal("acc's surviving use must be the keeper binop")
	}
	if params := ir.LambdaParams(red); len(params) != 1 || len(params[0]) != 1 || params[0][0] != acc {
		t.Fatalf("the reduce's lambda must declare exactly [acc], got %v", params)
	}
	assertCleanVerify(t, g, "the inlined reduce")
}

// TestInlineIdempotence pins that a second lower+inline changes nothing.
func TestInlineIdempotence(t *testing.T) {
	g, _, _, _, _, _, _, _, _, _ := docExample(t)
	runLower(t, g)
	runInline(t, g)
	print1 := fingerprint(g)

	runLower(t, g)
	runInline(t, g)

	if got := fingerprint(g); got != print1 {
		t.Fatalf("second lower+inline changed the graph:\n--- first ---\n%s\n--- second ---\n%s", print1, got)
	}
	assertCleanVerify(t, g, "the twice-simplified graph")
}

// ---------------------------------------------------------------------------
// simplify-verify
// ---------------------------------------------------------------------------

// TestSimplifyVerifyPass pins the invariant pass: a clean lowered+inlined
// graph passes; a live IterClusterProject fails; a Map body paramRef
// resolving to a still-declared param fails.
func TestSimplifyVerifyPass(t *testing.T) {
	t.Run("clean", func(t *testing.T) {
		g, _, _, _, _, _, _, _, _, _ := docExample(t)
		runLower(t, g)
		runInline(t, g)
		if err := (simplify.VerifyPass{}).Run(nil, g); err != nil {
			t.Fatalf("clean lowered graph must pass simplify-verify: %v", err)
		}
	})

	t.Run("live project", func(t *testing.T) {
		g, _, _, _, _, _, _, _, _, _ := docExample(t)
		runLower(t, g)
		runInline(t, g)
		// Re-add a stray project (hand-corruption probe): any live
		// IterClusterProject fails the no-projects invariant.
		stray := g.NewConst("test", ir.InternScalar(ir.TF32), 0)
		g.NewIterClusterProject("test", stray, 7)
		err := (simplify.VerifyPass{}).Run(nil, g)
		if err == nil || !strings.Contains(err.Error(), "iterclusterproject") {
			t.Fatalf("a live project must fail simplify-verify, got: %v", err)
		}
	})

	t.Run("uninlined map body", func(t *testing.T) {
		g, _, _, _, _, _, _, _, _, mp := docExample(t)
		runLower(t, g) // lower only: the map body still refs its params
		err := (simplify.VerifyPass{}).Run(nil, g)
		if err == nil || !strings.Contains(err.Error(), "param") {
			t.Fatalf("a map body ref to a declared param must fail simplify-verify, got: %v", err)
		}
		_ = mp
	})
}

// ---------------------------------------------------------------------------
// Pipeline wiring (102/103/104, between dce-verify and gvn)
// ---------------------------------------------------------------------------

// TestSimplifyPipelineWiring pins the registry order and the default
// pipeline membership, and runs the doc example end-to-end through the
// default pipeline's Level-1 prefix (the full BuildPipeline cannot run on
// one graph today: the sea passes refuse materialized graphs while
// schedule demands one — the gvn test's noted split).
func TestSimplifyPipelineWiring(t *testing.T) {
	want := []struct {
		id       string
		priority int
	}{
		{"dce", 100},
		{"dce-verify", 101},
		{"simplify-lower", 102},
		{"simplify-inline", 103},
		{"simplify-verify", 104},
		{"gvn", 110},
	}
	specs := opt.Registered()
	pos := make(map[string]int)
	prio := make(map[string]int)
	for i, s := range specs {
		pos[s.ID] = i
		prio[s.ID] = s.Priority
	}
	for i := 1; i < len(want); i++ {
		if pos[want[i-1].id] >= pos[want[i].id] {
			t.Fatalf("%s must order before %s in Registered()", want[i-1].id, want[i].id)
		}
	}
	for _, w := range want {
		if got := prio[w.id]; got != w.priority {
			t.Fatalf("%s priority = %d, want %d", w.id, got, w.priority)
		}
	}

	// The default pipeline contains all three, in order.
	pipe := opt.BuildPipeline()
	seenOrder := []string{}
	for _, p := range pipe {
		switch p.ID() {
		case "dce", "dce-verify", "simplify-lower", "simplify-inline", "simplify-verify", "gvn", "gvn-verify":
			seenOrder = append(seenOrder, p.ID())
		}
	}
	wantSeq := []string{"dce", "dce-verify", "simplify-lower", "simplify-inline", "simplify-verify", "gvn", "gvn-verify"}
	if len(seenOrder) != len(wantSeq) {
		t.Fatalf("pipeline membership %v, want %v", seenOrder, wantSeq)
	}
	for i := range wantSeq {
		if seenOrder[i] != wantSeq[i] {
			t.Fatalf("pipeline order %v, want %v", seenOrder, wantSeq)
		}
	}

	// End-to-end: the doc example through the L1 prefix of the default
	// pipeline verifies, lowers, and schedules.
	g, _, _, _, _, _, loadL, loadR, _, mp := docExample(t)
	var l1 []opt.Pass
	for _, p := range opt.BuildPipeline() {
		if p.ID() == "schedule" {
			break
		}
		l1 = append(l1, p)
	}
	pm := opt.NewManager(l1)
	if err := pm.Run(g); err != nil {
		t.Fatalf("default L1 pipeline over the doc example: %v", err)
	}
	if live := nodesOfKind(g, ir.KindIterClusterProject); len(live) != 0 {
		t.Fatalf("projects survived the pipeline: %d", len(live))
	}
	if ops := ir.OperandsOf(mp); len(ops) != 4 || ops[0] != loadL || ops[1] != loadR ||
		ops[3].KindOf() != ir.KindLambda {
		t.Fatalf("map inputs must be the loads after the pipeline, got %v", ops)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize after the pipeline: %v", err)
	}
	assertCleanVerify(t, g, "the pipeline-run doc example")
}

// TestSimplifyPassContracts pins the Pass surface: IDs, Requires chaining,
// produced stage, and preserved analyses.
func TestSimplifyPassContracts(t *testing.T) {
	lower := simplify.LowerPass{}
	if lower.ID() != "simplify-lower" || lower.Requires() != nil {
		t.Fatalf("lower contract: ID=%q Requires=%v", lower.ID(), lower.Requires())
	}
	inl := simplify.InlinePass{}
	if inl.ID() != "simplify-inline" || len(inl.Requires()) != 1 || inl.Requires()[0] != "simplify-lower" {
		t.Fatalf("inline contract: ID=%q Requires=%v", inl.ID(), inl.Requires())
	}
	ver := simplify.VerifyPass{}
	if ver.ID() != "simplify-verify" || len(ver.Requires()) != 1 || ver.Requires()[0] != "simplify-inline" {
		t.Fatalf("verify contract: ID=%q Requires=%v", ver.ID(), ver.Requires())
	}
	for _, p := range []opt.Pass{lower, inl, ver} {
		if p.Produces() != opt.StageConstructed {
			t.Fatalf("%s must produce StageConstructed", p.ID())
		}
		if p.Preserved() != nil {
			t.Fatalf("%s must preserve nothing", p.ID())
		}
	}
}
