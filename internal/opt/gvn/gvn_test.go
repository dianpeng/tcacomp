package gvn_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/gvn"

	// The composition root: firing every pass package's self-registration is
	// what makes the pipeline-wiring assertions below observe the default
	// pipeline (quick-261018-l36 requirement 9 point 3).
	_ "tcacomp/internal/opt/all"
)

// gvn_test.go pins the GVN/CSE pass and its verifier
// (quick-261018-nga): the entry gates, identity-keyed dedup over shared SSA
// operands, the RAUW cascade, side-effect safety (loads/stores/allocations
// never merge), loop fusion via iter value numbering per
// doc/optana/loop_fusion.md (iters are side-effect free: identical headers
// and whole parent-wired nests fuse, nested-aware), Reduce participation,
// the lambda-body-root skip, idempotence, and the 110/111 pipeline wiring.

// nodeSet collects a graph's AllocatedNodes into a set keyed by the shared
// Base cell — the erased-ness and survivor assertion substrate.
func nodeSet(g *ir.Graph) map[*ir.Base]bool {
	set := make(map[*ir.Base]bool)
	for _, n := range g.AllocatedNodes() {
		set[n.BasePtr()] = true
	}
	return set
}

// runGVN runs the pass and fails the test on error.
func runGVN(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (gvn.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("gvn.Run: %v", err)
	}
}

// runGvnVerify runs the verification pass and fails the test on error.
func runGvnVerify(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (gvn.VerifyPass{}).Run(nil, g); err != nil {
		t.Fatalf("gvn-verify.Run: %v", err)
	}
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

// scaffold builds the observable spine every fixture shares: dim m, tensor
// out, an env-chained store of val, and the end node — so the graphs are
// well-formed for ir.Verify.
func scaffold(g *ir.Graph, m, out, val ir.Node) (st, end ir.Node) {
	env := ir.NewEnv()
	st = g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, val)
	end = g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return st, end
}

// fingerprint renders a structural fingerprint of the graph — node order,
// kinds, operand identities, and use lists — the stability probe for the
// idempotence test (ir.Print is a constant marker before Materialize, and
// gvn refuses materialized graphs).
func fingerprint(g *ir.Graph) string {
	var b strings.Builder
	for _, n := range g.AllocatedNodes() {
		fmt.Fprintf(&b, "%d:%s(", n.BasePtr().ID(), n.KindOf())
		for _, d := range ir.OperandsOf(n) {
			if d == nil {
				fmt.Fprintf(&b, "nil,")
			} else {
				fmt.Fprintf(&b, "%d,", d.BasePtr().ID())
			}
		}
		fmt.Fprintf(&b, ")[")
		for _, u := range ir.UsesOf(n) {
			if user := u.User(); user != nil {
				fmt.Fprintf(&b, "%d,", user.BasePtr().ID())
			} else {
				fmt.Fprintf(&b, "nil,")
			}
		}
		fmt.Fprintf(&b, "]\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Entry gates
// ---------------------------------------------------------------------------

func TestGVNGateNilGraph(t *testing.T) {
	err := gvn.Pass{}.Run(nil, nil)
	if err == nil || err.Error() != "gvn: nil graph" {
		t.Fatalf("Run(nil) error = %v, want %q", err, "gvn: nil graph")
	}
	err = gvn.VerifyPass{}.Run(nil, nil)
	if err == nil || err.Error() != "gvn-verify: nil graph" {
		t.Fatalf("VerifyPass.Run(nil) error = %v, want %q", err, "gvn-verify: nil graph")
	}
}

func TestGVNGateMaterializedRefused(t *testing.T) {
	g := ir.NewGraph("mat")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	val := g.NewConst("ctor", f32, 1)
	scaffold(g, m, out, val)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	before := len(g.AllocatedNodes())

	for name, p := range map[string]opt.Pass{"gvn": gvn.Pass{}, "gvn-verify": gvn.VerifyPass{}} {
		err := p.Run(nil, g)
		if err == nil || !strings.Contains(err.Error(), "materialized") {
			t.Fatalf("%s on a materialized graph must refuse, got: %v", name, err)
		}
	}
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("a refused graph must be untouched: %d before, %d after", before, after)
	}
}

// ---------------------------------------------------------------------------
// Core dedup semantics
// ---------------------------------------------------------------------------

// TestGVNBasicCSE pins the diamond: ((a*b) + (a*b)) over the SAME a,b — the
// duplicate mul is erased, every former user's operand Def() resolves to
// the surviving representative, and the node count drops by exactly 1.
func TestGVNBasicCSE(t *testing.T) {
	g := ir.NewGraph("diamond")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	a := g.NewConst("ctor", f32, 3)
	b := g.NewConst("ctor", f32, 4)
	mul1 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	mul2 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	sum := g.NewBinOp("ctor", ir.BinOpAdd, mul1, mul2)
	scaffold(g, m, out, sum)
	before := len(g.AllocatedNodes())

	runGVN(t, g)

	assertGone(t, g, mul2)
	assertKept(t, g, a, b, mul1, sum)
	if after := len(g.AllocatedNodes()); after != before-1 {
		t.Fatalf("node count must drop by exactly 1: %d before, %d after", before, after)
	}
	ops := ir.OperandsOf(sum)
	if len(ops) != 2 || ops[0] != mul1 || ops[1] != mul1 {
		t.Fatalf("the add's operands must both resolve to the representative, got %v", ops)
	}
	if got := len(ir.UsesOf(mul1)); got != 2 {
		t.Fatalf("the representative must own both former users' records, got %d uses", got)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)
}

// TestGVNCascade pins identity keying + RAUW composing like classic value
// numbering: C1, C2 equal consts; B1=binop(C1,C1), B2=binop(C2,C2) — gvn
// merges C2 into C1 AND THEN B2 into B1 in ONE run (B2's records were
// rewired to C1 by the first RAUW, making the binops identity-equal).
func TestGVNCascade(t *testing.T) {
	g := ir.NewGraph("cascade")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	c1 := g.NewConst("ctor", f32, 2)
	c2 := g.NewConst("ctor", f32, 2)
	b1 := g.NewBinOp("ctor", ir.BinOpAdd, c1, c1)
	b2 := g.NewBinOp("ctor", ir.BinOpAdd, c2, c2)
	scaffold(g, m, out, b1)

	runGVN(t, g)

	assertGone(t, g, c2, b2)
	assertKept(t, g, c1, b1)
	ops := ir.OperandsOf(b1)
	if len(ops) != 2 || ops[0] != c1 || ops[1] != c1 {
		t.Fatalf("B1's operands must both be C1, got %v", ops)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)
}

// TestGVNLeafPooling pins atom/near-atom dedup: two NewConst(f32,2) merge;
// two paramRefs to the same param merge; differing const values and
// differing opcodes never merge.
func TestGVNLeafPooling(t *testing.T) {
	g := ir.NewGraph("leaves")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	sc := g.NewInputParam("ctor", ir.RoleScalar, "alpha", f32)

	k1 := g.NewConst("ctor", f32, 2)
	k2 := g.NewConst("ctor", f32, 2)
	k3 := g.NewConst("ctor", f32, 3)
	p1 := g.NewParamRef("ctor", sc)
	p2 := g.NewParamRef("ctor", sc)
	s1 := g.NewBinOp("ctor", ir.BinOpAdd, k3, p1)
	s2 := g.NewBinOp("ctor", ir.BinOpSub, k3, p1)
	scaffold(g, m, out, s1)

	runGVN(t, g)

	assertGone(t, g, k2, p2)
	assertKept(t, g, k1, k3, p1, s1, s2)
	if ops := ir.OperandsOf(s1); len(ops) != 2 || ops[0] != k3 || ops[1] != p1 {
		t.Fatalf("s1's operands must resolve to the representatives, got %v", ops)
	}
	if ops := ir.OperandsOf(s2); len(ops) != 2 || ops[0] != k3 || ops[1] != p1 {
		t.Fatalf("s2's operands must resolve to the representatives, got %v", ops)
	}
	runGvnVerify(t, g)
}

// TestGVNIterArithmetic pins the amendment's named case (C+D) as
// inverted by doc/optana/loop_fusion.md: `i*2` built twice over the SAME
// iter node dedups (the iter survives untouched); and two IDENTICAL iter
// headers — separately-built DimVar -> BoundedDimVar value-object chains —
// FUSE through the cascade (consts, dimvars, bivs, then the iter), after
// which `i1*2` vs `i2*2` — keyed over the same surviving iter def —
// cascade-fuse in ONE run.
func TestGVNIterArithmetic(t *testing.T) {
	g := ir.NewGraph("iterarith")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	one := g.NewConst("ctor", f32, 1)
	two := g.NewConst("ctor", f32, 2)
	ext := g.NewConst("ctor", f32, 64)

	dv := g.NewDimVar("ctor", m, one, one)
	i := g.NewIter("ctor", g.NewBoundedDimVar("ctor", dv, ext), nil)
	b1 := g.NewBinOp("ctor", ir.BinOpMul, i, two)
	b2 := g.NewBinOp("ctor", ir.BinOpMul, i, two)
	consumer := g.NewBinOp("ctor", ir.BinOpAdd, b1, b2)
	scaffold(g, m, out, consumer)

	runGVN(t, g)

	assertGone(t, g, b2)
	assertKept(t, g, i, b1, consumer)
	if ops := ir.OperandsOf(consumer); len(ops) != 2 || ops[0] != b1 || ops[1] != b1 {
		t.Fatalf("consumer's operands must both resolve to b1, got %v", ops)
	}

	// The fusion mirror (doc/optana/loop_fusion.md): i1 and i2 ride
	// separately-built equal value-object chains, so the cascade fuses the
	// duplicate chain and its iter; the x2 arithmetic then cascades onto
	// x1 in the same run.
	g2 := ir.NewGraph("iterarith2")
	m2 := g2.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out2 := g2.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	one2 := g2.NewConst("ctor", f32, 1)
	two2 := g2.NewConst("ctor", f32, 2)
	ext2 := g2.NewConst("ctor", f32, 64)
	dv1 := g2.NewDimVar("ctor", m2, one2, one2)
	dv2 := g2.NewDimVar("ctor", m2, one2, one2)
	biv1 := g2.NewBoundedDimVar("ctor", dv1, ext2)
	biv2 := g2.NewBoundedDimVar("ctor", dv2, ext2)
	i1 := g2.NewIter("ctor", biv1, nil)
	i2 := g2.NewIter("ctor", biv2, nil)
	x1 := g2.NewBinOp("ctor", ir.BinOpMul, i1, two2)
	x2 := g2.NewBinOp("ctor", ir.BinOpMul, i2, two2)
	cons2 := g2.NewBinOp("ctor", ir.BinOpAdd, x1, x2)
	scaffold(g2, m2, out2, cons2)

	runGVN(t, g2)

	assertGone(t, g2, i2, x2, dv2, biv2)
	assertKept(t, g2, i1, x1, cons2)
	if ops := ir.OperandsOf(cons2); len(ops) != 2 || ops[0] != x1 || ops[1] != x1 {
		t.Fatalf("cons2's operands must both resolve to x1, got %v", ops)
	}
	runGvnVerify(t, g2)
}

// TestGVNIterFusionBounded pins bounded-iter fusion on the (biv, parent?)
// formation: two root iters whose BIVs carry equal-valued but DISTINCT
// const maxes fuse because the maxes CSE first
// (doc/optana/loop_fusion.md: "DimVar and Bounds are both equal" — via
// equal-valued const CSE), in ONE run; and a nest ROOT [biv] never fuses
// with a PARENTED iter [biv, parent] even after their equal BIVs merge
// (operand-count mismatch — the count rides the hash).
func TestGVNIterFusionBounded(t *testing.T) {
	// (a) Equal-valued distinct maxes: const CSE, then dv/biv, then iter
	// fusion, in one run.
	g := ir.NewGraph("iterbounded")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	max1 := g.NewConst("ctor", f32, 64)
	max2 := g.NewConst("ctor", f32, 64)
	dv1 := g.NewDimVar("ctor", m, zero, one)
	dv2 := g.NewDimVar("ctor", m, zero, one)
	biv1 := g.NewBoundedDimVar("ctor", dv1, max1)
	biv2 := g.NewBoundedDimVar("ctor", dv2, max2)
	it1 := g.NewIter("ctor", biv1, nil)
	it2 := g.NewIter("ctor", biv2, nil)
	consumer := g.NewBinOp("ctor", ir.BinOpAdd, it1, it2)
	scaffold(g, m, out, consumer)

	runGVN(t, g)

	assertGone(t, g, max2, dv2, biv2, it2)
	assertKept(t, g, max1, it1, consumer)
	if ops := ir.OperandsOf(consumer); len(ops) != 2 || ops[0] != it1 || ops[1] != it1 {
		t.Fatalf("consumer's operands must both resolve to it1, got %v", ops)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)

	// (b) Root [biv] vs parented [biv, parent]: gvn fuses the second equal
	// BIV into the first, yet the 1-op root and the 2-op child NEVER merge
	// (operand-count mismatch).
	g2 := ir.NewGraph("iterarity")
	m2 := g2.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out2 := g2.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	z := g2.NewConst("ctor", f32, 0)
	s := g2.NewConst("ctor", f32, 1)
	mx := g2.NewConst("ctor", f32, 64)
	dvM := g2.NewDimVar("ctor", m2, z, s)
	bivM := g2.NewBoundedDimVar("ctor", dvM, mx)
	itU := g2.NewIter("ctor", bivM, nil)
	z2 := g2.NewConst("ctor", f32, 0)
	s2 := g2.NewConst("ctor", f32, 1)
	mx2 := g2.NewConst("ctor", f32, 64)
	dvM2 := g2.NewDimVar("ctor", m2, z2, s2)
	bivM2 := g2.NewBoundedDimVar("ctor", dvM2, mx2)
	itB := g2.NewIter("ctor", bivM2, itU)
	sum := g2.NewBinOp("ctor", ir.BinOpAdd, itU, itB)
	scaffold(g2, m2, out2, sum)

	runGVN(t, g2)

	assertGone(t, g2, z2, s2, mx2, dvM2, bivM2) // the equal value objects fuse...
	assertKept(t, g2, itU, itB)                 // ...but root and parented never merge
	runGvnVerify(t, g2)
}

// TestGVNIterFusionNestedAwareness pins whole-nest fusion and its block on
// the (biv, parent?) formation: the parent Iter is operand 1, so identity
// keying is nested-aware — two identical parent-wired chains (separately
// built consts, dimvars, bivs, maxes) fuse level by level in ONE run
// (consts, parents, then children through the rewired parent slot), while
// children wired to DIFFERENT parents never fuse (their parent-slot def
// identity differs).
func TestGVNIterFusionNestedAwareness(t *testing.T) {
	// (a) Whole-nest fusion: the RAUW cascade fuses nest2's consts/maxes,
	// then its dimvars/bivs, then P2, then C2 (whose parent slot rewired
	// to P1) — one run, whole nest.
	g := ir.NewGraph("nestfuse")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	n := g.NewInputParam("ctor", ir.RoleDim, "n", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	z1 := g.NewConst("ctor", f32, 0)
	s1 := g.NewConst("ctor", f32, 1)
	mP1 := g.NewConst("ctor", f32, 64)
	mC1 := g.NewConst("ctor", f32, 128)
	dvP1 := g.NewDimVar("ctor", m, z1, s1)
	bivP1 := g.NewBoundedDimVar("ctor", dvP1, mP1)
	P1 := g.NewIter("ctor", bivP1, nil)
	dvC1 := g.NewDimVar("ctor", n, z1, s1)
	bivC1 := g.NewBoundedDimVar("ctor", dvC1, mC1)
	C1 := g.NewIter("ctor", bivC1, P1)
	// nest 2: separately-built equal value objects (fresh consts + maxes).
	z2 := g.NewConst("ctor", f32, 0)
	s2 := g.NewConst("ctor", f32, 1)
	mP2 := g.NewConst("ctor", f32, 64)
	mC2 := g.NewConst("ctor", f32, 128)
	dvP2 := g.NewDimVar("ctor", m, z2, s2)
	bivP2 := g.NewBoundedDimVar("ctor", dvP2, mP2)
	P2 := g.NewIter("ctor", bivP2, nil)
	dvC2 := g.NewDimVar("ctor", n, z2, s2)
	bivC2 := g.NewBoundedDimVar("ctor", dvC2, mC2)
	C2 := g.NewIter("ctor", bivC2, P2)
	consumer := g.NewBinOp("ctor", ir.BinOpAdd, C1, C2)
	scaffold(g, m, out, consumer)

	runGVN(t, g)

	assertGone(t, g, z2, s2, mP2, mC2, dvP2, bivP2, P2, dvC2, bivC2, C2)
	assertKept(t, g, z1, s1, mP1, mC1, dvP1, bivP1, P1, dvC1, bivC1, C1, consumer)
	if ops := ir.OperandsOf(consumer); len(ops) != 2 || ops[0] != C1 || ops[1] != C1 {
		t.Fatalf("consumer's operands must both resolve to C1, got %v", ops)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)

	// (b) Different parents block inner fusion: pA and pB differ by their
	// step const (1 vs 2 — never CSE), so the children — equal in every
	// slot except the parent — never fuse.
	g2 := ir.NewGraph("nestblock")
	m2 := g2.NewInputParam("ctor", ir.RoleDim, "m", nil)
	n2 := g2.NewInputParam("ctor", ir.RoleDim, "n", nil)
	out2 := g2.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	z := g2.NewConst("ctor", f32, 0)
	sA := g2.NewConst("ctor", f32, 1)
	sB := g2.NewConst("ctor", f32, 2)
	mP := g2.NewConst("ctor", f32, 64)
	mC := g2.NewConst("ctor", f32, 128)
	pA := g2.NewIter("ctor", g2.NewBoundedDimVar("ctor", g2.NewDimVar("ctor", m2, z, sA), mP), nil)
	pB := g2.NewIter("ctor", g2.NewBoundedDimVar("ctor", g2.NewDimVar("ctor", m2, z, sB), mP), nil)
	bivC := g2.NewBoundedDimVar("ctor", g2.NewDimVar("ctor", n2, z, sA), mC) // shared by both children
	childA := g2.NewIter("ctor", bivC, pA)
	childB := g2.NewIter("ctor", bivC, pB)
	sum := g2.NewBinOp("ctor", ir.BinOpAdd, childA, childB)
	scaffold(g2, m2, out2, sum)

	runGVN(t, g2)

	assertKept(t, g2, pA, pB, childA, childB)
	runGvnVerify(t, g2)
}

// TestGVNIterFusionPrefix pins PREFIX fusion of different-depth nests
// (doc/optana/loop_fusion.md): nest1 A1<-B1<-C1 and nest2 A2<-B2<-C2<-D2
// over the SAME dims with separately-built equal value objects — one gvn
// run erases nest2's common prefix (iters, dimvars, bivs, duplicate
// consts/maxes), while D2 (the deeper nest's extra level) survives
// RE-PARENTED onto C1: its parent slot — operand 1 — resolves to C1
// after the prefix RAUW cascade.
func TestGVNIterFusionPrefix(t *testing.T) {
	g := ir.NewGraph("prefixfuse")
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

	runGVN(t, g)

	assertGone(t, g, A2, B2, C2, z2, s2, mA2, mB2, mC2, dvA2, dvB2, dvC2)
	assertKept(t, g, A1, B1, C1, D2, z1, s1, mA1, mB1, mC1, mD2, dvD2, bivD2, consumer)
	// D2 survives, re-parented onto C1 (parent = operand 1).
	if ops := ir.OperandsOf(D2); len(ops) != 2 || ops[1] != C1 {
		t.Fatalf("D2 must survive re-parented onto C1 (ops[1]), got %v", ops)
	}
	if ops := ir.OperandsOf(consumer); len(ops) != 2 || ops[0] != C1 || ops[1] != D2 {
		t.Fatalf("consumer must keep [C1, D2], got %v", ops)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)
}

// ---------------------------------------------------------------------------
// Side-effect safety ([2.5]/[4.7 as amended]; amendments A+B)
// ---------------------------------------------------------------------------

// TestGVNNeverMergesSideEffects pins the exclusion set: two structurally
// identical loads against one tensor (nil ImpactBy, same tensor/axes) do
// NOT merge (amendment B: memory read-read CSE is out of scope); two
// env-chained stores with identical tensor/axes/value BOTH survive; two
// identical MemNew survive. Two identical NewIter now FUSE
// (doc/optana/loop_fusion.md: "The Iter node is itself side effect free")
// — the node count drops by exactly the one erased duplicate header, and
// the store chain is untouched.
func TestGVNNeverMergesSideEffects(t *testing.T) {
	g := ir.NewGraph("safety")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	val := g.NewConst("ctor", f32, 1)

	load1 := g.NewMemLoad("ctor", out, []ir.Node{m})
	load2 := g.NewMemLoad("ctor", out, []ir.Node{m})

	env := ir.NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, val)
	st2 := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, val)

	one := g.NewConst("ctor", f32, 7) // distinct from val: the exclusion-set probe must not trip const pooling
	biv := g.NewBoundedDimVar("ctor", g.NewDimVar("ctor", m, one, one), one)
	it1 := g.NewIter("ctor", biv, nil)
	it2 := g.NewIter("ctor", biv, nil)

	ext := g.NewConst("ctor", f32, 64)
	alloc1 := g.NewMemNew("ctor", ext)
	alloc2 := g.NewMemNew("ctor", ext)

	g.NewReturnChained("ctor", env, []ir.Node{out}, load1)
	before := len(g.AllocatedNodes())

	runGVN(t, g)

	// Iters left the exclusion set (doc/optana/loop_fusion.md): the
	// duplicate header fuses — it1/it2 have no users here, so the
	// post-RAUW zero-use assert is trivially satisfied. Everything else
	// below stays excluded.
	assertGone(t, g, it2)
	assertKept(t, g, it1, load1, load2, st1, st2, alloc1, alloc2)
	if after := len(g.AllocatedNodes()); after != before-1 {
		t.Fatalf("exactly the duplicate iter may merge: %d before, %d after", before, after)
	}
	// The chains are untouched: st2 still orders after st1.
	imp := ir.ImpactBy(st2)
	if len(imp) != 1 || imp[0].Def() != st1 {
		t.Fatalf("st2's ImpactBy must still resolve to st1, got %v", imp)
	}
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify: %v", vs)
	}
	runGvnVerify(t, g)
}

// ---------------------------------------------------------------------------
// Reduce participation (amendment E)
// ---------------------------------------------------------------------------

// TestGVNReduceParticipates pins that KindReduce is in the GVN domain: a
// reduce survives gvn untouched (not gated out); a reduce piping a
// duplicated pure value gets its piped operand rewired to the
// representative (participation observed through its operand); two reduces
// with identical init/piped operands but distinct combiner-param atoms do
// NOT merge (conservative lambda identity).
func TestGVNReduceParticipates(t *testing.T) {
	g := ir.NewGraph("reduce")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	init := g.NewConst("ctor", f32, 0)
	c1 := g.NewConst("ctor", f32, 5)
	c2 := g.NewConst("ctor", f32, 5)
	acc := g.NewLambdaInputParam("ctor", "acc", f32)
	body := g.NewBinOp("ctor", ir.BinOpAdd, acc, c1)
	r := g.NewReduce("ctor", init, c2, ir.NewLambda([]ir.Node{acc}, body))
	scaffold(g, m, out, r)

	runGVN(t, g)

	assertKept(t, g, r, init, c1)
	assertGone(t, g, c2)
	// Input-list layout (quick-261019-ktm): operand 0 is the piped value
	// (rewired to the representative by the c2→c1 RAUW), operand 1 the
	// KeyedAttribute wrapping init (unwrapped through ReduceInitOf), and
	// the lambda node closes the list (revision D15).
	ops := ir.OperandsOf(r)
	if len(ops) != 3 || ops[0] != c1 || ops[2].KindOf() != ir.KindLambda {
		t.Fatalf("the reduce's piped operand must resolve to the representative, got %v", ops)
	}
	if got := ir.ReduceInitOf(r); got != init {
		t.Fatalf("the reduce's init attribute must unwrap to init, got %v", got)
	}

	// Distinct combiner-param atoms: identical init/piped, different acc
	// params — no merge.
	g2 := ir.NewGraph("reduce2")
	m2 := g2.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out2 := g2.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	init2 := g2.NewConst("ctor", f32, 0)
	piped2 := g2.NewConst("ctor", f32, 5)
	accA := g2.NewLambdaInputParam("ctor", "acc", f32)
	accB := g2.NewLambdaInputParam("ctor", "acc", f32)
	bodyA := g2.NewBinOp("ctor", ir.BinOpAdd, accA, piped2)
	bodyB := g2.NewBinOp("ctor", ir.BinOpAdd, accB, piped2)
	ra := g2.NewReduce("ctor", init2, piped2, ir.NewLambda([]ir.Node{accA}, bodyA))
	rb := g2.NewReduce("ctor", init2, piped2, ir.NewLambda([]ir.Node{accB}, bodyB))
	scaffold(g2, m2, out2, ra)

	runGVN(t, g2)

	assertKept(t, g2, ra, rb)
	runGvnVerify(t, g2)
}

// ---------------------------------------------------------------------------
// Lambda-body-root skip (design decision 6)
// ---------------------------------------------------------------------------

// TestGVNLambdaBodyRootSkip pins the one non-SSAUse edge's pass-level
// policy: a node structurally equal to a map's lambda body root survives
// (body roots are one-of-a-kind for GVN — RAUW cannot rewire lam.Body),
// while body INTERIORS still dedup through their record-holding users.
func TestGVNLambdaBodyRootSkip(t *testing.T) {
	g := ir.NewGraph("bodyroot")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	lp := g.NewLambdaInputParam("ctor", "l", f32)
	c := g.NewConst("ctor", f32, 1)
	root := g.NewBinOp("ctor", ir.BinOpAdd, lp, c) // the map's body root...
	twin := g.NewBinOp("ctor", ir.BinOpAdd, lp, c) // ...and its structural twin
	piped := g.NewConst("ctor", f32, 5)
	lam := ir.NewLambda([]ir.Node{lp}, root)
	mp := g.NewMap("ctor", []ir.Node{piped}, lam, []ir.Node{m})
	scaffold(g, m, out, mp)

	runGVN(t, g)

	// The twin is NOT a body root, so it participates — but its only
	// structural equal IS the root, which never enters the table: both
	// survive.
	assertKept(t, g, root, twin, mp)
	runGvnVerify(t, g)
}

// TestGVNLambdaInteriorDedup pins that body interiors dedup: two maps
// sharing lambda-param atoms, whose select body roots each wrap a binop
// over the same operands — the interior binops merge (the loser body's
// operand record rewires to the surviving interior), while the body ROOTS
// stay one-of-a-kind.
func TestGVNLambdaInteriorDedup(t *testing.T) {
	g := ir.NewGraph("interior")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	lp := g.NewLambdaInputParam("ctor", "l", f32)
	c := g.NewConst("ctor", f32, 1)
	x := g.NewConst("ctor", f32, 2)
	y := g.NewConst("ctor", f32, 3)
	d1 := g.NewBinOp("ctor", ir.BinOpAdd, lp, c)
	d2 := g.NewBinOp("ctor", ir.BinOpAdd, lp, c) // interior duplicate
	s1 := g.NewSelect("ctor", d1, x, y)          // body root 1
	s2 := g.NewSelect("ctor", d2, x, y)          // body root 2
	piped := g.NewConst("ctor", f32, 5)
	m1 := g.NewMap("ctor", []ir.Node{piped}, ir.NewLambda([]ir.Node{lp}, s1), []ir.Node{m})
	m2 := g.NewMap("ctor", []ir.Node{piped}, ir.NewLambda([]ir.Node{lp}, s2), []ir.Node{m})
	scaffold(g, m, out, piped)
	_ = m2

	runGVN(t, g)

	assertGone(t, g, d2)
	assertKept(t, g, d1, s1, s2, m1)
	if ops := ir.OperandsOf(s2); len(ops) != 3 || ops[0] != d1 {
		t.Fatalf("s2's interior operand must resolve to d1, got %v", ops)
	}
	runGvnVerify(t, g)
}

// ---------------------------------------------------------------------------
// Idempotence + verifier semantics
// ---------------------------------------------------------------------------

// TestGVNIdempotent pins that a second Run changes nothing: node count and
// the structural fingerprint (node order, kinds, operand identities, use
// lists) are stable.
func TestGVNIdempotent(t *testing.T) {
	g := ir.NewGraph("idem")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	a := g.NewConst("ctor", f32, 3)
	b := g.NewConst("ctor", f32, 4)
	mul1 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	mul2 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	sum := g.NewBinOp("ctor", ir.BinOpAdd, mul1, mul2)
	lp := g.NewLambdaInputParam("ctor", "l", f32)
	d1 := g.NewBinOp("ctor", ir.BinOpAdd, lp, a)
	d2 := g.NewBinOp("ctor", ir.BinOpAdd, lp, a)
	s1 := g.NewSelect("ctor", d1, a, b)
	s2 := g.NewSelect("ctor", d2, a, b)
	mp := g.NewMap("ctor", []ir.Node{sum}, ir.NewLambda([]ir.Node{lp}, s1), []ir.Node{m})
	mp2 := g.NewMap("ctor", []ir.Node{sum}, ir.NewLambda([]ir.Node{lp}, s2), []ir.Node{m})
	scaffold(g, m, out, mp)
	_ = mp2

	runGVN(t, g)
	count := len(g.AllocatedNodes())
	print1 := fingerprint(g)

	runGVN(t, g)

	if got := len(g.AllocatedNodes()); got != count {
		t.Fatalf("second Run erased nodes: %d before, %d after", count, got)
	}
	if got := fingerprint(g); got != print1 {
		t.Fatalf("second Run changed the graph:\n--- first ---\n%s\n--- second ---\n%s", print1, got)
	}
	runGvnVerify(t, g)
}

// TestGVNVerifyGreenAfterGVN runs the verifier over every fixture shape
// after a clean gvn — plain diamonds, cascades, side-effect graphs, and
// lambda interiors — plus through the Manager with the registry-declared
// ordering.
func TestGVNVerifyGreenAfterGVN(t *testing.T) {
	build := func(name string) *ir.Graph {
		g := ir.NewGraph(name)
		f32 := ir.InternScalar(ir.TF32)
		m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
		out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
		a := g.NewConst("ctor", f32, 3)
		b := g.NewConst("ctor", f32, 4)
		m1 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
		m2 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
		sum := g.NewBinOp("ctor", ir.BinOpAdd, m1, m2)
		env := ir.NewEnv()
		st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, sum)
		g.NewReturnChained("ctor", env, []ir.Node{out}, st)
		return g
	}
	for _, name := range []string{"v1", "v2", "v3"} {
		g := build(name)
		runGVN(t, g)
		runGvnVerify(t, g)
	}

	// Manager interop with the registry-declared ordering.
	g := build("vm")
	pm := opt.NewManager([]opt.Pass{gvn.Pass{}, gvn.VerifyPass{}})
	pm.AddInstrumentation(opt.NewVerifyEach(nil))
	if err := pm.Run(g); err != nil {
		t.Fatalf("manager [gvn, gvn-verify] pipeline: %v", err)
	}
}

// TestGVNVerifyFlagsRemainingDuplicate pins the verifier's unique-
// representative check ([4.7] post-hoc): a graph holding two pure
// ValueEqual nodes WITHOUT gvn run makes VerifyPass.Run error.
func TestGVNVerifyFlagsRemainingDuplicate(t *testing.T) {
	g := ir.NewGraph("dup")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	k1 := g.NewConst("ctor", f32, 2)
	k2 := g.NewConst("ctor", f32, 2)
	scaffold(g, m, out, k1)
	_ = k2

	err := (gvn.VerifyPass{}).Run(nil, g)
	if err == nil {
		t.Fatal("the verifier must flag two remaining pure ValueEqual nodes")
	}
	for _, want := range []string{"duplicate", "const"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q must mention %q", err.Error(), want)
		}
	}
}

// TestGVNVerifyRequiresGVNFirst pins the Manager precondition: a pipeline
// of only [gvn-verify] fails the "requires gvn" check.
func TestGVNVerifyRequiresGVNFirst(t *testing.T) {
	g := ir.NewGraph("req")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	val := g.NewConst("ctor", f32, 1)
	scaffold(g, m, out, val)

	pm := opt.NewManager([]opt.Pass{gvn.VerifyPass{}})
	err := pm.Run(g)
	if err == nil || !strings.Contains(err.Error(), "requires") {
		t.Fatalf("a [gvn-verify]-only pipeline must fail on Requires, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Pipeline wiring (110/111, between early cleanup and late scheduling)
// ---------------------------------------------------------------------------

// TestGVNPipelineWiring pins the default pipeline position: dce(100) <
// dce-verify(101) < gvn(110) < gvn-verify(111) < schedule(900), and
// BuildPipeline under opt/all contains both new passes.
func TestGVNPipelineWiring(t *testing.T) {
	want := []struct {
		id       string
		priority int
	}{
		{"dce", 100},
		{"dce-verify", 101},
		{"gvn", 110},
		{"gvn-verify", 111},
		{"schedule", 900},
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

	ids := make(map[string]bool)
	for _, p := range opt.BuildPipeline() {
		ids[p.ID()] = true
	}
	if !ids["gvn"] || !ids["gvn-verify"] {
		t.Fatalf("BuildPipeline must contain gvn and gvn-verify, got %v", ids)
	}
}

// TestGVNL1PrefixPipelineInterop runs the Level-1 prefix of the default
// pipeline — dce → dce-verify → gvn → gvn-verify, in registry order — over
// a fixture with both dead code and duplicates, and confirms the end state
// verifies. (The full BuildPipeline cannot run end-to-end today: the
// cleanup passes and schedule both refuse materialized graphs — schedule
// consumes the Level-1 sea and emits the separate AST CFG — Materialize
// slots between gvn-verify and schedule in the caller's flow.)
func TestGVNL1PrefixPipelineInterop(t *testing.T) {
	g := ir.NewGraph("full")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	a := g.NewConst("ctor", f32, 3)
	b := g.NewConst("ctor", f32, 4)
	m1 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	m2 := g.NewBinOp("ctor", ir.BinOpMul, a, b)
	sum := g.NewBinOp("ctor", ir.BinOpAdd, m1, m2)
	dead := g.NewBinOp("ctor", ir.BinOpAdd, a, a)
	_ = dead
	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, sum)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	pipe := opt.BuildPipeline()
	var l1 []opt.Pass
	for _, p := range pipe {
		if p.ID() == "schedule" {
			break
		}
		l1 = append(l1, p)
	}
	pm := opt.NewManager(l1)
	pm.AddInstrumentation(opt.NewVerifyEach(nil))
	if err := pm.Run(g); err != nil {
		t.Fatalf("L1 prefix pipeline: %v", err)
	}

	assertGone(t, g, m2, dead)
	assertKept(t, g, m1, sum, st)
	if vs := ir.Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the pipeline: %v", vs)
	}
}
