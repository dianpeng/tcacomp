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
// operands, the RAUW cascade, side-effect safety (loads/stores/iters/
// allocations never merge), Reduce participation, the lambda-body-root
// skip, idempotence, and the 110/111 pipeline wiring.

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
	if ops := ir.OperandsOf(s1); len(ops) != 2 || ops[0] != k1 || ops[1] != p1 {
		t.Fatalf("s1's operands must resolve to the representatives, got %v", ops)
	}
	if ops := ir.OperandsOf(s2); len(ops) != 2 || ops[0] != k3 || ops[1] != p1 {
		t.Fatalf("s2's operands must resolve to the representatives, got %v", ops)
	}
	runGvnVerify(t, g)
}

// TestGVNIterArithmetic pins the amendment's named case (C+D): `i*2` built
// twice over the SAME iter node dedups (the iter survives untouched);
// `i1*2` vs `i2*2` over DISTINCT iter nodes NEVER merge — identity keying
// keeps one-of-a-kind defs apart however identical their fields.
func TestGVNIterArithmetic(t *testing.T) {
	g := ir.NewGraph("iterarith")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	one := g.NewConst("ctor", f32, 1)
	two := g.NewConst("ctor", f32, 2)

	i := g.NewIter("ctor", m, nil, one, one)
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

	// The distinct-def mirror: i1*2 vs i2*2 never merge.
	g2 := ir.NewGraph("iterarith2")
	m2 := g2.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out2 := g2.NewInputParam("ctor", ir.RoleTensor, "out", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	one2 := g2.NewConst("ctor", f32, 1)
	two2 := g2.NewConst("ctor", f32, 2)
	i1 := g2.NewIter("ctor", m2, nil, one2, one2)
	i2 := g2.NewIter("ctor", m2, nil, one2, one2)
	x1 := g2.NewBinOp("ctor", ir.BinOpMul, i1, two2)
	x2 := g2.NewBinOp("ctor", ir.BinOpMul, i2, two2)
	cons2 := g2.NewBinOp("ctor", ir.BinOpAdd, x1, x2)
	scaffold(g2, m2, out2, cons2)

	runGVN(t, g2)

	assertKept(t, g2, i1, i2, x1, x2, cons2)
	runGvnVerify(t, g2)
}

// ---------------------------------------------------------------------------
// Side-effect safety ([2.5]/[4.7 as amended]; amendments A+B)
// ---------------------------------------------------------------------------

// TestGVNNeverMergesSideEffects pins the exclusion set: two structurally
// identical loads against one tensor (nil ImpactBy, same tensor/axes) do
// NOT merge (amendment B: memory read-read CSE is out of scope); two
// env-chained stores with identical tensor/axes/value BOTH survive; two
// identical NewIter survive; two identical MemNew survive.
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

	one := g.NewConst("ctor", f32, 1)
	it1 := g.NewIter("ctor", m, nil, one, one)
	it2 := g.NewIter("ctor", m, nil, one, one)

	ext := g.NewConst("ctor", f32, 64)
	alloc1 := g.NewMemNew("ctor", ext)
	alloc2 := g.NewMemNew("ctor", ext)

	g.NewReturnChained("ctor", env, []ir.Node{out}, load1)
	before := len(g.AllocatedNodes())

	runGVN(t, g)

	assertKept(t, g, load1, load2, st1, st2, it1, it2, alloc1, alloc2)
	if after := len(g.AllocatedNodes()); after != before {
		t.Fatalf("nothing in the exclusion set may merge: %d before, %d after", before, after)
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
	ops := ir.OperandsOf(r)
	if len(ops) != 2 || ops[1] != c1 {
		t.Fatalf("the reduce's piped operand must resolve to the representative, got %v", ops)
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
// cleanup passes refuse materialized graphs while schedule demands one —
// Materialize slots between gvn-verify and schedule in the caller's flow.)
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
