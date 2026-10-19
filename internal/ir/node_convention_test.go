package ir

import (
	"reflect"
	"testing"
)

// node_convention_test.go pins the Input/ImpactBy-only convention
// (quick-261019-ktm revision, D14-D16): an IR node's inputs live ONLY in
// its Input (operand) list — data dependencies — and its ImpactBy list —
// side-effect dependencies. No node may hide SSA references in record
// fields. Machine-checked by the verifier's reflect walk.

// ---------------------------------------------------------------------------
// D16: the reflect-based convention enforcement
// ---------------------------------------------------------------------------

// hiddenRefNode is a forged node carrying a hidden SSAUse field — the
// convention violation class the verifier must flag structurally.
type hiddenRefNode struct {
	*Base
	operandList
	secret SSAUse
}

func (n *hiddenRefNode) scalarExprNode() {}

// forgeHiddenNode allocates a hiddenRefNode straight into the graph's
// registry (in-package white-box; the construction path never makes these).
func forgeHiddenNode(g *Graph, refTo Node) Node {
	b := &Base{id: g.NewID(), kind: KindConst, graph: g, loc: Loc{PassOrigin: "test"}}
	n := &hiddenRefNode{Base: b}
	var asNode Node = n
	def, _ := asNode.(Value)
	b.ssa = &SSAVar{id: b.id, node: n, define: def}
	if refTo != nil {
		n.secret = SSAUse{rec: refTo.BasePtr().ssa.addRef(n)}
	}
	g.nodes = append(g.nodes, n)
	return n
}

// TestConventionReflectEnforcement pins the machine check: a node with a
// hidden SSAUse field fails verification with the convention message, while
// every convention-compliant constructor graph verifies clean.
func TestConventionReflectEnforcement(t *testing.T) {
	g := NewGraph("convention")
	f32 := InternScalar(TF32)
	sc := g.NewInputParam("ctor", RoleScalar, "alpha", f32)
	bad := forgeHiddenNode(g, sc)
	g.NewReturn("ctor", bad)
	assertViolation(t, g, "outside the Input/ImpactBy lists")

	// An empty hidden field is still a violation by shape (the field exists).
	g2 := NewGraph("convention2")
	g2.NewInputParam("ctor", RoleScalar, "alpha", f32)
	bad2 := forgeHiddenNode(g2, nil)
	g2.NewReturn("ctor", bad2)
	assertViolation(t, g2, "outside the Input/ImpactBy lists")
}

// TestConventionNoHiddenPayloadBearers pins the sweep itself: after D14/D15
// the ONLY payload-bearing node kinds left are the ImpactBy bearers
// (memLoad/memStore/return) — every other kind's inputs ride its operand
// slots.
func TestConventionNoHiddenPayloadBearers(t *testing.T) {
	for _, n := range conventionGraph(t).AllocatedNodes() {
		_, isEffectBearer := n.(effectBearer)
		if _, isPayload := n.(payloadBearer); isPayload && !isEffectBearer {
			t.Errorf("node %d (%s) carries payload uses without being an ImpactBy bearer — the convention allows none",
				n.BasePtr().ID(), n.KindOf())
		}
	}
}

// ---------------------------------------------------------------------------
// D14: operand-only layouts (paramRef, iter, itercluster, dimvar, memstore)
// ---------------------------------------------------------------------------

// conventionGraph builds one graph touching every restructured kind.
func conventionGraph(t *testing.T) *Graph {
	t.Helper()
	g := NewGraph("conv")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	k := g.NewInputParam("ctor", RoleDim, "k", nil)
	lhs := g.NewInputParam("ctor", RoleTensor, "lhs", InternTensor(TF32, []DimRef{{Name: "m"}}))
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}, {Name: "k"}}))
	sc := g.NewInputParam("ctor", RoleScalar, "alpha", f32)

	// paramRef: the reference IS operand 0.
	ref := g.NewParamRef("ctor", sc)
	if ops := OperandsOf(ref); len(ops) != 1 || ops[0] != sc {
		t.Fatalf("paramRef operands must be [param], got %v", ops)
	}

	// iter: [dim] and [dim, bound, init, step]; TensorDimOf bounds allowed.
	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	unbounded := g.NewIter("ctor", m, nil, zero, one)
	if ops := OperandsOf(unbounded); len(ops) != 3 || ops[0] != m || ops[1] != zero || ops[2] != one {
		t.Fatalf("unbounded iter operands must be [dim, init, step], got %v", ops)
	}
	tdim := g.NewTensorDimOf("ctor", lhs, 0)
	bounded := g.NewIter("ctor", m, tdim, zero, one)
	if ops := OperandsOf(bounded); len(ops) != 4 || ops[0] != m || ops[1] != tdim || ops[2] != zero || ops[3] != one {
		t.Fatalf("bounded iter operands must be [dim, bound, init, step], got %v", ops)
	}

	// dimvar: [dim, init, step].
	dv := g.NewDimVar("ctor", m, zero, one)
	if ops := OperandsOf(dv); len(ops) != 3 || ops[0] != m || ops[1] != zero || ops[2] != one {
		t.Fatalf("dimvar operands must be [dim, init, step], got %v", ops)
	}
	_ = dv

	// memstore: [value, tensor, axes...].
	st := g.NewMemStore("ctor", out, []Node{m, k}, one)
	if ops := OperandsOf(st); len(ops) != 4 || ops[0] != one || ops[1] != out || ops[2] != m || ops[3] != k {
		t.Fatalf("memstore operands must be [value, tensor, axes...], got %v", ops)
	}

	// itercluster: dims lead the input list; inputs follow.
	load := g.NewMemLoad("ctor", lhs, []Node{m})
	cl := g.NewIterCluster("ctor", []Node{m, k}, load, bounded)
	if ops := OperandsOf(cl); len(ops) != 4 || ops[0] != m || ops[1] != k || ops[2] != load || ops[3] != bounded {
		t.Fatalf("cluster operands must be [dims..., inputs...], got %v", ops)
	}
	if dims := ClusterDims(cl); len(dims) != 2 || dims[0] != m || dims[1] != k {
		t.Fatalf("ClusterDims must read the leading dim operands, got %v", dims)
	}
	if ins := ClusterInputs(cl); len(ins) != 2 || ins[0] != load || ins[1] != bounded {
		t.Fatalf("ClusterInputs must read the non-dim operands, got %v", ins)
	}

	g.NewReturn("ctor", st, cl, ref)
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("convention graph must verify clean, got:\n%s", violationsText(vs))
	}
	return g
}

// TestConventionShapeChecks pins the new fail-closed shapes: iter/dimvar
// bind dim InputParams at operand 0, paramRefs bind InputParams, cluster
// dims lead the input list.
func TestConventionShapeChecks(t *testing.T) {
	f32 := InternScalar(TF32)

	t.Run("iter rejects non-dim operand 0", func(t *testing.T) {
		g := NewGraph("s1")
		zero := g.NewConst("ctor", f32, 0)
		bad := g.NewIter("ctor", zero, nil, zero, zero)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "iter operand 0 must be a dim InputParam")
	})

	t.Run("dimvar rejects non-dim operand 0", func(t *testing.T) {
		g := NewGraph("s2")
		zero := g.NewConst("ctor", f32, 0)
		bad := g.NewDimVar("ctor", zero, zero, zero)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "dimvar operand 0 must be a dim InputParam")
	})

	t.Run("paramref rejects non-param operand", func(t *testing.T) {
		g := NewGraph("s3")
		bad := g.NewParamRef("ctor", g.NewConst("ctor", f32, 0))
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "param operand 0 must be an InputParam node")
	})

	t.Run("cluster rejects dim after input", func(t *testing.T) {
		g := NewGraph("s4")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		lhs := g.NewInputParam("ctor", RoleTensor, "lhs", InternTensor(TF32, []DimRef{{Name: "m"}}))
		load := g.NewMemLoad("ctor", lhs, []Node{m})
		bad := g.NewNode(KindIterCluster, "ctor", load, m) // input before dim
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "cluster dims must lead the input list")
	})
}

// ---------------------------------------------------------------------------
// D15: the lambda node — params and body ride a KindLambda Input list
// ---------------------------------------------------------------------------

// TestConventionLambdaNode pins the lambda layout: Map/Reduce carry their
// lambda as the TRAILING operand; the lambda node's operands are
// [params..., body]; the derived read faces (LambdaOf, LambdaBodies,
// LambdaParams) agree; the map's value-operand count still matches the
// lambda's parameter count.
func TestConventionLambdaNode(t *testing.T) {
	g := NewGraph("lambda-node")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))

	v := g.NewLambdaInputParam("ctor", "v", f32)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), g.NewConst("ctor", f32, 2))
	val := g.NewConst("ctor", f32, 1)
	mp := g.NewMap("ctor", []Node{val}, NewLambda([]Node{v}, body), []Node{m})

	ops := OperandsOf(mp)
	if len(ops) != 3 {
		t.Fatalf("map operands must be [value, As, lambda], got %v", ops)
	}
	lam := LambdaOf(mp)
	if lam == nil || lam.KindOf() != KindLambda || ops[2] != lam {
		t.Fatalf("the map's trailing operand must be its lambda node, got %v", ops[2])
	}
	if lops := OperandsOf(lam); len(lops) != 2 || lops[0] != v || lops[1] != body {
		t.Fatalf("lambda operands must be [params..., body], got %v", lops)
	}
	if bodies := LambdaBodies(mp); len(bodies) != 1 || bodies[0] != body {
		t.Fatalf("LambdaBodies must derive the body, got %v", bodies)
	}
	if ps := LambdaParams(mp); len(ps) != 1 || len(ps[0]) != 1 || ps[0][0] != v {
		t.Fatalf("LambdaParams must derive [v], got %v", ps)
	}

	acc := g.NewLambdaInputParam("ctor", "acc", f32)
	now := g.NewLambdaInputParam("ctor", "now", f32)
	comb := g.NewBinOp("ctor", BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	init := g.NewConst("ctor", f32, 0)
	red := g.NewReduce("ctor", init, mp, NewLambda([]Node{acc, now}, comb))
	rops := OperandsOf(red)
	if len(rops) != 3 || rops[0] != mp || rops[1].KindOf() != KindKeyedAttribute || rops[2].KindOf() != KindLambda {
		t.Fatalf("reduce operands must be [piped, attr, lambda], got %v", rops)
	}
	if ps := LambdaParams(red); len(ps) != 1 || len(ps[0]) != 2 || ps[0][0] != acc || ps[0][1] != now {
		t.Fatalf("reduce lambda params must derive [acc, now], got %v", ps)
	}

	st := g.NewMemStore("ctor", out, []Node{m}, red)
	g.NewReturn("ctor", st)
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("lambda-node graph must verify clean, got:\n%s", violationsText(vs))
	}

	// The convention reflect-walk covers the lambda node too: no hidden
	// fields anywhere (implicitly checked by Verify above).
}

// TestConventionLambdaValueIdentity pins GVN participation of the lambda
// node: two lambdas over the same param atoms and body node are one value;
// distinct bodies keep them apart (identity keying over the Input list).
func TestConventionLambdaValueIdentity(t *testing.T) {
	g := NewGraph("lambda-value")
	f32 := InternScalar(TF32)
	lp := g.NewLambdaInputParam("ctor", "l", f32)
	c := g.NewConst("ctor", f32, 1)
	body := g.NewBinOp("ctor", BinOpAdd, lp, c)

	lam1 := asValue(t, g.NewNode(KindLambda, "ctor", lp, body))
	lam2 := asValue(t, g.NewNode(KindLambda, "ctor", lp, body))
	if !lam1.ValueEqual(lam2, pureAll{}) || lam1.ValueHash(pureAll{}) != lam2.ValueHash(pureAll{}) {
		t.Fatal("lambdas over the same param atoms and body node must be one value")
	}
	body2 := g.NewBinOp("ctor", BinOpAdd, lp, c)
	lam3 := asValue(t, g.NewNode(KindLambda, "ctor", lp, body2))
	if lam1.ValueEqual(lam3, pureAll{}) {
		t.Fatal("lambdas over distinct body nodes must compare unequal (identity keying)")
	}
}

// TestConventionChokePoints pins the reimplemented mutation choke points
// over the lambda node's operand storage: SetLambdaBody rewires the body
// operand; ClearLambdaParams drops dead param operand slots and keeps
// params with live uses.
func TestConventionChokePoints(t *testing.T) {
	g := NewGraph("choke")
	f32 := InternScalar(TF32)
	m := g.NewInputParam("ctor", RoleDim, "m", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	load := g.NewMemLoad("ctor", out, []Node{m})

	v := g.NewLambdaInputParam("ctor", "v", f32)
	ref := g.NewParamRef("ctor", v)
	mp := g.NewMap("ctor", []Node{load}, NewLambda([]Node{v}, ref), nil)
	g.NewReturn("ctor", mp)
	mctx := &MutContext{Pass: "test"}

	// SetLambdaBody rewires the lambda's body operand.
	repl := g.NewConst("ctor", f32, 7)
	if err := mctx.SetLambdaBody(mp, repl); err != nil {
		t.Fatalf("SetLambdaBody: %v", err)
	}
	if bodies := LambdaBodies(mp); len(bodies) != 1 || bodies[0] != repl {
		t.Fatalf("SetLambdaBody must rewire the body operand, got %v", bodies)
	}
	if uses := UsesOf(ref); len(uses) != 0 {
		t.Fatalf("the old body must lose its lambda use, got %d", len(uses))
	}

	// ClearLambdaParams drops the dead param declaration; a param with a
	// live external use survives.
	keeper := g.NewBinOp("ctor", BinOpMul, v, repl)
	keep := func(p Node) bool {
		for _, u := range UsesOf(p) {
			if user := u.User(); user != nil && user.KindOf() != KindLambda {
				return true
			}
		}
		return false
	}
	if err := mctx.ClearLambdaParams(mp, keep); err != nil {
		t.Fatalf("ClearLambdaParams: %v", err)
	}
	if ps := LambdaParams(mp); len(ps) != 1 || len(ps[0]) != 1 || ps[0][0] != v {
		t.Fatalf("the live-used param must stay declared, got %v", ps)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("choke-point graph must verify clean, got:\n%s", violationsText(vs))
	}
	_ = keeper
}

// TestConventionReflectCoversAllKinds sanity-pins the type-level sweep: the
// forbidden-type table sees the SSA-carrying types (used by the verifier's
// convention walk).
func TestConventionReflectCoversAllKinds(t *testing.T) {
	if !hiddenRefFieldType(reflect.TypeOf(SSAUse{}), false) {
		t.Fatal("SSAUse fields must classify as hidden references")
	}
	if !hiddenRefFieldType(reflect.TypeOf([]SSAUse{}), false) {
		t.Fatal("[]SSAUse fields must classify as hidden on non-effect bearers")
	}
	if hiddenRefFieldType(reflect.TypeOf([]SSAUse{}), true) {
		t.Fatal("[]SSAUse fields are the ImpactBy list on effect bearers — allowed")
	}
	if !hiddenRefFieldType(reflect.TypeOf(Node(nil)), false) {
		t.Fatal("Node fields must classify as hidden references")
	}
	if hiddenRefFieldType(reflect.TypeOf(Lambda{}), true) {
		t.Fatal("Lambda records must classify as hidden references even on effect bearers")
	}
}
