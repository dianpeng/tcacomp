package ir

import (
	"strings"
	"testing"
)

// node_iterfmt_test.go pins the Input-list restructure (quick-261019-ktm):
// the five new node kinds (IterClusterProject, KeyedAttribute, As,
// InductionVariable, TensorDimOf), the Map/Reduce/IterCluster operand-list
// restructure (D1-D5), the fail-closed verifier shapes, the printer's
// unwrap paths, and the GVN identity coverage for idx/key content fields.

// iterfmtGraph is the shared fixture: dims a/b/c, rank-2 tensors lhs/rhs,
// and a scalar param — the doc/optana/lower_itercluster.md myfoo shape's
// symbol set.
func iterfmtGraph(t *testing.T) (g *Graph, a, b, c, lhs, rhs, alpha Node) {
	t.Helper()
	g = NewGraph("iterfmt")
	a = g.NewInputParam("ctor", RoleDim, "a", nil)
	b = g.NewInputParam("ctor", RoleDim, "b", nil)
	c = g.NewInputParam("ctor", RoleDim, "c", nil)
	lhs = g.NewInputParam("ctor", RoleTensor, "lhs", InternTensor(TF32, []DimRef{{Name: "a"}, {Name: "b"}}))
	rhs = g.NewInputParam("ctor", RoleTensor, "rhs", InternTensor(TF32, []DimRef{{Name: "a"}, {Name: "c"}}))
	alpha = g.NewInputParam("ctor", RoleScalar, "alpha", InternScalar(TF32))
	return g, a, b, c, lhs, rhs, alpha
}

// TestIterfmtNodesConstructAndVerify pins that each new node kind
// constructs through its constructor with the documented operand shape and
// verifies clean when rooted through the end node: project over a cluster
// (D4), attribute over a value (D2), As over dims (D1), TensorDimOf over a
// tensor + interned TU32 axis const (D6), InductionVariable over an iter.
func TestIterfmtNodesConstructAndVerify(t *testing.T) {
	g, a, b, c, lhs, rhs, _ := iterfmtGraph(t)

	loadL := g.NewMemLoad("ctor", lhs, []Node{a, b})
	loadR := g.NewMemLoad("ctor", rhs, []Node{a, c})
	cluster := g.NewIterCluster("ctor", []Node{a, b, c}, loadL, loadR)
	// Dims LEAD the input list, inputs follow (Input/ImpactBy-only
	// convention, quick-261019-ktm revision).
	if ops := OperandsOf(cluster); len(ops) != 5 || ops[0] != a || ops[1] != b || ops[2] != c ||
		ops[3] != loadL || ops[4] != loadR {
		t.Fatalf("cluster operands must be [dims..., inputs...] in order, got %v", ops)
	}
	if dims := ClusterDims(cluster); len(dims) != 3 || dims[0] != a || dims[1] != b || dims[2] != c {
		t.Fatalf("ClusterDims must read the leading dims, got %v", dims)
	}
	if ins := ClusterInputs(cluster); len(ins) != 2 || ins[0] != loadL || ins[1] != loadR {
		t.Fatalf("ClusterInputs must read the inputs in order, got %v", ins)
	}

	p0 := g.NewIterClusterProject("ctor", cluster, 0)
	p1 := g.NewIterClusterProject("ctor", cluster, 1)
	if ops := OperandsOf(p0); len(ops) != 1 || ops[0] != cluster {
		t.Fatalf("project operand 0 must be the cluster, got %v", ops)
	}

	f32 := InternScalar(TF32)
	init := g.NewConst("ctor", f32, 0)
	attr := g.NewKeyedAttribute("ctor", "init", init)
	if ops := OperandsOf(attr); len(ops) != 1 || ops[0] != init {
		t.Fatalf("attribute operand 0 must wrap the value, got %v", ops)
	}

	as := g.NewAs("ctor", a, b, c)
	if ops := OperandsOf(as); len(ops) != 3 || ops[0] != a || ops[1] != b || ops[2] != c {
		t.Fatalf("as operands must be the dims in order, got %v", ops)
	}

	tdim := g.NewTensorDimOf("ctor", lhs, 1)
	ops := OperandsOf(tdim)
	if len(ops) != 2 || ops[0] != lhs {
		t.Fatalf("tensordimof operand 0 must be the tensor, got %v", ops)
	}
	if ax, ok := ops[1].(*constNode); !ok || ax.typ != InternScalar(TU32) || ax.val != 1 {
		t.Fatalf("tensordimof operand 1 must be the TU32 axis const 1, got %v", ops[1])
	}

	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	iter := g.NewIter("ctor", a, tdim, zero, one)
	iv := g.NewInductionVariable("ctor", iter)
	if ops := OperandsOf(iv); len(ops) != 1 || ops[0] != iter {
		t.Fatalf("inductionvariable operand 0 must be the iter, got %v", ops)
	}

	// Everything stays live through the end node: loads as returned values,
	// the project/attr/as/tdim/iv chain through the cluster's second input
	// replaced by the iv expression. Verify must report ZERO violations.
	g.NewReturn("ctor", loadL, p0, p1, attr, as, iv)
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("iterfmt graph must verify clean, got:\n%s", violationsText(vs))
	}
}

// TestIterfmtVerifyShapes pins the fail-closed shape checks (Task 1 action
// 8), mirroring TestDimvarVerifyShapes: each malformed shape reports its
// specific verifier message.
func TestIterfmtVerifyShapes(t *testing.T) {
	f32 := InternScalar(TF32)

	t.Run("project rejects non-cluster operand", func(t *testing.T) {
		g := NewGraph("s1")
		bad := g.NewIterClusterProject("ctor", g.NewConst("ctor", f32, 0), 0)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "iterclusterproject operand 0 must be an itercluster")
	})

	t.Run("inductionvariable rejects non-iter operand", func(t *testing.T) {
		g := NewGraph("s2")
		bad := g.NewInductionVariable("ctor", g.NewConst("ctor", f32, 0))
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "inductionvariable operand 0 must be an iter")
	})

	t.Run("tensordimof rejects scalar tensor operand", func(t *testing.T) {
		g := NewGraph("s3")
		sc := g.NewInputParam("ctor", RoleScalar, "alpha", f32)
		bad := g.NewTensorDimOf("ctor", sc, 0)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "tensor operand must be a RoleTensor InputParam node")
	})

	t.Run("tensordimof rejects non-const axis", func(t *testing.T) {
		g := NewGraph("s4")
		lhs := g.NewInputParam("ctor", RoleTensor, "lhs", InternTensor(TF32, []DimRef{{Name: "a"}}))
		a := g.NewInputParam("ctor", RoleDim, "a", nil)
		notConst := g.NewParamRef("ctor", a)
		bad := g.NewNode(KindTensorDimOf, "ctor", lhs, notConst)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "tensordimof operand 1 must be a const axis index")
	})

	t.Run("as rejects non-dim operand", func(t *testing.T) {
		g := NewGraph("s5")
		sc := g.NewInputParam("ctor", RoleScalar, "alpha", f32)
		bad := g.NewAs("ctor", sc)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "as operand 0 must be a dim InputParam")
	})

	t.Run("reduce rejects bare init operand", func(t *testing.T) {
		g := NewGraph("s6")
		piped := g.NewConst("ctor", f32, 5)
		bare := g.NewConst("ctor", f32, 0)
		bad := g.NewNode(KindReduce, "ctor", piped, bare)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "reduce operand 1 must be a KeyedAttribute (init)")
	})

	t.Run("map rejects as in non-last position", func(t *testing.T) {
		g := NewGraph("s7")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		val := g.NewConst("ctor", f32, 1)
		as := g.NewAs("ctor", m)
		bad := g.NewNode(KindMap, "ctor", as, val) // As first, value last
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "as must be the last input")
	})

	t.Run("map rejects two as nodes", func(t *testing.T) {
		g := NewGraph("s8")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		as1 := g.NewAs("ctor", m)
		as2 := g.NewAs("ctor", m)
		bad := g.NewNode(KindMap, "ctor", as1, as2)
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "as must be the last input")
	})

	t.Run("map input count must equal lambda parameter count", func(t *testing.T) {
		g := NewGraph("s9")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		v := g.NewLambdaInputParam("ctor", "v", f32)
		val := g.NewConst("ctor", f32, 1)
		bad := g.NewMap("ctor", nil, NewLambda([]Node{v}, val), []Node{m}) // 0 inputs, 1 param
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "map input count must equal lambda parameter count")
	})
}

// TestIterfmtInputListAccessors pins D1/D2's read faces: AsOf scans the
// operand list backwards for the trailing As, ReduceInitOf unwraps operand
// 1's KeyedAttribute, and both report nil on legacy shapes (no As, wrong
// operand kind).
func TestIterfmtInputListAccessors(t *testing.T) {
	g, a, b, c, _, _, _ := iterfmtGraph(t)
	f32 := InternScalar(TF32)

	// Map: [values..., As] — AsOf finds the trailing As; a map without one
	// reports nil.
	v := g.NewLambdaInputParam("ctor", "v", f32)
	body := g.NewParamRef("ctor", v)
	val := g.NewConst("ctor", f32, 1)
	mp := g.NewMap("ctor", []Node{val}, NewLambda([]Node{v}, body), []Node{a, b, c})
	as := AsOf(mp)
	if as == nil || as.KindOf() != KindAs {
		t.Fatal("AsOf must find the map's trailing As node")
	}
	if dims := OperandsOf(as); len(dims) != 3 || dims[0] != a || dims[1] != b || dims[2] != c {
		t.Fatalf("As operands must be the as-order dims, got %v", dims)
	}
	plain := g.NewMap("ctor", []Node{val}, NewLambda([]Node{v}, body), nil)
	if AsOf(plain) != nil {
		t.Fatal("AsOf must report nil for a map without an as-clause")
	}

	// Reduce: [piped, KeyedAttribute("init", init)] — ReduceInitOf unwraps
	// the attribute's value operand.
	init := g.NewConst("ctor", f32, 0)
	acc := g.NewLambdaInputParam("ctor", "acc", f32)
	comb := g.NewParamRef("ctor", acc)
	red := g.NewReduce("ctor", init, val, NewLambda([]Node{acc}, comb))
	if got := ReduceInitOf(red); got != init {
		t.Fatalf("ReduceInitOf must unwrap the init attribute, got %v", got)
	}
	if ops := OperandsOf(red); len(ops) != 3 || ops[0] != val || ops[2].KindOf() != KindLambda {
		t.Fatalf("reduce operands must be [piped, init attribute, lambda], got %v", ops)
	}
	if ReduceInitOf(mp) != nil {
		t.Fatal("ReduceInitOf must report nil for a non-reduce node")
	}

	// The constructor graphs verify clean end to end.
	g.NewReturn("ctor", mp, red, plain)
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("accessor fixture must verify clean, got:\n%s", violationsText(vs))
	}
}

// TestIterfmtPrintUnwrap pins the printer's unwrap paths (D11/D12): the
// reduce stage text keeps the pre-change spelling with piped at operand 0
// and init rendered through ReduceInitOf; the map's as-clause renders from
// the As node's operands; a TensorDimOf-bounded iter renders the bound's
// value instead of the "?" paramNameOf fallback.
func TestIterfmtPrintUnwrap(t *testing.T) {
	g, a, b, _, lhs, _, _ := iterfmtGraph(t)
	f32 := InternScalar(TF32)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "a"}, {Name: "b"}}))

	v := g.NewLambdaInputParam("ctor", "v", f32)
	acc := g.NewLambdaInputParam("ctor", "acc", f32)
	now := g.NewLambdaInputParam("ctor", "now", f32)
	val := g.NewConst("ctor", f32, 1)
	body := g.NewBinOp("ctor", BinOpMul, g.NewParamRef("ctor", v), val)
	comb := g.NewBinOp("ctor", BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	mp := g.NewMap("ctor", []Node{val}, NewLambda([]Node{v}, body), []Node{a, b})
	red := g.NewReduce("ctor", g.NewConst("ctor", f32, 0), mp,
		NewLambda([]Node{acc, now}, comb))
	store := g.NewMemStore("ctor", out, []Node{a, b}, red)
	g.NewReturn("ctor", store)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	p := Print(g)
	for _, want := range []string{
		"map(1.0, lambda(v): v * 1.0) as (@a, @b)",
		"reduce(init=0.0, %0, lambda(acc, now): acc + now)",
	} {
		if !strings.Contains(p, want) {
			t.Fatalf("print must contain %q, got:\n%s", want, p)
		}
	}

	// A TensorDimOf-bounded iter renders @a/tensordimof(lhs, 0) — no "?".
	tdim := g.NewTensorDimOf("ctor", lhs, 0)
	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	iter := g.NewIter("ctor", a, tdim, zero, one)
	if got, want := (&renderer{}).value(iter), "@a/tensordimof(lhs, 0)"; got != want {
		t.Fatalf("tensordimof-bounded iter renders %q, want %q", got, want)
	}
	// A param-bounded iter keeps the legacy @a/@b spelling, and the nil
	// bound stays bare.
	bb := g.NewInputParam("ctor", RoleDim, "bbound", nil)
	if got, want := (&renderer{}).value(g.NewIter("ctor", a, bb, zero, one)), "@a/bbound"; got != want {
		t.Fatalf("param-bounded iter renders %q, want %q", got, want)
	}
	if got, want := (&renderer{}).value(g.NewIter("ctor", a, nil, zero, one)), "@a"; got != want {
		t.Fatalf("unbounded iter renders %q, want %q", got, want)
	}
}

// TestIterfmtValueIdentity pins D13's GVN identity coverage: the project's
// idx and the attribute's key are content fields (mixed/compared), while
// the operand-free new kinds carry identity purely through operands — two
// InductionVariables over the SAME iter are one value.
func TestIterfmtValueIdentity(t *testing.T) {
	g, a, b, _, lhs, _, _ := iterfmtGraph(t)
	f32 := InternScalar(TF32)

	// Two MemLoad inputs give the cluster a 2-slot input list.
	loadL := g.NewMemLoad("ctor", lhs, []Node{a})
	loadR := g.NewMemLoad("ctor", lhs, []Node{b})
	cluster := g.NewIterCluster("ctor", []Node{a, b}, loadL, loadR)

	p0 := asValue(t, g.NewIterClusterProject("ctor", cluster, 0))
	p0b := asValue(t, g.NewIterClusterProject("ctor", cluster, 0))
	p1 := asValue(t, g.NewIterClusterProject("ctor", cluster, 1))
	if !p0.ValueEqual(p0b, pureAll{}) || p0.ValueHash(pureAll{}) != p0b.ValueHash(pureAll{}) {
		t.Fatal("projects over the same cluster with the same idx must be one value")
	}
	if p0.ValueEqual(p1, pureAll{}) {
		t.Fatal("projects with idx 0 vs 1 must compare unequal (idx is a content field)")
	}
	if p0.ValueHash(pureAll{}) == p1.ValueHash(pureAll{}) {
		t.Fatal("projects with idx 0 vs 1 must hash differently")
	}

	// KeyedAttribute keys are content: same key + same value ⇒ one value;
	// differing keys ⇒ unequal and differently hashed.
	init := g.NewConst("ctor", f32, 0)
	k1 := asValue(t, g.NewKeyedAttribute("ctor", "init", init))
	k1b := asValue(t, g.NewKeyedAttribute("ctor", "init", init))
	k2 := asValue(t, g.NewKeyedAttribute("ctor", "other", init))
	if !k1.ValueEqual(k1b, pureAll{}) || k1.ValueHash(pureAll{}) != k1b.ValueHash(pureAll{}) {
		t.Fatal("attributes with the same key and value must be one value")
	}
	if k1.ValueEqual(k2, pureAll{}) {
		t.Fatal("attributes with keys init vs other must compare unequal")
	}
	if k1.ValueHash(pureAll{}) == k2.ValueHash(pureAll{}) {
		t.Fatal("attributes with differing keys must hash differently")
	}

	// InductionVariables dedupe by operand identity: same iter ⇒ one value.
	tdim := g.NewTensorDimOf("ctor", lhs, 0)
	iter := g.NewIter("ctor", a, tdim, g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
	iv1 := asValue(t, g.NewInductionVariable("ctor", iter))
	iv2 := asValue(t, g.NewInductionVariable("ctor", iter))
	if !iv1.ValueEqual(iv2, pureAll{}) || iv1.ValueHash(pureAll{}) != iv2.ValueHash(pureAll{}) {
		t.Fatal("two InductionVariables over the same iter must be one value")
	}

	// TensorDimOf identity rides its operands (amendment D): two
	// NewTensorDimOf calls intern distinct axis consts, so the pair compares
	// unequal; explicitly sharing the axis-const node makes them one value.
	t1 := asValue(t, g.NewTensorDimOf("ctor", lhs, 0))
	t2 := asValue(t, g.NewTensorDimOf("ctor", lhs, 0))
	if t1.ValueEqual(t2, pureAll{}) {
		t.Fatal("tensordimofs with distinct axis-const nodes must compare unequal (identity keying)")
	}
	shared := asValue(t, g.NewTensorDimOf("ctor", lhs, 0))
	t3 := asValue(t, g.NewNode(KindTensorDimOf, "ctor", lhs, OperandsOf(shared)[1]))
	if !shared.ValueEqual(t3, pureAll{}) || shared.ValueHash(pureAll{}) != t3.ValueHash(pureAll{}) {
		t.Fatal("tensordimofs sharing the tensor and axis-const nodes must be one value")
	}
}

// violationsText renders a violation slice for failure messages.
func violationsText(vs []Violation) string {
	var b strings.Builder
	for _, v := range vs {
		b.WriteString(v.String())
		b.WriteString("\n")
	}
	return b.String()
}
