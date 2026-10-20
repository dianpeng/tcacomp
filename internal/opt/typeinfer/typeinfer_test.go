package typeinfer_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/simplify"
	"tcacomp/internal/opt/typeinfer"

	// The composition root: firing every pass package's self-registration
	// is what makes the pipeline-wiring assertions below observe the
	// default pipeline.
	_ "tcacomp/internal/opt/all"
)

// typeinfer_test.go pins the type inference pass (quick-261019-tu3,
// doc/optana/type_inf.md rules (1)-(5) extended to the 31-kind catalog):
// per-kind typing rules, idempotency (types AND LastTouchedBy stable),
// fail-closed errors naming the offending node, the matmul_scale oracle
// through the simplify+typeinfer pipeline, and the 105 registration.

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// mustRun runs the pass and fails the test on any error.
func mustRun(t *testing.T, g *ir.Graph) {
	t.Helper()
	if err := (typeinfer.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("typeinfer: %v", err)
	}
}

// runTI returns the pass error (for fail-closed cases).
func runTI(g *ir.Graph) error {
	return (typeinfer.Pass{}).Run(nil, g)
}

// f32T / u32T / i32T / boolT / f16T are the interned scalar shortcuts.
func f32T() ir.Type  { return ir.InternScalar(ir.TF32) }
func u32T() ir.Type  { return ir.InternScalar(ir.TU32) }
func i32T() ir.Type  { return ir.InternScalar(ir.TI32) }
func boolT() ir.Type { return ir.InternScalar(ir.TBool) }

// tensorT builds an interned tensor type over symbolic dims.
func tensorT(elem ir.ElemType, names ...string) ir.Type {
	refs := make([]ir.DimRef, len(names))
	for i, n := range names {
		refs[i] = ir.DimRef{Name: n}
	}
	return ir.InternTensor(elem, refs)
}

// wantType fails unless got is non-nil and Equal to want.
func wantType(t *testing.T, n ir.Node, what string, want ir.Type) {
	t.Helper()
	got := ir.TypeOf(n)
	if got == nil {
		t.Fatalf("%s: TypeOf = nil, want %v", what, want)
	}
	if !got.Equal(want) {
		t.Fatalf("%s: TypeOf = %v, want %v", what, got, want)
	}
}

// wantErrNames fails unless err is non-nil, mentions node id, and contains
// every keyword (fail-closed, node-naming).
func wantErrNames(t *testing.T, err error, id uint64, keywords ...string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want an error naming node %d, got nil", id)
	}
	msg := err.Error()
	if !strings.Contains(msg, fmt.Sprintf("node %d", id)) {
		t.Fatalf("error must name node %d, got: %v", id, err)
	}
	for _, kw := range keywords {
		if !strings.Contains(msg, kw) {
			t.Fatalf("error must mention %q, got: %v", kw, err)
		}
	}
}

// ---------------------------------------------------------------------------
// Expr rules (spec (1): scalar-only semantics; syntax.md §6.2 classes)
// ---------------------------------------------------------------------------

// TestExprScalarRules pins the scalar-expression family: const payload,
// param passthrough, the operator classes, select/cast/call.
func TestExprScalarRules(t *testing.T) {
	g := ir.NewGraph("expr")
	f32c := g.NewConst("t", f32T(), 1.5)
	i32c := g.NewConst("t", i32T(), 3)
	boolc := g.NewConst("t", boolT(), 1)

	sp := g.NewInputParam("t", ir.RoleScalar, "s", f32T())
	ref := g.NewParamRef("t", sp)

	// inputparam roles: dim -> u32 (computed), tensor -> its tensor type.
	dp := g.NewInputParam("t", ir.RoleDim, "d", nil)
	tp := g.NewInputParam("t", ir.RoleTensor, "t", tensorT(ir.TF32, "a", "b"))

	add := g.NewBinOp("t", ir.BinOpAdd, f32c, g.NewConst("t", f32T(), 2))
	lt := g.NewBinOp("t", ir.BinOpLT, add, f32c)
	lnot := g.NewUnaryOp("t", ir.UnaryOpLNot, boolc)
	bnot := g.NewUnaryOp("t", ir.UnaryOpNot, i32c)
	sel := g.NewSelect("t", boolc, f32c, add)
	cast := g.NewCast("t", ir.InternScalar(ir.TF16), f32c)
	call := g.NewCall("t", "expf", f32c)

	mustRun(t, g)

	wantType(t, f32c, "const f32", f32T())
	wantType(t, ref, "param ref over typed scalar param", f32T())
	wantType(t, sp, "scalar inputparam", f32T())
	wantType(t, dp, "dim inputparam", u32T())
	wantType(t, tp, "tensor inputparam", tensorT(ir.TF32, "a", "b"))
	wantType(t, add, "add f32+f32", f32T())
	wantType(t, lt, "lt f32,f32", boolT())
	wantType(t, lnot, "!bool", boolT())
	wantType(t, bnot, "~i32", i32T())
	wantType(t, sel, "select bool/f32/f32", f32T())
	wantType(t, cast, "cast f32->f16", ir.InternScalar(ir.TF16))
	wantType(t, call, "call expf(f32)", f32T())
}

// TestExprScalarErrors pins the fail-closed expr rejections, each on its
// own graph: mixed operand types, wrong operand classes, select arm
// mismatch, unknown builtin, non-f32 builtin argument.
func TestExprScalarErrors(t *testing.T) {
	cases := []struct {
		name    string
		build   func(g *ir.Graph) ir.Node
		keyword string
	}{
		{"addMixedTypes", func(g *ir.Graph) ir.Node {
			return g.NewBinOp("t", ir.BinOpAdd, g.NewConst("t", f32T(), 1), g.NewConst("t", i32T(), 1))
		}, "types differ"},
		{"landNonBool", func(g *ir.Graph) ir.Node {
			return g.NewBinOp("t", ir.BinOpLAnd, g.NewConst("t", f32T(), 1), g.NewConst("t", f32T(), 1))
		}, "bool"},
		{"modFloats", func(g *ir.Graph) ir.Node {
			return g.NewBinOp("t", ir.BinOpMod, g.NewConst("t", f32T(), 1), g.NewConst("t", f32T(), 1))
		}, "integer"},
		{"selectArmMismatch", func(g *ir.Graph) ir.Node {
			return g.NewSelect("t", g.NewConst("t", boolT(), 1), g.NewConst("t", f32T(), 1), g.NewConst("t", i32T(), 1))
		}, "differ"},
		{"callUnknown", func(g *ir.Graph) ir.Node {
			return g.NewCall("t", "nosuch", g.NewConst("t", f32T(), 1))
		}, "not a registered"},
		{"callNonF32Arg", func(g *ir.Graph) ir.Node {
			return g.NewCall("t", "fmaf", g.NewConst("t", f32T(), 1), g.NewConst("t", i32T(), 1), g.NewConst("t", f32T(), 1))
		}, "f32"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := ir.NewGraph(tc.name)
			n := tc.build(g)
			wantErrNames(t, runTI(g), n.BasePtr().ID(), tc.keyword)
		})
	}
}

// TestParamUntypedFailClosed pins the spec's Param rule: a scalar/tensor
// parameter with no user type errors naming the node (RoleDim is exempt —
// its computed type is u32).
func TestParamUntypedFailClosed(t *testing.T) {
	g := ir.NewGraph("untyped")
	bad := g.NewInputParam("t", ir.RoleScalar, "s", nil)
	g.NewParamRef("t", bad)
	wantErrNames(t, runTI(g), bad.BasePtr().ID(), "type")
}

// ---------------------------------------------------------------------------
// Loop/IV value kinds (spec (2): everything u32)
// ---------------------------------------------------------------------------

// TestLoopValueKindsU32 pins iter/forloop/dimvar/bdimvar/inductionvariable/
// tensordimof to scalar u32.
func TestLoopValueKindsU32(t *testing.T) {
	g := ir.NewGraph("loops")
	d := g.NewInputParam("t", ir.RoleDim, "d", nil)
	c0 := g.NewConst("t", f32T(), 0)
	c1 := g.NewConst("t", f32T(), 1)
	tn := g.NewInputParam("t", ir.RoleTensor, "t", tensorT(ir.TF32, "a"))

	dv := g.NewDimVar("t", d, c0, c1)
	biv := g.NewBoundedDimVar("t", dv, c1)
	it := g.NewIter("t", biv, nil) // (biv, parent?) formation — bounded by construction
	iv := g.NewInductionVariable("t", it)
	td := g.NewTensorDimOf("t", tn, 0)
	fl := g.NewForLoop("t", dv, c1, c1)

	mustRun(t, g)
	for _, n := range []ir.Node{it, dv, biv, iv, td, fl} {
		wantType(t, n, n.KindOf().String(), u32T())
	}
}

// ---------------------------------------------------------------------------
// Lambda rules (spec (3))
// ---------------------------------------------------------------------------

// TestMapTyping pins the Map rule: type from the As clause (rank = len(As),
// dtype = body elem, symbolic DimRefs by dim-param name).
func TestMapTyping(t *testing.T) {
	g := ir.NewGraph("map")
	m := g.NewInputParam("t", ir.RoleDim, "m", nil)
	k := g.NewInputParam("t", ir.RoleDim, "k", nil)
	l := g.NewLambdaInputParam("t", "l", f32T())
	r := g.NewLambdaInputParam("t", "r", f32T())
	lhs := g.NewInputParam("t", ir.RoleTensor, "lhs", tensorT(ir.TF32, "m", "k"))
	rhs := g.NewInputParam("t", ir.RoleTensor, "rhs", tensorT(ir.TF32, "m", "k"))
	body := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", l), g.NewParamRef("t", r))
	mp := g.NewMap("t", []ir.Node{g.NewParamRef("t", lhs), g.NewParamRef("t", rhs)},
		ir.NewLambda([]ir.Node{l, r}, body), []ir.Node{m, k})
	g.NewReturn("t", mp)

	mustRun(t, g)
	wantType(t, mp, "map as(m,k) f32 body", tensorT(ir.TF32, "m", "k"))
}

// TestMapWithoutAsErrors pins tca.md §2.7's mandatory As: a map without an
// as-order fails closed naming the map.
func TestMapWithoutAsErrors(t *testing.T) {
	g := ir.NewGraph("mapnoas")
	v := g.NewLambdaInputParam("t", "v", f32T())
	val := g.NewConst("t", f32T(), 1)
	mp := g.NewMap("t", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, g.NewParamRef("t", v)), nil)
	g.NewReturn("t", mp)
	wantErrNames(t, runTI(g), mp.BasePtr().ID(), "as")
}

// TestReduceTyping pins the Reduce rule: the piped rank-3 tensor reduces to
// its rank-2 head (last dim dropped), rank-1 to the scalar elem.
func TestReduceTyping(t *testing.T) {
	g := ir.NewGraph("red")
	a := g.NewInputParam("t", ir.RoleDim, "a", nil)
	b := g.NewInputParam("t", ir.RoleDim, "b", nil)
	c := g.NewInputParam("t", ir.RoleDim, "c", nil)
	v := g.NewLambdaInputParam("t", "v", f32T())
	mp := g.NewMap("t", []ir.Node{g.NewConst("t", f32T(), 1)},
		ir.NewLambda([]ir.Node{v}, g.NewParamRef("t", v)), []ir.Node{a, b, c})

	acc := g.NewLambdaInputParam("t", "acc", f32T())
	now := g.NewLambdaInputParam("t", "now", f32T())
	comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
	red := g.NewReduce("t", g.NewConst("t", f32T(), 0), mp, ir.NewLambda([]ir.Node{acc, now}, comb))
	g.NewReturn("t", red)

	mustRun(t, g)
	wantType(t, mp, "rank-3 map", tensorT(ir.TF32, "a", "b", "c"))
	wantType(t, red, "rank-2 reduce head", tensorT(ir.TF32, "a", "b"))
}

// TestReduceErrors pins the reduce rejections: scalar piped value
// (nothing to reduce), heterogeneous init (tca.md §4.2), and a provided As
// of the wrong rank.
func TestReduceErrors(t *testing.T) {
	newReducer := func(g *ir.Graph) (red ir.Node, piped ir.Node) {
		a := g.NewInputParam("t", ir.RoleDim, "a", nil)
		b := g.NewInputParam("t", ir.RoleDim, "b", nil)
		v := g.NewLambdaInputParam("t", "v", f32T())
		mp := g.NewMap("t", []ir.Node{g.NewConst("t", f32T(), 1)},
			ir.NewLambda([]ir.Node{v}, g.NewParamRef("t", v)), []ir.Node{a, b})
		acc := g.NewLambdaInputParam("t", "acc", f32T())
		now := g.NewLambdaInputParam("t", "now", f32T())
		comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
		return g.NewReduce("t", g.NewConst("t", f32T(), 0), mp, ir.NewLambda([]ir.Node{acc, now}, comb)), mp
	}

	t.Run("scalarPiped", func(t *testing.T) {
		g := ir.NewGraph("redscalar")
		acc := g.NewLambdaInputParam("t", "acc", f32T())
		now := g.NewLambdaInputParam("t", "now", f32T())
		comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
		red := g.NewReduce("t", g.NewConst("t", f32T(), 0), g.NewConst("t", f32T(), 1),
			ir.NewLambda([]ir.Node{acc, now}, comb))
		wantErrNames(t, runTI(g), red.BasePtr().ID(), "reduce")
	})

	t.Run("heterogeneousInit", func(t *testing.T) {
		g := ir.NewGraph("redinit")
		_, mp := newReducer(g)
		// The init KeyedAttribute wraps a u32 const while the piped value
		// carries f32 elements: the homogeneous contract (tca.md §4.2) is
		// violated.
		attr := g.NewKeyedAttribute("t", "init", g.NewConst("t", u32T(), 0))
		acc := g.NewLambdaInputParam("t", "acc", f32T())
		now := g.NewLambdaInputParam("t", "now", f32T())
		comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
		red := g.NewNode(ir.KindReduce, "t", mp, attr, g.NewLambdaNode("t", []ir.Node{acc, now}, comb))
		wantErrNames(t, runTI(g), red.BasePtr().ID(), "homogeneous")
	})

	t.Run("asWrongRank", func(t *testing.T) {
		g := ir.NewGraph("redas")
		_, mp := newReducer(g)
		a := g.NewInputParam("t", ir.RoleDim, "a", nil)
		acc := g.NewLambdaInputParam("t", "acc", f32T())
		now := g.NewLambdaInputParam("t", "now", f32T())
		comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
		// Rank-2 piped value, As of length 2 != the rank-1 derived shape.
		red := g.NewNode(ir.KindReduce, "t", mp,
			g.NewKeyedAttribute("t", "init", g.NewConst("t", f32T(), 0)),
			g.NewAs("t", a, a),
			g.NewLambdaNode("t", []ir.Node{acc, now}, comb))
		wantErrNames(t, runTI(g), red.BasePtr().ID(), "as")
	})
}

// ---------------------------------------------------------------------------
// Memory rules (spec (4), HIR and LIR alike)
// ---------------------------------------------------------------------------

// TestMemLoadDeref pins the partial/full deref rule: [M,K,N] + 1 index ->
// [K,N]; + 3 indexes -> f32; + 4 indexes -> over-dereferenced error.
func TestMemLoadDeref(t *testing.T) {
	g := ir.NewGraph("ld")
	m := g.NewInputParam("t", ir.RoleDim, "M", nil)
	k := g.NewInputParam("t", ir.RoleDim, "K", nil)
	n := g.NewInputParam("t", ir.RoleDim, "N", nil)
	tn := g.NewInputParam("t", ir.RoleTensor, "t", tensorT(ir.TF32, "M", "K", "N"))

	ld1 := g.NewMemLoad("t", tn, []ir.Node{m})
	ld3 := g.NewMemLoad("t", tn, []ir.Node{m, k, n})
	g.NewReturn("t", ld1, ld3)

	mustRun(t, g)
	wantType(t, ld1, "1-axis partial deref", tensorT(ir.TF32, "K", "N"))
	wantType(t, ld3, "3-axis full deref", f32T())

	g2 := ir.NewGraph("ld4")
	m2 := g2.NewInputParam("t", ir.RoleDim, "M", nil)
	k2 := g2.NewInputParam("t", ir.RoleDim, "K", nil)
	n2 := g2.NewInputParam("t", ir.RoleDim, "N", nil)
	j2 := g2.NewInputParam("t", ir.RoleDim, "J", nil)
	tn2 := g2.NewInputParam("t", ir.RoleTensor, "t", tensorT(ir.TF32, "M", "K", "N"))
	ld4 := g2.NewMemLoad("t", tn2, []ir.Node{m2, k2, n2, j2})
	wantErrNames(t, runTI(g2), ld4.BasePtr().ID(), "dereferenced")
}

// TestMemStoreTyping pins the store rule: the store returns the stored
// value's type.
func TestMemStoreTyping(t *testing.T) {
	g := ir.NewGraph("st")
	m := g.NewInputParam("t", ir.RoleDim, "M", nil)
	k := g.NewInputParam("t", ir.RoleDim, "K", nil)
	n := g.NewInputParam("t", ir.RoleDim, "N", nil)
	tn := g.NewInputParam("t", ir.RoleTensor, "out", tensorT(ir.TF32, "K", "N"))
	ld := g.NewMemLoad("t", g.NewInputParam("t", ir.RoleTensor, "in", tensorT(ir.TF32, "M", "K", "N")), []ir.Node{m})
	st := g.NewMemStore("t", tn, []ir.Node{k, n}, ld)
	g.NewReturn("t", st)

	mustRun(t, g)
	wantType(t, st, "store returns the stored value", tensorT(ir.TF32, "K", "N"))
}

// TestUntypedKindsStayNil pins the closed untyped set: memnew, memnewsmem,
// fragnew, as, and lambda stay TypeOf nil after a successful run.
func TestUntypedKindsStayNil(t *testing.T) {
	g := ir.NewGraph("untyped-kinds")
	m := g.NewInputParam("t", ir.RoleDim, "m", nil)
	v := g.NewLambdaInputParam("t", "v", f32T())
	val := g.NewConst("t", f32T(), 1)
	body := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", v), val)
	mp := g.NewMap("t", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, body), []ir.Node{m})
	mn := g.NewMemNew("t", val)
	sn := g.NewMemNewSMem("t", val)
	fn := g.NewFragNew("t", val)
	g.NewReturn("t", mp, mn, sn, fn)

	mustRun(t, g)
	for _, n := range []ir.Node{mn, sn, fn} {
		if got := ir.TypeOf(n); got != nil {
			t.Fatalf("%s must stay untyped, got %v", n.KindOf(), got)
		}
	}
	if as := ir.AsOf(mp); as != nil && ir.TypeOf(as) != nil {
		t.Fatal("as must stay untyped")
	}
	if lam := ir.LambdaOf(mp); lam != nil && ir.TypeOf(lam) != nil {
		t.Fatal("lambda must stay untyped")
	}
}

// TestSMemLoadUntypedBuffer pins the LIR fail-closed rule: a load from a
// buffer that carries no dtype errors naming the buffer (cache lowering
// must bind buffer types first).
func TestSMemLoadUntypedBuffer(t *testing.T) {
	g := ir.NewGraph("smem")
	buf := g.NewMemNewSMem("t", g.NewConst("t", f32T(), 64))
	ax := g.NewInputParam("t", ir.RoleDim, "i", nil)
	ld := g.NewMemLoadSMem("t", buf, ax)
	g.NewReturn("t", ld)

	err := runTI(g)
	wantErrNames(t, err, buf.BasePtr().ID(), "dtype")
}

// ---------------------------------------------------------------------------
// IterCluster rules (spec (2): lowered before TI; legacy wrapper passes through)
// ---------------------------------------------------------------------------

// TestIterClusterRules pins the legacy one-input wrapper passthrough and the
// fail-closed shapes: a live project errors on sight; a two-input cluster
// (the IterCluster format) errors.
func TestIterClusterRules(t *testing.T) {
	t.Run("legacyPassthrough", func(t *testing.T) {
		g := ir.NewGraph("wrap")
		d := g.NewInputParam("t", ir.RoleDim, "m", nil)
		v := g.NewLambdaInputParam("t", "v", f32T())
		val := g.NewConst("t", f32T(), 1)
		body := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", v), val)
		mp := g.NewMap("t", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, body), []ir.Node{d})
		ic := g.NewIterCluster("t", []ir.Node{d}, mp)
		g.NewReturn("t", ic)

		mustRun(t, g)
		wantType(t, mp, "wrapped map", tensorT(ir.TF32, "m"))
		wantType(t, ic, "legacy cluster passthrough", tensorT(ir.TF32, "m"))
	})

	t.Run("liveProjectFails", func(t *testing.T) {
		g := ir.NewGraph("proj")
		d := g.NewInputParam("t", ir.RoleDim, "m", nil)
		load := g.NewMemLoad("t", g.NewInputParam("t", ir.RoleTensor, "t", tensorT(ir.TF32, "m")), []ir.Node{d})
		cl := g.NewIterCluster("t", []ir.Node{d}, load)
		p := g.NewIterClusterProject("t", cl, 0)
		v := g.NewLambdaInputParam("t", "v", f32T())
		mp := g.NewMap("t", []ir.Node{p}, ir.NewLambda([]ir.Node{v}, g.NewParamRef("t", v)), []ir.Node{d})
		g.NewReturn("t", mp)
		wantErrNames(t, runTI(g), p.BasePtr().ID(), "iterclusterproject")
	})

	t.Run("twoInputClusterFails", func(t *testing.T) {
		g := ir.NewGraph("two")
		d := g.NewInputParam("t", ir.RoleDim, "m", nil)
		t1 := g.NewInputParam("t", ir.RoleTensor, "a", tensorT(ir.TF32, "m"))
		t2 := g.NewInputParam("t", ir.RoleTensor, "b", tensorT(ir.TF32, "m"))
		l1 := g.NewMemLoad("t", t1, []ir.Node{d})
		l2 := g.NewMemLoad("t", t2, []ir.Node{d})
		cl := g.NewIterCluster("t", []ir.Node{d}, l1, l2)
		g.NewReturn("t", cl)
		wantErrNames(t, runTI(g), cl.BasePtr().ID(), "lowered")
	})
}

// ---------------------------------------------------------------------------
// Idempotency + gates + cycle (spec preamble: re-runnable, fail-closed)
// ---------------------------------------------------------------------------

// TestIdempotent pins TI-1: a second run changes nothing — every TypeOf is
// identical AND every LastTouchedBy is stable.
func TestIdempotent(t *testing.T) {
	g := ir.NewGraph("idem")
	d := g.NewInputParam("t", ir.RoleDim, "m", nil)
	k := g.NewInputParam("t", ir.RoleDim, "k", nil)
	v := g.NewLambdaInputParam("t", "v", f32T())
	val := g.NewConst("t", f32T(), 1)
	body := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", v), val)
	mp := g.NewMap("t", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, body), []ir.Node{d, k})
	acc := g.NewLambdaInputParam("t", "acc", f32T())
	now := g.NewLambdaInputParam("t", "now", f32T())
	comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
	red := g.NewReduce("t", g.NewConst("t", f32T(), 0), mp, ir.NewLambda([]ir.Node{acc, now}, comb))
	out := g.NewInputParam("t", ir.RoleTensor, "out", tensorT(ir.TF32, "d", "k"))
	st := g.NewMemStore("t", out, []ir.Node{d, k}, red)
	g.NewReturn("t", st)

	mustRun(t, g)
	types1 := map[uint64]ir.Type{}
	touched1 := map[uint64]string{}
	for _, n := range g.AllocatedNodes() {
		types1[n.BasePtr().ID()] = ir.TypeOf(n)
		touched1[n.BasePtr().ID()] = n.BasePtr().LastTouchedBy()
	}

	mustRun(t, g) // re-run over the SAME graph
	for _, n := range g.AllocatedNodes() {
		id := n.BasePtr().ID()
		ty1, ok := types1[id]
		if !ok {
			t.Fatalf("node %d appeared after the second run", id)
		}
		ty2 := ir.TypeOf(n)
		if (ty1 == nil) != (ty2 == nil) {
			t.Fatalf("node %d (%s): TypeOf changed nil-ness across re-run", id, n.KindOf())
		}
		if ty1 != nil && !ty1.Equal(ty2) {
			t.Fatalf("node %d (%s): TypeOf changed across re-run: %v -> %v", id, n.KindOf(), ty1, ty2)
		}
		if got := n.BasePtr().LastTouchedBy(); got != touched1[id] {
			t.Fatalf("node %d (%s): LastTouchedBy changed across re-run: %q -> %q", id, n.KindOf(), touched1[id], got)
		}
	}
}

// TestRunGates pins the level gates: nil and materialized graphs are
// deterministic errors, evaluated before any mutation.
func TestRunGates(t *testing.T) {
	if err := runTI(nil); err == nil || !strings.Contains(err.Error(), "nil graph") {
		t.Fatalf("Run(nil) error = %v, want nil-graph refusal", err)
	}
	g := ir.NewGraph("mat")
	d := g.NewInputParam("t", ir.RoleDim, "m", nil)
	v := g.NewLambdaInputParam("t", "v", f32T())
	val := g.NewConst("t", f32T(), 1)
	mp := g.NewMap("t", []ir.Node{val}, ir.NewLambda([]ir.Node{v}, g.NewParamRef("t", v)), []ir.Node{d})
	g.NewReturn("t", mp)
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	err := runTI(g)
	if err == nil || !strings.Contains(err.Error(), "materialized") {
		t.Fatalf("a materialized graph must be refused, got: %v", err)
	}
}

// TestCycleFailClosed pins the data-dependency cycle refusal: a 2-node
// operand cycle (AddUse in both directions) errors naming a cycle node and
// leaves the graph completely unstamped.
func TestCycleFailClosed(t *testing.T) {
	g := ir.NewGraph("cyc")
	a := g.NewConst("t", f32T(), 1)
	b := g.NewConst("t", f32T(), 2)
	ctx := &ir.MutContext{Pass: "test"}
	ctx.AddUse(a, b) // a's new operand slot -> b
	ctx.AddUse(b, a) // b's new operand slot -> a: the cycle

	err := runTI(g)
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("an operand cycle must fail closed with a cycle error, got: %v", err)
	}
	for _, n := range g.AllocatedNodes() {
		if ir.TypeOf(n) != nil {
			t.Fatalf("a refused graph must be unstamped: node %d (%s) carries a type", n.BasePtr().ID(), n.KindOf())
		}
	}
}

// ---------------------------------------------------------------------------
// Oracle integration (the doc/tca.md §5.1 shape through the real pipeline)
// ---------------------------------------------------------------------------

// matmulOracle hand-builds the matmul_scale oracle shape the pkg builder
// emits (the established pass-test pattern: pkg.Build materializes, and
// typeinfer runs on the Level-1 sea): dims m/k/n, lhs[m,n], rhs[n,k],
// output[m,k], scalar scale; cluster-wrapped root map as(m,k,n); reduce;
// piped map as(m,k); store; chained return.
func matmulOracle(t *testing.T) (g *ir.Graph, rootMap, reduce, tailMap, store ir.Node) {
	t.Helper()
	g = ir.NewGraph("matmul_scale")
	f := f32T()
	m := g.NewInputParam("t", ir.RoleDim, "m", nil)
	k := g.NewInputParam("t", ir.RoleDim, "k", nil)
	n := g.NewInputParam("t", ir.RoleDim, "n", nil)
	lhs := g.NewInputParam("t", ir.RoleTensor, "lhs", tensorT(ir.TF32, "m", "n"))
	rhs := g.NewInputParam("t", ir.RoleTensor, "rhs", tensorT(ir.TF32, "n", "k"))
	out := g.NewInputParam("t", ir.RoleTensor, "output", tensorT(ir.TF32, "m", "k"))
	scale := g.NewInputParam("t", ir.RoleScalar, "scale", f)

	l := g.NewLambdaInputParam("t", "l", f)
	r := g.NewLambdaInputParam("t", "r", f)
	body := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", l), g.NewParamRef("t", r))
	rootMap = g.NewMap("t", []ir.Node{g.NewParamRef("t", lhs), g.NewParamRef("t", rhs)},
		ir.NewLambda([]ir.Node{l, r}, body), []ir.Node{m, k, n})
	ic1 := g.NewIterCluster("t", []ir.Node{m, k, n}, rootMap)

	acc := g.NewLambdaInputParam("t", "acc", f)
	now := g.NewLambdaInputParam("t", "now", f)
	comb := g.NewBinOp("t", ir.BinOpAdd, g.NewParamRef("t", acc), g.NewParamRef("t", now))
	reduce = g.NewReduce("t", g.NewConst("t", f, 0), ic1, ir.NewLambda([]ir.Node{acc, now}, comb))

	v := g.NewLambdaInputParam("t", "v", f)
	tbody := g.NewBinOp("t", ir.BinOpMul, g.NewParamRef("t", v), g.NewParamRef("t", scale))
	tailMap = g.NewMap("t", []ir.Node{reduce, g.NewParamRef("t", scale)},
		ir.NewLambda([]ir.Node{v}, tbody), []ir.Node{m, k})
	ic2 := g.NewIterCluster("t", nil, tailMap)

	env := ir.NewEnv()
	store = g.NewMemStoreChained("t", env, out, []ir.Node{m, k}, ic2)
	g.NewReturnChained("t", env, []ir.Node{out}, store)
	return g, rootMap, reduce, tailMap, store
}

// TestOracleIntegration runs the simplify passes + typeinfer over the
// matmul_scale oracle shape and asserts every stage's computed type plus
// the full postcondition (every non-erased node typed outside the untyped
// whitelist).
func TestOracleIntegration(t *testing.T) {
	g, rootMap, reduce, tailMap, store := matmulOracle(t)
	pm := opt.NewManager([]opt.Pass{
		simplify.LowerPass{},
		simplify.InlinePass{},
		simplify.VerifyPass{},
		typeinfer.Pass{},
	})
	if err := pm.Run(g); err != nil {
		t.Fatalf("simplify+typeinfer pipeline over the oracle: %v", err)
	}

	wantType(t, rootMap, "root map", tensorT(ir.TF32, "m", "k", "n"))
	wantType(t, reduce, "reduce", tensorT(ir.TF32, "m", "k"))
	wantType(t, tailMap, "tail map", tensorT(ir.TF32, "m", "k"))
	wantType(t, store, "store", tensorT(ir.TF32, "m", "k"))

	untyped := map[ir.Kind]bool{
		ir.KindMemNew:     true,
		ir.KindMemNewSMem: true,
		ir.KindFragNew:    true,
		ir.KindAs:         true,
		ir.KindLambda:     true,
	}
	for _, n := range g.AllocatedNodes() {
		if ir.TypeOf(n) == nil && !untyped[n.KindOf()] {
			// Return with 0 or >1 value operands is the documented
			// multi-value exemption; the oracle's return has exactly 1.
			t.Fatalf("node %d (%s) is typed-nil outside the untyped whitelist", n.BasePtr().ID(), n.KindOf())
		}
	}
}

// ---------------------------------------------------------------------------
// Registration + pipeline wiring (TI-8)
// ---------------------------------------------------------------------------

// TestRegistration pins the registry entry and the default pipeline
// position: priority 105, Requires simplify-lower, ordered between
// simplify-verify (104) and gvn (110).
func TestRegistration(t *testing.T) {
	spec, ok := opt.Lookup("typeinfer")
	if !ok {
		t.Fatal("typeinfer must be registered")
	}
	if spec.Priority != 105 {
		t.Fatalf("typeinfer priority = %d, want 105", spec.Priority)
	}
	if len(spec.Requires) != 1 || spec.Requires[0] != "simplify-lower" {
		t.Fatalf("typeinfer Requires = %v, want [simplify-lower]", spec.Requires)
	}

	p := typeinfer.Pass{}
	if p.ID() != "typeinfer" {
		t.Fatalf("Pass.ID() = %q", p.ID())
	}
	if req := p.Requires(); len(req) != 1 || req[0] != "simplify-lower" {
		t.Fatalf("Pass.Requires() = %v, want [simplify-lower]", req)
	}
	if p.Produces() != opt.StageConstructed {
		t.Fatalf("Pass.Produces() = %v, want StageConstructed", p.Produces())
	}
	if p.Preserved() != nil {
		t.Fatalf("Pass.Preserved() = %v, want nil", p.Preserved())
	}

	pipe := opt.BuildPipeline()
	pos := make(map[string]int)
	for i, pass := range pipe {
		pos[pass.ID()] = i
	}
	if _, ok := pos["typeinfer"]; !ok {
		t.Fatal("the default pipeline must contain typeinfer")
	}
	if pos["simplify-verify"] >= pos["typeinfer"] || pos["typeinfer"] >= pos["gvn"] {
		t.Fatalf("typeinfer must order between simplify-verify and gvn: %v", pos)
	}
}
