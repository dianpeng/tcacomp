package ir

import (
	"strings"
	"testing"
)

// node_runtime_test.go pins the RUNTIME node family (quick-261022-l1o,
// doc/optana/final_lower.md as amended; the launch-map query nodes of
// quick-261021-jay renamed to the Runtime vocabulary, LR-6): the payload/
// shape contracts (string payloads on the node structs, P-3's closed
// vocabularies and cross-checks), the value-numbering content (payload
// mixing, identity-keyed operands), the canonical print rendering, and the
// kind taxonomy (family assignments + append-only enum placement).

// u32const allocates a const-u32 node (the DoSoC operand shape).
func u32const(g *Graph, v float64) Node {
	return g.NewConst("ctor", InternScalar(TU32), v)
}

// f32const allocates a const-f32 node (the init-value shape).
func f32const(g *Graph, v float64) Node {
	return g.NewConst("ctor", InternScalar(TF32), v)
}

// combinerOf builds a two-formal combiner lambda (acc, now) -> acc + now —
// the reduce-exemption shape that survives simplify-inline.
func combinerOf(g *Graph) Node {
	f := InternScalar(TF32)
	acc := g.NewLambdaInputParam("ctor", "acc", f)
	now := g.NewLambdaInputParam("ctor", "now", f)
	body := g.NewBinOp("ctor", BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	return g.NewLambdaNode("ctor", []Node{acc, now}, body)
}

// TestRuntimeAttributeOfQueries pins the renamed read face: the query field
// payload round-trips through the constructors and RuntimeAttributeOf
// (shared with the GetAttribute payload); foreign nodes report "".
func TestRuntimeAttributeOfQueries(t *testing.T) {
	g := NewGraph("runtime")
	d := u32const(g, 0)
	gq := g.NewRuntimeGridQuery("ctor", "index", d)
	tq := g.NewRuntimeThreadQuery("ctor", "dimension", d)
	if got := RuntimeAttributeOf(gq); got != "index" {
		t.Fatalf("RuntimeAttributeOf(gridquery) = %q, want %q", got, "index")
	}
	if got := RuntimeAttributeOf(tq); got != "dimension" {
		t.Fatalf("RuntimeAttributeOf(threadquery) = %q, want %q", got, "dimension")
	}
	if got := RuntimeAttributeOf(d); got != "" {
		t.Fatalf("RuntimeAttributeOf(const) = %q, want \"\"", got)
	}
	if got := RuntimeAttributeOf(nil); got != "" {
		t.Fatalf("RuntimeAttributeOf(nil) = %q, want \"\"", got)
	}
}

// TestRuntimeMethodOf pins the method read face: the payload round-trips
// through the constructor; foreign nodes report "".
func TestRuntimeMethodOf(t *testing.T) {
	g := NewGraph("methodof")
	red := g.NewRuntimeWarpReducer("ctor", f32const(g, 0), combinerOf(g))
	call := g.NewRuntimeCallMethod("ctor", "Combine", red)
	if got := RuntimeMethodOf(call); got != "Combine" {
		t.Fatalf("RuntimeMethodOf(call) = %q, want %q", got, "Combine")
	}
	if got := RuntimeMethodOf(red); got != "" {
		t.Fatalf("RuntimeMethodOf(reducer) = %q, want \"\"", got)
	}
	if got := RuntimeMethodOf(nil); got != "" {
		t.Fatalf("RuntimeMethodOf(nil) = %q, want \"\"", got)
	}
}

// TestRuntimeQueryVerifyShape pins the query shape contract (§5 companion 3,
// renamed): a well-formed query verifies clean; a bad field string, a
// non-const DoSoC, an f32 DoSoC, and a multi-operand query are each shape
// violations.
func TestRuntimeQueryVerifyShape(t *testing.T) {
	wellformed := func() *Graph {
		g := NewGraph("shape")
		d := u32const(g, 1)
		gq := g.NewRuntimeGridQuery("ctor", "index", d)
		tq := g.NewRuntimeThreadQuery("ctor", "dimension", u32const(g, 2))
		g.NewReturn("ctor", gq, tq) // anchor: Level-1 Verify walks from the end node
		return g
	}
	if vs := Verify(wellformed()); len(vs) != 0 {
		t.Fatalf("well-formed queries must verify clean, got %v", vs)
	}

	badField := func() *Graph {
		g := NewGraph("badfield")
		gq := g.NewRuntimeGridQuery("ctor", "rank", u32const(g, 0))
		g.NewReturn("ctor", gq)
		return g
	}
	if vs := Verify(badField()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "index\" or \"dimension") {
		t.Fatalf("bad field must be a shape violation, got %v", vs)
	}

	nonConst := func() *Graph {
		g := NewGraph("nonconst")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		gq := g.NewRuntimeGridQuery("ctor", "index", m) // dim params are typed u32 but are not consts
		g.NewReturn("ctor", gq)
		return g
	}
	if vs := Verify(nonConst()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "const-u32 DoSoC") {
		t.Fatalf("non-const DoSoC must be a shape violation, got %v", vs)
	}

	f32DoSoC := func() *Graph {
		g := NewGraph("f32dosoc")
		gq := g.NewRuntimeGridQuery("ctor", "index", g.NewConst("ctor", InternScalar(TF32), 0))
		g.NewReturn("ctor", gq)
		return g
	}
	if vs := Verify(f32DoSoC()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "const-u32 DoSoC") {
		t.Fatalf("f32 DoSoC const must be a shape violation, got %v", vs)
	}

	// A multi-operand query via the raw NewNode path fails closed twice:
	// the second operand is recorded as surplus (WR-01, the fixed-arity
	// gate fires before wiring) and the unset field violates the
	// vocabulary.
	multiOperand := NewGraph("multi")
	raw := multiOperand.NewNode(KindRuntimeThreadQuery, "ctor", u32const(multiOperand, 0), u32const(multiOperand, 1))
	multiOperand.NewReturn("ctor", raw)
	vs := Verify(multiOperand)
	if len(vs) != 2 || !strings.Contains(vs[0].Msg, "surplus") || !strings.Contains(vs[1].Msg, "index\" or \"dimension") {
		t.Fatalf("multi-operand query must fail closed on surplus + field, got %v", vs)
	}
}

// runtimeFx is the shared runtime-object fixture: a kernel info object, a
// grid + thread mapping pair, a warp reducer over a typed init and a live
// combiner lambda, and one of each generic node — anchored at the end node
// so Verify walks everything.
func runtimeFx() (*Graph, Node, Node, Node, Node, Node, Node, Node) {
	g := NewGraph("runtimefx")
	ki := g.NewRuntimeKernelInfo("ctor")
	gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
	tm := g.NewRuntimeThreadMapping("ctor", ki, []int{4, -1})
	red := g.NewRuntimeWarpReducer("ctor", f32const(g, 0), combinerOf(g))
	init := g.NewRuntimeCallMethod("ctor", "Init", red)
	combine := g.NewRuntimeCallMethod("ctor", "Combine", red)
	idx := g.NewRuntimeGetAttribute("ctor", "index", gm, u32const(g, 0))
	g.NewReturn("ctor", init, combine, idx, tm) // anchor
	return g, ki, gm, tm, red, init, combine, idx
}

// TestRuntimeObjectVerifyShape pins P-3: the full well-formed family
// verifies clean, and every closed-vocabulary / operand-domain / encoding
// violation is a named shape error.
func TestRuntimeObjectVerifyShape(t *testing.T) {
	g, _, _, _, _, _, _, _ := runtimeFx()
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("the well-formed runtime family must verify clean, got %v", vs)
	}

	expect := func(name string, build func(*Graph), want string) {
		t.Helper()
		g := NewGraph(name)
		build(g)
		vs := Verify(g)
		if len(vs) == 0 {
			t.Fatalf("%s must fail closed naming %q, got no violations", name, want)
		}
		for _, v := range vs {
			if strings.Contains(v.Msg, want) {
				return // the named shape violation is present in the set
			}
		}
		t.Fatalf("%s must fail closed naming %q, got %v", name, want, vs)
	}

	// KernelInfo rejects any operand — twice over: the fixed-arity-0
	// surplus gate (WR-01) fires first on the raw NewNode path, and the
	// dedicated shape case flags the operand count.
	expect("ki-operand", func(g *Graph) {
		ki := g.NewNode(KindRuntimeKernelInfo, "ctor", u32const(g, 0))
		g.NewReturn("ctor", ki)
	}, "operands")

	// Mappings: wrong object, non-const portion, portion count != ndim,
	// ndim < 1, -1 not last.
	expect("map-badobj", func(g *Graph) {
		gm := g.NewRuntimeGridMapping("ctor", u32const(g, 0), []int{-1})
		g.NewReturn("ctor", gm)
	}, "RuntimeKernelInfo")
	expect("map-nonconst", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		raw := g.NewNode(KindRuntimeGridMapping, "ctor", ki,
			g.NewConst("ctor", InternScalar(TI32), 1), m)
		g.NewReturn("ctor", raw) // anchor: Level-1 Verify walks from the end node
	}, "must be a const")
	expect("map-count", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeThreadMapping("ctor", ki, []int{2, 4})
		// Hand-corrupt: declare ndim 2 but the constructor wired portions
		// [2,4] — instead probe the mismatch through the raw node path.
		_ = gm
		raw := g.NewNode(KindRuntimeGridMapping, "ctor", ki,
			g.NewConst("ctor", InternScalar(TI32), 3),
			g.NewConst("ctor", InternScalar(TI32), 2))
		g.NewReturn("ctor", raw)
	}, "portions")
	expect("map-ndim0", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		raw := g.NewNode(KindRuntimeGridMapping, "ctor", ki,
			g.NewConst("ctor", InternScalar(TI32), 0))
		g.NewReturn("ctor", raw)
	}, "ndim")
	expect("map-remainder", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeThreadMapping("ctor", ki, []int{-1, 4})
		g.NewReturn("ctor", gm)
	}, "LAST portion")

	// Reducers: swapped and missing operands.
	expect("red-swapped", func(g *Graph) {
		lam := combinerOf(g)
		raw := g.NewNode(KindRuntimeSequentialReducer, "ctor", lam, f32const(g, 0))
		g.NewReturn("ctor", raw)
	}, "typed init value")
	expect("red-missing", func(g *Graph) {
		raw := g.NewNode(KindRuntimeWarpReducer, "ctor", f32const(g, 0))
		g.NewReturn("ctor", raw)
	}, "exactly (init, lambda)")

	// CallMethod: unknown payload, non-object operand 0, method on a
	// non-reducer object.
	expect("call-badpayload", func(g *Graph) {
		red := g.NewRuntimeWarpReducer("ctor", f32const(g, 0), combinerOf(g))
		call := g.NewRuntimeCallMethod("ctor", "current_value", red)
		g.NewReturn("ctor", call)
	}, "\"Init\", \"Reduce\", \"Combine\"")
	expect("call-badobj", func(g *Graph) {
		call := g.NewRuntimeCallMethod("ctor", "Init", u32const(g, 0))
		g.NewReturn("ctor", call)
	}, "reducer construction object")
	expect("call-nonreducer-obj", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		call := g.NewRuntimeCallMethod("ctor", "Combine", ki)
		g.NewReturn("ctor", call)
	}, "reducer construction object")

	// GetAttribute: unknown payload, total_* on non-KernelInfo,
	// index/dimension on non-mapping, index/dimension without exactly one
	// const-u32 arg.
	expect("get-badpayload", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		ga := g.NewRuntimeGetAttribute("ctor", "current_value", ki)
		g.NewReturn("ctor", ga)
	}, "\"total_cta\"")
	expect("get-total-on-mapping", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
		ga := g.NewRuntimeGetAttribute("ctor", "total_cta", gm)
		g.NewReturn("ctor", ga)
	}, "KernelInfo object")
	expect("get-index-on-ki", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		ga := g.NewRuntimeGetAttribute("ctor", "index", ki, u32const(g, 0))
		g.NewReturn("ctor", ga)
	}, "mapping object")
	// Flat-index() ruling (2026-10-23): the ZERO-ARG index getattr is the
	// legal flat form — the old "noargs fails" pin is superseded. The
	// zero-arg DIMENSION getattr still fails wanting exactly 1.
	expect("get-dimension-noargs", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
		ga := g.NewRuntimeGetAttribute("ctor", "dimension", gm)
		g.NewReturn("ctor", ga)
	}, "exactly 1")
	expect("get-index-twoargs", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
		ga := g.NewRuntimeGetAttribute("ctor", "dimension", gm, u32const(g, 0), u32const(g, 1))
		g.NewReturn("ctor", ga)
	}, "exactly 1")
	expect("get-index-nonconst-arg", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		ga := g.NewRuntimeGetAttribute("ctor", "index", gm, m)
		g.NewReturn("ctor", ga)
	}, "const")
	expect("get-total-withargs", func(g *Graph) {
		ki := g.NewRuntimeKernelInfo("ctor")
		ga := g.NewRuntimeGetAttribute("ctor", "total_cta", ki, u32const(g, 0))
		g.NewReturn("ctor", ga)
	}, "want 0")
}

// TestRuntimeKindTaxonomy pins LR-5's family assignments and the append-only
// enum placement: queries stay FamilyAbstract; the KernelInfo/mappings/
// CallMethod/GetAttribute set is FamilyRuntime; the three reducers are
// FamilyCollective's first members; every new kind is TierLIR.
func TestRuntimeKindTaxonomy(t *testing.T) {
	wantFamily := map[Kind]Family{
		KindRuntimeGridQuery:         FamilyAbstract,
		KindRuntimeThreadQuery:       FamilyAbstract,
		KindRuntimeKernelInfo:        FamilyRuntime,
		KindRuntimeGridMapping:       FamilyRuntime,
		KindRuntimeThreadMapping:     FamilyRuntime,
		KindRuntimeCallMethod:        FamilyRuntime,
		KindRuntimeGetAttribute:      FamilyRuntime,
		KindRuntimeSequentialReducer: FamilyCollective,
		KindRuntimeWarpReducer:       FamilyCollective,
		KindRuntimeGlobalReducer:     FamilyCollective,
	}
	for k, f := range wantFamily {
		if got := k.Family(); got != f {
			t.Errorf("Kind(%s).Family() = %v, want %v", k, got, f)
		}
		if got := k.Tier(); got != TierLIR {
			t.Errorf("Kind(%s).Tier() = %v, want TierLIR", k, got)
		}
	}
	// Canonical spellings (P-1).
	wantString := map[Kind]string{
		KindRuntimeGridQuery:         "runtimegridquery",
		KindRuntimeThreadQuery:       "runtimethreadquery",
		KindRuntimeKernelInfo:        "runtimekernelinfo",
		KindRuntimeGridMapping:       "runtimegridmapping",
		KindRuntimeThreadMapping:     "runtimethreadmapping",
		KindRuntimeSequentialReducer: "runtimesequentialreducer",
		KindRuntimeWarpReducer:       "runtimewarpreducer",
		KindRuntimeGlobalReducer:     "runtimeglobalreducer",
		KindRuntimeCallMethod:        "runtimecallmethod",
		KindRuntimeGetAttribute:      "runtimegetattribute",
	}
	for k, s := range wantString {
		if got := k.String(); got != s {
			t.Errorf("Kind(%d).String() = %q, want %q", k, got, s)
		}
	}
	// The 8 new kinds append before KindEnd, after the renamed queries.
	if !(KindRuntimeGetAttribute < KindEnd && KindRuntimeCallMethod < KindEnd) {
		t.Fatal("the final-lower kinds must sit before KindEnd")
	}
	if !(KindRuntimeThreadQuery < KindRuntimeKernelInfo && KindRuntimeKernelInfo < KindRuntimeGridMapping) {
		t.Fatal("the queries must precede the constructions in enum order")
	}
}

// TestRuntimeFlatQueryVerifyShape pins the flat-index() ruling (2026-10-23):
// the ZERO-OPERAND query form is verifier-legal ONLY for field "index" (the
// thread's/Crank's absolute flattened 1-D rank — the threadIdx.x/blockIdx.x
// analog); field "dimension" still demands exactly one const-u32 DoSoC. The
// one-operand forms behave exactly as today (TestRuntimeQueryVerifyShape).
func TestRuntimeFlatQueryVerifyShape(t *testing.T) {
	wellformed := func() (*Graph, Node, Node) {
		g := NewGraph("flat")
		gq := g.NewRuntimeGridQueryFlat("ctor")
		tq := g.NewRuntimeThreadQueryFlat("ctor")
		g.NewReturn("ctor", gq, tq)
		return g, gq, tq
	}
	g, gq, tq := wellformed()
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("zero-operand flat queries must verify clean, got %v", vs)
	}
	if got := RuntimeAttributeOf(gq); got != "index" {
		t.Fatalf("flat grid query field = %q, want \"index\"", got)
	}
	if got := RuntimeAttributeOf(tq); got != "index" {
		t.Fatalf("flat thread query field = %q, want \"index\"", got)
	}

	// A zero-operand "dimension" query: flip the flat ctor's field — the
	// shape case must reject it, the flat form is index-only.
	badDim := NewGraph("flatdim")
	dq := badDim.NewRuntimeThreadQueryFlat("ctor").(*runtimeQueryNode)
	dq.field = "dimension"
	badDim.NewReturn("ctor", dq)
	if vs := Verify(badDim); len(vs) != 1 || !strings.Contains(vs[0].Msg, "dimension") {
		t.Fatalf("zero-operand dimension query must be a shape violation, got %v", vs)
	}

	// The getattr mirror: attribute "index" with ZERO extra args is legal
	// (the flat form); "dimension" keeps exactly one.
	ga1 := NewGraph("flatgetattr")
	ki1 := ga1.NewRuntimeKernelInfo("ctor")
	tm1 := ga1.NewRuntimeThreadMapping("ctor", ki1, []int{4, -1})
	ga1.NewReturn("ctor", ga1.NewRuntimeGetAttribute("ctor", "index", tm1))
	if vs := Verify(ga1); len(vs) != 0 {
		t.Fatalf("zero-arg index getattr must verify clean, got %v", vs)
	}
	g2 := NewGraph("flatgetattrdim")
	ki2 := g2.NewRuntimeKernelInfo("ctor")
	tm2 := g2.NewRuntimeThreadMapping("ctor", ki2, []int{4, -1})
	g2.NewReturn("ctor", g2.NewRuntimeGetAttribute("ctor", "dimension", tm2))
	if vs := Verify(g2); len(vs) != 1 || !strings.Contains(vs[0].Msg, "exactly 1") {
		t.Fatalf("zero-arg dimension getattr must fail wanting exactly 1 arg, got %v", vs)
	}
}

// TestRuntimeFlatQueryValueNumbering pins the flat form's value contract:
// two flat queries of the same kind are GVN-equal; a flat query is NOT equal
// to the dim-0 one-operand query (distinct operand lists hash distinct — the
// slot count participates in ValueHash).
func TestRuntimeFlatQueryValueNumbering(t *testing.T) {
	g := NewGraph("flatvn")
	se := pureAll{}
	f1 := asValue(t, g.NewRuntimeThreadQueryFlat("ctor"))
	f2 := asValue(t, g.NewRuntimeThreadQueryFlat("ctor"))
	d0 := asValue(t, g.NewRuntimeThreadQuery("ctor", "index", u32const(g, 0)))
	gf := asValue(t, g.NewRuntimeGridQueryFlat("ctor"))

	if !f1.ValueEqual(f2, se) {
		t.Fatal("two flat thread queries must be value-equal")
	}
	if f1.ValueHash(se) != f2.ValueHash(se) {
		t.Fatal("equal flat queries must hash identically (the GVN-CSE contract)")
	}
	if f1.ValueEqual(d0, se) {
		t.Fatal("a flat query must not be value-equal to the dim-0 one-operand query")
	}
	if f1.ValueHash(se) == d0.ValueHash(se) {
		t.Fatal("the flat and one-operand forms must hash distinctly (operand-count keying)")
	}
	if f1.ValueEqual(gf, se) {
		t.Fatal("flat grid and thread queries are distinct kinds")
	}
}

// TestRuntimeQueryValueNumbering pins §5 companion 6: the field participates
// in the content hash/equality (distinct fields never merge); the DoSoC is
// an operand and identity-keyed; the GVN-CSE contract "equal ⇒ same hash"
// holds.
func TestRuntimeQueryValueNumbering(t *testing.T) {
	g := NewGraph("vn")
	se := pureAll{}
	d0 := u32const(g, 0)
	d0b := u32const(g, 0) // a distinct def with the same value
	a := asValue(t, g.NewRuntimeGridQuery("ctor", "index", d0))
	b := asValue(t, g.NewRuntimeGridQuery("ctor", "index", d0))
	c := asValue(t, g.NewRuntimeGridQuery("ctor", "dimension", d0))
	d := asValue(t, g.NewRuntimeThreadQuery("ctor", "index", d0))
	e := asValue(t, g.NewRuntimeGridQuery("ctor", "index", d0b))

	if !a.ValueEqual(b, se) {
		t.Fatal("same kind, same field, same DoSoC def must be value-equal")
	}
	if a.ValueHash(se) != b.ValueHash(se) {
		t.Fatal("equal values must hash identically (the GVN-CSE contract)")
	}
	if a.ValueEqual(c, se) {
		t.Fatal("differing fields must not be value-equal")
	}
	if a.ValueEqual(d, se) {
		t.Fatal("differing kinds must not be value-equal")
	}
	if a.ValueEqual(e, se) {
		t.Fatal("differing DoSoC defs must not be value-equal (identity keying)")
	}
}

// TestRuntimeObjectValueNumbering pins the new nodes' content contract
// (quick-261022-l1o): two RuntimeCallMethod nodes with identical operands
// but different payloads hash/compare unequal; two
// RuntimeGetAttribute(%sameObj, "index", %d) reads are one value; the
// constructions compare by kind + operands. NOTE: EffectOf marks every
// RuntimeCallMethod full, so call nodes hash to the one-of-a-kind sentinel
// under a real analysis — the payload distinction is pinned through the
// content hash path and getattr identity instead (the calls never enter a
// GVN bucket by design, P-2).
func TestRuntimeObjectValueNumbering(t *testing.T) {
	g := NewGraph("objvn")
	se := pureAll{}
	ki := g.NewRuntimeKernelInfo("ctor")
	gm := g.NewRuntimeGridMapping("ctor", ki, []int{-1})
	dosoc := u32const(g, 0) // ONE shared const def: DoSoC operands are identity-keyed

	i1 := asValue(t, g.NewRuntimeGetAttribute("ctor", "index", gm, dosoc))
	i2 := asValue(t, g.NewRuntimeGetAttribute("ctor", "index", gm, dosoc))
	d1 := asValue(t, g.NewRuntimeGetAttribute("ctor", "dimension", gm, dosoc))
	if !i1.ValueEqual(i2, se) {
		t.Fatal("two getattr reads over the same object and same-const DoSoC must be value-equal")
	}
	if i1.ValueHash(se) != i2.ValueHash(se) {
		t.Fatal("equal getattr reads must hash identically")
	}
	if i1.ValueEqual(d1, se) {
		t.Fatal("differing attribute payloads must not be value-equal")
	}
	// Two construction defs over the same operands are one VALUE (GVN-sound
	// per the doc: the objects are immutable; and no GVN pass runs after
	// final-lower anyway, so value-identical reducer constructions can
	// never be CSE-merged — P-2's note).
	if k1, k2 := asValue(t, ki), asValue(t, g.NewRuntimeKernelInfo("ctor")); !k1.ValueEqual(k2, se) {
		t.Fatal("two empty-operand KernelInfo constructions are one value (kind + empty operands)")
	}

	// The method payload participates through mixString: identical object,
	// differing payload — unequal content hash (compared through the
	// package-private contentHash, since EffectOf excludes calls from the
	// Value surface under a real analysis).
	red := g.NewRuntimeWarpReducer("ctor", f32const(g, 0), combinerOf(g))
	initCall := g.NewRuntimeCallMethod("ctor", "Init", red)
	combineCall := g.NewRuntimeCallMethod("ctor", "Combine", red)
	if contentHash(initCall) == contentHash(combineCall) {
		t.Fatal("differing method payloads must hash differently (mixString participation)")
	}
	if contentEqual(initCall, combineCall) {
		t.Fatal("differing method payloads must not be content-equal")
	}
}

// TestForLoopAnnotations pins P3/M1: trailing KeyedAttribute annotations
// verify clean, a trailing non-annotation operand is a shape violation, and
// the legacy three-operand shape still verifies.
func TestForLoopAnnotations(t *testing.T) {
	annotated := func() *Graph {
		g := NewGraph("annot")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		dv := g.NewDimVar("ctor", m, g.NewConst("ctor", InternScalar(TF32), 0), g.NewConst("ctor", InternScalar(TF32), 1))
		bdv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", InternScalar(TF32), 8))
		soc := g.NewKeyedAttribute("ctor", "soc", u32const(g, 1))
		loc := g.NewKeyedAttribute("ctor", "loc", u32const(g, 100))
		fl := g.NewForLoop("ctor", dv, bdv, u32const(g, 0), soc, loc)
		g.NewReturn("ctor", fl) // anchor
		return g
	}
	if vs := Verify(annotated()); len(vs) != 0 {
		t.Fatalf("annotated for-loop must verify clean, got %v", vs)
	}

	legacy := func() *Graph {
		g := NewGraph("legacy")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		dv := g.NewDimVar("ctor", m, g.NewConst("ctor", InternScalar(TF32), 0), g.NewConst("ctor", InternScalar(TF32), 1))
		bdv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", InternScalar(TF32), 8))
		fl := g.NewForLoop("ctor", dv, bdv, u32const(g, 0))
		g.NewReturn("ctor", fl) // anchor
		return g
	}
	if vs := Verify(legacy()); len(vs) != 0 {
		t.Fatalf("legacy three-operand for-loop must verify clean, got %v", vs)
	}

	badTrailing := func() *Graph {
		g := NewGraph("badtrailing")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		dv := g.NewDimVar("ctor", m, g.NewConst("ctor", InternScalar(TF32), 0), g.NewConst("ctor", InternScalar(TF32), 1))
		bdv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", InternScalar(TF32), 8))
		fl := g.NewForLoop("ctor", dv, bdv, u32const(g, 0), g.NewConst("ctor", InternScalar(TU32), 0))
		g.NewReturn("ctor", fl) // anchor
		return g
	}
	vs := Verify(badTrailing())
	if len(vs) != 1 || !strings.Contains(vs[0].Msg, "KeyedAttribute annotations (M1)") {
		t.Fatalf("non-annotation trailing operand must be a shape violation, got %v", vs)
	}
}
