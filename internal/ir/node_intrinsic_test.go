package ir

import (
	"strings"
	"testing"
)

// node_intrinsic_test.go pins the loop-tile-lower launch-map query nodes
// (quick-261021-jay, doc/optana/loop_tile_lower.md §5): the P1 payload/shape
// contract (string field payload on the node struct, single const-u32 DoSoC
// operand, field vocabulary), the §5 companion-3 verifier shape checks, the
// §5 companion-6 value-numbering content (field mixing, DoSoC identity
// keying), and the canonical print rendering.

// u32const allocates a const-u32 node (the DoSoC operand shape).
func u32const(g *Graph, v float64) Node {
	return g.NewConst("ctor", InternScalar(TU32), v)
}

// TestIntrinsicFieldOf pins the P1 read face: the field payload round-trips
// through the constructors and IntrinsicFieldOf; foreign nodes report "".
func TestIntrinsicFieldOf(t *testing.T) {
	g := NewGraph("intrinsic")
	d := u32const(g, 0)
	gm := g.NewIntrinsicGridMap("ctor", "index", d)
	tm := g.NewIntrinsicThreadMap("ctor", "dimension", d)
	if got := IntrinsicFieldOf(gm); got != "index" {
		t.Fatalf("IntrinsicFieldOf(gridmap) = %q, want %q", got, "index")
	}
	if got := IntrinsicFieldOf(tm); got != "dimension" {
		t.Fatalf("IntrinsicFieldOf(threadmap) = %q, want %q", got, "dimension")
	}
	if got := IntrinsicFieldOf(d); got != "" {
		t.Fatalf("IntrinsicFieldOf(const) = %q, want \"\"", got)
	}
	if got := IntrinsicFieldOf(nil); got != "" {
		t.Fatalf("IntrinsicFieldOf(nil) = %q, want \"\"", got)
	}
}

// TestIntrinsicVerifyShape pins §5 companion 3: a well-formed query verifies
// clean; a bad field string, a non-const DoSoC, and a multi-operand query
// are each shape violations (zero-new-invariant-class extensions of the
// existing per-kind shape class).
func TestIntrinsicVerifyShape(t *testing.T) {
	wellformed := func() *Graph {
		g := NewGraph("shape")
		d := u32const(g, 1)
		gm := g.NewIntrinsicGridMap("ctor", "index", d)
		tm := g.NewIntrinsicThreadMap("ctor", "dimension", u32const(g, 2))
		g.NewReturn("ctor", gm, tm) // anchor: Level-1 Verify walks from the end node
		return g
	}
	if vs := Verify(wellformed()); len(vs) != 0 {
		t.Fatalf("well-formed queries must verify clean, got %v", vs)
	}

	badField := func() *Graph {
		g := NewGraph("badfield")
		gm := g.NewIntrinsicGridMap("ctor", "rank", u32const(g, 0))
		g.NewReturn("ctor", gm)
		return g
	}
	if vs := Verify(badField()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "index\" or \"dimension") {
		t.Fatalf("bad field must be a shape violation, got %v", vs)
	}

	nonConst := func() *Graph {
		g := NewGraph("nonconst")
		m := g.NewInputParam("ctor", RoleDim, "m", nil)
		gm := g.NewIntrinsicGridMap("ctor", "index", m) // dim params are typed u32 but are not consts
		g.NewReturn("ctor", gm)
		return g
	}
	if vs := Verify(nonConst()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "const-u32 DoSoC") {
		t.Fatalf("non-const DoSoC must be a shape violation, got %v", vs)
	}

	f32Const := func() *Graph {
		g := NewGraph("f32const")
		gm := g.NewIntrinsicGridMap("ctor", "index", g.NewConst("ctor", InternScalar(TF32), 0))
		g.NewReturn("ctor", gm)
		return g
	}
	if vs := Verify(f32Const()); len(vs) != 1 || !strings.Contains(vs[0].Msg, "const-u32 DoSoC") {
		t.Fatalf("f32 DoSoC const must be a shape violation, got %v", vs)
	}

	// A multi-operand query via the raw NewNode path fails closed twice:
	// the second operand is recorded as surplus (WR-01, the fixed-arity
	// gate fires before wiring) and the unset field violates the
	// vocabulary. Both are extensions of the existing per-kind shape class.
	multiOperand := NewGraph("multi")
	raw := multiOperand.NewNode(KindIntrinsicThreadMap, "ctor", u32const(multiOperand, 0), u32const(multiOperand, 1))
	multiOperand.NewReturn("ctor", raw)
	vs := Verify(multiOperand)
	if len(vs) != 2 || !strings.Contains(vs[0].Msg, "surplus") || !strings.Contains(vs[1].Msg, "index\" or \"dimension") {
		t.Fatalf("multi-operand query must fail closed on surplus + field, got %v", vs)
	}
}

// TestIntrinsicValueNumbering pins §5 companion 6: the field participates in
// the content hash/equality (distinct fields never merge); the DoSoC is an
// operand and identity-keyed (distinct consts with the same value DO merge
// — operands key by def node ID... two distinct const nodes with equal value
// hash differently, which the identity-keying contract pins); the GVN-CSE
// contract "equal ⇒ same hash" holds.
func TestIntrinsicValueNumbering(t *testing.T) {
	g := NewGraph("vn")
	se := pureAll{}
	d0 := u32const(g, 0)
	d0b := u32const(g, 0) // a distinct def with the same value
	a := asValue(t, g.NewIntrinsicGridMap("ctor", "index", d0))
	b := asValue(t, g.NewIntrinsicGridMap("ctor", "index", d0))
	c := asValue(t, g.NewIntrinsicGridMap("ctor", "dimension", d0))
	d := asValue(t, g.NewIntrinsicThreadMap("ctor", "index", d0))
	e := asValue(t, g.NewIntrinsicGridMap("ctor", "index", d0b))

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
