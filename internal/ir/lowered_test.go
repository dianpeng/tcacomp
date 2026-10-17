package ir

import (
	"strings"
	"testing"
)

// ivGraph builds two normalized dimvars @x and @y plus bounds for the
// IV/BIV relation tests: x=(0,1), y=(0,1); bounds M=64, N=32.
func ivGraph(t *testing.T) (g *Graph, x, y, m64, n32 Node) {
	t.Helper()
	g = NewGraph("iv")
	dx := g.NewInputParam("ctor", RoleDim, "x", nil)
	dy := g.NewInputParam("ctor", RoleDim, "y", nil)
	f32 := InternScalar(TF32)
	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	x = g.NewDimVar("ctor", dx, zero, one)
	y = g.NewDimVar("ctor", dy, zero, one)
	m64 = g.NewConst("ctor", f32, 64)
	n32 = g.NewConst("ctor", f32, 32)
	return g, x, y, m64, n32
}

// TestSameIVValueIdentity pins the value-object spec: IV equality is
// (init, step) tuple equality — across distinct nodes, names, and dim
// bindings — never name identity.
func TestSameIVValueIdentity(t *testing.T) {
	g, x, y, _, _ := ivGraph(t)
	if !SameIV(x, x) {
		t.Fatal("an IV must equal itself")
	}
	if !SameIV(x, y) {
		t.Fatal("distinct dimvars with equal (init, step) tuples must be equal (value objects)")
	}

	// A different step breaks equality.
	f32 := InternScalar(TF32)
	two := g.NewConst("ctor", f32, 2)
	dz := g.NewInputParam("ctor", RoleDim, "z", nil)
	z := g.NewDimVar("ctor", dz, g.NewConst("ctor", f32, 0), two)
	if SameIV(x, z) {
		t.Fatal("IVs with different steps must not be equal")
	}
	// Totality: a non-dimvar node never equals an IV.
	if SameIV(x, g.NewConst("ctor", f32, 0)) {
		t.Fatal("non-dimvar nodes must never equal a dimvar")
	}
}

func TestSameBIVAndRelations(t *testing.T) {
	_, x, y, m64, n32 := ivGraph(t)
	bx64 := newGuardedBIV(x, m64)
	by64 := newGuardedBIV(y, m64)
	bx32 := newGuardedBIV(x, n32)

	if !SameBIV(bx64, by64) {
		t.Fatal("BIVs with equal dimvars and equal max are equal")
	}
	if SameBIV(bx64, bx32) {
		t.Fatal("BIVs with different max are not equal")
	}

	// Include: the spec's formal rule A ~> B iff A.max <= B.max (normalized
	// IVs only); the relation is NOT commutative. 32 ~> 64 holds, 64 ~> 32
	// fails.
	if !Includes(bx32, bx64) {
		t.Fatal("A(32) ~> B(64) must hold per the spec rule A.max <= B.max")
	}
	if Includes(bx64, bx32) {
		t.Fatal("include is not commutative: A(64) ~> B(32) must fail")
	}
	// Belong is the reverse.
	if !BelongsTo(bx64, bx32) {
		t.Fatal("B(64) <~ A(32) must hold (belong is the reverse of include)")
	}
	if BelongsTo(bx32, bx64) {
		t.Fatal("A(32) <~ B(64) must fail")
	}
}

// newGuardedBIV builds a bounded dimvar through a graph-shaped helper so
// the relation tests never share graph state between subtests.
func newGuardedBIV(iv, max Node) Node {
	g := iv.BasePtr().graph
	return g.NewBoundedDimVar("ctor", iv, max)
}

// TestIncludesRequiresNormalization pins the spec's precondition: the
// include relation holds for NORMALIZED (init=0, step=1) IVs only.
func TestIncludesRequiresNormalization(t *testing.T) {
	g := NewGraph("iv-norm")
	dx := g.NewInputParam("ctor", RoleDim, "x", nil)
	dy := g.NewInputParam("ctor", RoleDim, "y", nil)
	f32 := InternScalar(TF32)
	one := g.NewConst("ctor", f32, 1)
	// x normalized (0,1); w unnormalized (1,1).
	x := g.NewDimVar("ctor", dx, g.NewConst("ctor", f32, 0), one)
	w := g.NewDimVar("ctor", dy, one, one)
	bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
	bw := g.NewBoundedDimVar("ctor", w, g.NewConst("ctor", f32, 64))

	if Includes(bx, bw) || Includes(bw, bx) {
		t.Fatal("include must yield false when either IV is not normalized")
	}
	if SameIV(x, w) {
		t.Fatal("different init tuples must not be equal")
	}
}

// TestLoweredVerifyShapes pins the fail-closed shape checks: a bounded
// dimvar binds a dimvar, tiles bind a RoleTensor InputParam, and a
// for-loop binds a dimvar.
func TestLoweredVerifyShapes(t *testing.T) {
	f32 := InternScalar(TF32)

	t.Run("bdimvar rejects non-dimvar iv", func(t *testing.T) {
		g := NewGraph("s1")
		bad := g.NewBoundedDimVar("ctor", g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 64))
		g.NewReturn("ctor", bad)
		assertViolation(t, g, "bounded dimvar operand 0 must be a dimvar")
	})

	t.Run("loadtile rejects non-tensor operand", func(t *testing.T) {
		g := NewGraph("s2")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		lt := g.NewLoadTile("ctor", g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", lt)
		assertViolation(t, g, "tile tensor operand must be a RoleTensor InputParam")
	})

	t.Run("storetile rejects non-tensor operand", func(t *testing.T) {
		g := NewGraph("s3")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		st := g.NewStoreTile("ctor", g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", st)
		assertViolation(t, g, "tile tensor operand must be a RoleTensor InputParam")
	})

	t.Run("forloop rejects non-dimvar iv", func(t *testing.T) {
		g := NewGraph("s4")
		loop := g.NewForLoop("ctor", g.NewConst("ctor", f32, 0),
			g.NewConst("ctor", f32, 64), g.NewConst("ctor", f32, 0))
		g.NewReturn("ctor", loop)
		assertViolation(t, g, "for-loop operand 0 must be a dimvar")
	})
}

// assertViolation fails unless Verify reports a violation whose message
// contains want.
func assertViolation(t *testing.T, g *Graph, want string) {
	t.Helper()
	vs := Verify(g)
	for _, v := range vs {
		if strings.Contains(v.Msg, want) {
			return
		}
	}
	t.Fatalf("verify must report %q, got %v", want, vs)
}

// TestLoweredShapesVerifyClean: the same shapes built CORRECTLY verify
// clean, and the canonical print forms are pinned.
func TestLoweredShapesVerifyClean(t *testing.T) {
	g := NewGraph("lowered")
	f32 := InternScalar(TF32)
	dx := g.NewInputParam("ctor", RoleDim, "x", nil)
	lhs := g.NewInputParam("ctor", RoleTensor, "lhs", InternTensor(TF32, []DimRef{{Name: "x"}}))
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "x"}}))

	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	x := g.NewDimVar("ctor", dx, zero, one)
	bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))

	lt := g.NewLoadTile("ctor", lhs, bx)
	g.NewStoreTile("ctor", lt, out, bx) // auto-registers as an effect root

	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("correctly built lowered graph must verify clean: %v", vs)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	p := Print(g)
	for _, want := range []string{
		"loadtile(lhs[@x/64.0])",
		"storetile(out[@x/64.0], %0)",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("print missing %q:\n%s", want, p)
		}
	}

	// Family classification (D-15).
	if KindLoadTile.Family() != FamilyTileOp || KindStoreTile.Family() != FamilyTileOp {
		t.Error("tile kinds must belong to FamilyTileOp")
	}
	if KindForLoop.Family() != FamilyLoop {
		t.Error("for-loop must belong to FamilyLoop")
	}
	if KindDimVar.Family() != FamilyAbstract || KindBoundedDimVar.Family() != FamilyAbstract {
		t.Error("dimvars are value-level abstract-suite nodes")
	}
}
