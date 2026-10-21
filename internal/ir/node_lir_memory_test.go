package ir

import (
	"strings"
	"testing"
)

// TestLIRMemoryVerifyShapes pins the fail-closed shape checks of the LIR
// memory family (D-3): SMEM accesses bind a MemNewSMem buffer, FRAG accesses
// bind a FragNew buffer.
func TestLIRMemoryVerifyShapes(t *testing.T) {
	f32 := InternScalar(TF32)

	t.Run("memloadsmem rejects non-smem src", func(t *testing.T) {
		g := NewGraph("s2")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		lt := g.NewMemLoadSMem("ctor", g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", lt)
		assertViolation(t, g, "smem buffer operand must be a MemNewSMem node")
	})

	t.Run("memstoresmem rejects non-smem dst", func(t *testing.T) {
		g := NewGraph("s3")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		st := g.NewMemStoreSMem("ctor", g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", st)
		assertViolation(t, g, "smem buffer operand must be a MemNewSMem node")
	})

	t.Run("fragload rejects non-frag src", func(t *testing.T) {
		g := NewGraph("s5")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		fl := g.NewFragLoad("ctor", g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", fl)
		assertViolation(t, g, "frag buffer operand must be a FragNew node")
	})

	t.Run("fragstore rejects non-frag dst", func(t *testing.T) {
		g := NewGraph("s6")
		x := g.NewDimVar("ctor", g.NewInputParam("ctor", RoleDim, "x", nil),
			g.NewConst("ctor", f32, 0), g.NewConst("ctor", f32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))
		st := g.NewFragStore("ctor", g.NewConst("ctor", f32, 1), g.NewConst("ctor", f32, 0), bx)
		g.NewReturn("ctor", st)
		assertViolation(t, g, "frag buffer operand must be a FragNew node")
	})
}

// TestLIRStoreAnnotations pins the LIR fill-store annotation convention
// (LO-01, quick-261021-o61 T5): a MemStoreSMem/FragStore's trailing
// KeyedAttribute operands (tile-lower's loadtile loader-selection variant)
// render as the DISTINCT attribute suffix `[key=value]` — never through the
// keyedattribute value spelling that would read as a coordinate subscript —
// and the verifier pins them well-formed: at most one per key, each with a
// const value.
func TestLIRStoreAnnotations(t *testing.T) {
	u32 := InternScalar(TU32)

	t.Run("annotation renders as attribute suffix", func(t *testing.T) {
		g := NewGraph("fillattr")
		dx := g.NewInputParam("ctor", RoleDim, "x", nil)
		x := g.NewDimVar("ctor", dx, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", u32, 64))
		smem := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		attr := g.NewKeyedAttribute("ctor", "loadtile", g.NewConst("ctor", u32, 1))
		sts := g.NewMemStoreSMem("ctor", g.NewConst("ctor", u32, 0), smem, bx, attr)
		g.NewReturn("ctor", sts)
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("a store with one well-formed annotation must verify clean: %v", vs)
		}
		if err := g.Materialize(); err != nil {
			t.Fatalf("Materialize: %v", err)
		}
		p := Print(g)
		if !strings.Contains(p, "[@x/64][loadtile=1]") {
			t.Errorf("the annotation must render as the distinct attribute suffix after the coordinates:\n%s", p)
		}
		if strings.Contains(p, "keyedattribute(") {
			t.Errorf("an annotation must never render through the keyedattribute value spelling (a coordinate-subscript look):\n%s", p)
		}
	})

	t.Run("duplicate key rejected", func(t *testing.T) {
		g := NewGraph("dupattr")
		dx := g.NewInputParam("ctor", RoleDim, "x", nil)
		x := g.NewDimVar("ctor", dx, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", u32, 64))
		frag := g.NewFragNew("ctor", g.NewConst("ctor", u32, 64))
		fst := g.NewFragStore("ctor", g.NewConst("ctor", u32, 0), frag, bx,
			g.NewKeyedAttribute("ctor", "loadtile", g.NewConst("ctor", u32, 2)),
			g.NewKeyedAttribute("ctor", "loadtile", g.NewConst("ctor", u32, 1)))
		g.NewReturn("ctor", fst)
		assertViolation(t, g, "repeats KeyedAttribute key")
	})

	t.Run("non-const annotation value rejected", func(t *testing.T) {
		g := NewGraph("ncattr")
		dx := g.NewInputParam("ctor", RoleDim, "x", nil)
		x := g.NewDimVar("ctor", dx, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", u32, 64))
		smem := g.NewMemNewSMem("ctor", g.NewConst("ctor", u32, 64))
		sts := g.NewMemStoreSMem("ctor", g.NewConst("ctor", u32, 0), smem, bx,
			g.NewKeyedAttribute("ctor", "loadtile", x)) // a dimvar, not a const
		g.NewReturn("ctor", sts)
		assertViolation(t, g, `trailing KeyedAttribute "loadtile" must carry a const value`)
	})
}

// TestLIRMemoryShapesVerifyClean: the LIR memory family built CORRECTLY
// verifies clean, the storage-class metadata is bound by the constructors,
// and the canonical print forms (D-6 lowercase kind names) are pinned. The
// LIR values are tied to the observable end node through a GMEM store so the
// graph is complete under both the effect-root and the one-end-node (R2)
// liveness models.
func TestLIRMemoryShapesVerifyClean(t *testing.T) {
	g := NewGraph("lowered")
	f32 := InternScalar(TF32)
	dx := g.NewInputParam("ctor", RoleDim, "x", nil)
	out := g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "x"}}))

	zero := g.NewConst("ctor", f32, 0)
	one := g.NewConst("ctor", f32, 1)
	x := g.NewDimVar("ctor", dx, zero, one)
	bx := g.NewBoundedDimVar("ctor", x, g.NewConst("ctor", f32, 64))

	smem := g.NewMemNewSMem("ctor", g.NewConst("ctor", f32, 64))
	lt := g.NewMemLoadSMem("ctor", smem, bx)
	sts := g.NewMemStoreSMem("ctor", lt, smem, bx)

	frag := g.NewFragNew("ctor", g.NewConst("ctor", f32, 64))
	fl := g.NewFragLoad("ctor", frag, bx)
	fst := g.NewFragStore("ctor", fl, frag, bx)

	// GMEM store tying the LIR stored values to the observable end node.
	sum := g.NewBinOp("ctor", BinOpAdd, sts, fst)
	env := NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []Node{dx}, sum)
	g.NewReturn("ctor", st)

	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("correctly built lowered graph must verify clean: %v", vs)
	}
	if err := g.Materialize(); err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	// StorageClass metadata (X-4): constructors bind the tier; StorageClassOf
	// is the read face.
	for n, want := range map[Node]StorageClass{
		lt:   StorageSMem,
		sts:  StorageSMem,
		smem: StorageSMem,
		fl:   StorageFrag,
		fst:  StorageFrag,
		frag: StorageFrag,
	} {
		got, ok := StorageClassOf(n)
		if !ok || got != want {
			t.Errorf("StorageClassOf(%s) = %v, %v; want %v, true", n.KindOf(), got, ok, want)
		}
	}
	if _, ok := StorageClassOf(st); ok {
		t.Error("HIR GMEM nodes carry no StorageClass metadata")
	}

	p := Print(g)
	for _, want := range []string{
		"memstoresmem(%2[@x/64.0], %1)",
		"fragstore(%12[@x/64.0], %11)",
		"memstore(%out[@x], %0 + %10)",
		"return %out",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("print missing %q:\n%s", want, p)
		}
	}

	// The allocation/load forms are not stage positions (they render as
	// operand references inside a store's statement), so the canonical LABEL
	// rendering pins them — stageText renders every LIR memory form with its
	// lowercase kind name (D-6).
	num := ComputeNumbering(g)
	for _, check := range []struct {
		n    Node
		want string
	}{
		{smem, "memnewsmem(64.0)"},
		{lt, "memloadsmem(%2[@x/64.0])"},
		{sts, "memstoresmem(%2[@x/64.0], %1)"},
		{frag, "fragnew(64.0)"},
		{fl, "fragload(%12[@x/64.0])"},
		{fst, "fragstore(%12[@x/64.0], %11)"},
	} {
		if got := Label(check.n, num); got != check.want {
			t.Errorf("Label = %q, want %q", got, check.want)
		}
	}

	// Family classification: the whole LIR memory family is FamilyMemory.
	for _, k := range []Kind{KindMemLoadSMem, KindMemStoreSMem, KindMemNewSMem,
		KindFragLoad, KindFragStore, KindFragNew} {
		if k.Family() != FamilyMemory {
			t.Errorf("%s must belong to FamilyMemory, got %v", k, k.Family())
		}
	}
	if KindForLoop.Family() != FamilyLoop {
		t.Error("for-loop must belong to FamilyLoop")
	}
	if KindDimVar.Family() != FamilyAbstract || KindBoundedDimVar.Family() != FamilyAbstract {
		t.Error("dimvars are value-level abstract-suite nodes")
	}
}
