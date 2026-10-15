package ir

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tcacomp/internal/golden"
)

// buildSmallChain builds a fixed chain through the exported construction
// surface: an iteration cluster over a root map (paren-exercising lambda
// body), a reduce, a store, and a return of the stored binding. Two calls
// produce two structurally identical graphs.
func buildSmallChain() *Graph {
	g := NewGraph("chain")
	m := g.AddDimParam("m")
	k := g.AddDimParam("k")
	out := g.AddTensorParam("output", InternTensor(TF32, []DimRef{{Name: "m"}, {Name: "k"}}))
	scale := g.AddScalarParam("scale", InternScalar(TF32))
	f32 := InternScalar(TF32)

	v := Param{name: "v", typ: f32}
	acc := Param{name: "acc", typ: f32}
	now := Param{name: "now", typ: f32}

	one := g.NewConst("test", f32, 1)
	two := g.NewConst("test", f32, 2)
	add := g.NewBinOp("test", BinOpAdd, one, two)
	body := g.NewBinOp("test", BinOpMul, g.NewParamRef("test", v), add)
	scaleRef := g.NewParamRef("test", scale)
	rootMap := g.NewMap("test", []Node{scaleRef}, Lambda{Params: []Param{v}, Body: body}, []Param{m, k})
	ic := g.NewIterCluster("test", []Param{m, k}, rootMap)

	init := g.NewConst("test", f32, 0)
	combBody := g.NewBinOp("test", BinOpAdd, g.NewParamRef("test", acc), g.NewParamRef("test", now))
	red := g.NewReduce("test", init, ic, Lambda{Params: []Param{acc, now}, Body: combBody})

	store := g.NewStore("test", Access{Tensor: out, Axes: []Param{m, k}}, red)
	g.AddStatement(store)
	g.AddStatement(g.NewReturn("test", store))
	return g
}

// smallChainPrint is the exact canonical text of buildSmallChain's graph;
// it pins the stage notation, the `%0` pipe convention (doc §2.6), the
// `%name` binding convention, and FormatConst's `0.0` spelling.
const smallChainPrint = `statement #15:
  iter(@m, @k) -> map(scale, lambda(v): v * (1.0 + 2.0)) as (@m, @k) |
  reduce(init=0.0, %0, lambda(acc, now): acc + now) |
  store(%output[@m][@k], %0)

statement #17:
  return %output`

func TestCanonicalPrint(t *testing.T) {
	g1 := buildSmallChain()
	g2 := buildSmallChain()

	p1 := Print(g1)
	p2 := Print(g2)
	if p1 != p2 {
		t.Fatalf("structurally identical graphs printed differently:\n--- g1 ---\n%s\n--- g2 ---\n%s", p1, p2)
	}
	if want := smallChainPrint + "\n"; p1 != want {
		t.Fatalf("canonical print mismatch:\n--- got ---\n%s\n--- want ---\n%s", p1, want)
	}

	// Numbering must be recomputed per print, never cached (Pitfall 3):
	// repeated prints of the same graph are byte-identical.
	if again := Print(g1); again != p1 {
		t.Fatalf("repeated print differs:\n--- first ---\n%s\n--- second ---\n%s", p1, again)
	}
}

func TestFormatConst(t *testing.T) {
	cases := []struct {
		v    float64
		want string
	}{
		{0, "0.0"},
		{1, "1.0"},
		{2, "2.0"},
		{1.5, "1.5"},
		{-3.25, "-3.25"},
		{1e10, "1e+10"},
	}
	for _, c := range cases {
		if got := FormatConst(c.v); got != c.want {
			t.Errorf("FormatConst(%v) = %q, want %q", c.v, got, c.want)
		}
	}
}

func TestGoldenHarness(t *testing.T) {
	path := filepath.Join("testdata", "golden_smoke.txt")
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	// Matching bytes (modulo trailing-newline normalization) pass.
	if err := golden.Check(path, want); err != nil {
		t.Fatalf("Check against identical bytes: %v", err)
	}

	// Differing bytes fail with a first-differing-line report.
	err = golden.Check(path, []byte("alpha\nbeta\nGAMMA\ndelta\nepsilon\n"))
	if err == nil {
		t.Fatal("Check against different bytes: nil error, want mismatch report")
	}
	msg := err.Error()
	if !strings.Contains(msg, "line 3") || !strings.Contains(msg, "GAMMA") || !strings.Contains(msg, "gamma") {
		t.Fatalf("mismatch report missing line number or context:\n%s", msg)
	}
}

// TestOracleGoldenBytes pins the committed oracle golden to the exact bytes
// lifted from doc/tca.md §5.1 (the <canonical_golden> block of plan 01-03).
// Plan 01-04 replaces this static pin with the dynamic printer comparison
// (TestMatmulScaleGolden); until the builder lands, this test guards the
// doc-pinned file against silent drift (T-1-03-02).
func TestOracleGoldenBytes(t *testing.T) {
	const want = `statement #1:
  iter(@m, @k, @n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) |
  reduce(init=0.0, %0, lambda(acc, now): acc + now) |
  iter() -> map(%0, lambda(v): v * scale) as (@m, @k) |
  store(%output[@m][@k], %0)

statement #2:
  return %output`

	got, err := os.ReadFile(filepath.Join("..", "..", "pkg", "testdata", "matmul_scale.ir"))
	if err != nil {
		t.Fatalf("read oracle golden: %v", err)
	}
	if string(got) != want+"\n" {
		t.Fatalf("oracle golden drifted from doc/tca.md §5.1:\n--- got ---\n%s\n--- want ---\n%s", got, want+"\n")
	}

	// Golden hygiene (T-1-03-01/Pitfall 4): no trailing whitespace, LF only,
	// exactly one trailing newline.
	for i, line := range strings.Split(strings.TrimSuffix(string(got), "\n"), "\n") {
		if strings.TrimRight(line, " \t\r") != line {
			t.Fatalf("golden line %d has trailing whitespace: %q", i+1, line)
		}
	}
	if strings.Contains(string(got), "\r") {
		t.Fatal("golden contains CR bytes; printer emits LF only")
	}
	if strings.HasSuffix(string(got), "\n\n") {
		t.Fatal("golden has more than one trailing newline")
	}
}
