package visualization_test

import (
	"bytes"
	"testing"

	"tcacomp/internal/golden"
	"tcacomp/internal/ir"
	"tcacomp/internal/visualization"
)

// buildDotGraph constructs the fixed small graph pinned for the DOT golden:
// one statement whose chain is an iteration cluster over a root map (one
// tensor operand, identity lambda) feeding a store of the same binding.
// Built exclusively through the ir exported construction surface.
func buildDotGraph() *ir.Graph {
	g := ir.NewGraph("dot_test")
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	f32 := ir.InternScalar(ir.TF32)
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	out := g.NewInputParam("ctor", ir.RoleTensor, "output", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	v := g.NewLambdaInputParam("dot-test", "v", f32)
	operand := g.NewParamRef("dot-test", lhs)
	rootMap := g.NewMap("dot-test", []ir.Node{operand},
		ir.NewLambda([]ir.Node{v}, g.NewParamRef("dot-test", v)),
		[]ir.Node{m})
	ic := g.NewIterCluster("dot-test", []ir.Node{m}, rootMap)
	g.NewStore("dot-test", out, []ir.Node{m}, ic) // auto-registers as an effect root
	if err := g.Materialize(); err != nil {
		panic(err)
	}
	return g
}

func emit(t *testing.T, g *ir.Graph) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := visualization.EmitDot(&buf, g); err != nil {
		t.Fatalf("EmitDot: %v", err)
	}
	return buf.Bytes()
}

func TestDotGolden(t *testing.T) {
	got := emit(t, buildDotGraph())
	golden.Assert(t, "testdata/matmul_scale.dot", got)

	// Determinism (D-05/Pitfall 1): emitting twice yields identical bytes.
	again := emit(t, buildDotGraph())
	if !bytes.Equal(got, again) {
		t.Fatalf("EmitDot not deterministic:\n--- first ---\n%s\n--- second ---\n%s", got, again)
	}
}
