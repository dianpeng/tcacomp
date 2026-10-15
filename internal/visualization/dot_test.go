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
//
// Note: a tensor operand is currently referenced via a bare paramRef to the
// tensor param (`lhs`); the axis-structured access encoding (`lhs[@m]`) is
// plan 01-04's builder/printer reconciliation, after which this fixture can
// be rebuilt to match.
func buildDotGraph() *ir.Graph {
	g := ir.NewGraph("dot_test")
	m := g.AddDimParam("m")
	f32 := ir.InternScalar(ir.TF32)
	lhs := g.AddTensorParam("lhs", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	out := g.AddTensorParam("output", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))

	v := ir.NewLambdaParam("v", f32)
	operand := g.NewParamRef("dot-test", lhs)
	rootMap := g.NewMap("dot-test", []ir.Node{operand},
		ir.Lambda{Params: []ir.Param{v}, Body: g.NewParamRef("dot-test", v)},
		[]ir.Param{m})
	ic := g.NewIterCluster("dot-test", []ir.Param{m}, rootMap)
	store := g.NewStore("dot-test", ir.Access{Tensor: out, Axes: []ir.Param{m}}, ic)
	g.AddStatement(store)
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
