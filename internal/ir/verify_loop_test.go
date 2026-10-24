package ir

import (
	"strings"
	"testing"
)

// verify_loop_test.go pins the LIR loop-structure invariant
// (quick-261021-o61 T4): the positive-step invariant (HI-04's IR-level half).
//
// (loop-scope deleted by fix1.md D-c ruling: with no body operand, the
// containment-based scoping check is vacuous. The tiling §8 SCOPE check
// guards the placement-level hazard; membership is via IV references.)

// dimVarOf1 builds a (0, step) u32 dimvar over a fresh dim param.
func dimVarOf1(g *Graph, step float64) Node {
	u32 := InternScalar(TU32)
	dim := g.NewInputParam("ctor", RoleDim, "d", nil)
	return g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, step))
}

// TestForLoopStepPositiveConst pins HI-04's IR-level half: a live ForLoop's
// dimvar step must be a positive constant — zero, negative, and symbolic
// steps are rejected with a loop-step violation naming the loop and dimvar.
func TestForLoopStepPositiveConst(t *testing.T) {
	// buildStep wires one loop whose dimvar's step is stepOf(g).
	buildStep := func(stepOf func(g *Graph) Node) (*Graph, Node) {
		g := NewGraph("step")
		u32 := InternScalar(TU32)
		dim := g.NewInputParam("ctor", RoleDim, "d", nil)
		iv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), stepOf(g))
		loop := g.NewForLoop("ctor", iv,
			g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 64)))
		g.NewReturn("ctor", loop)
		return g, loop
	}
	u32 := InternScalar(TU32)

	for _, tc := range []struct {
		name   string
		stepOf func(g *Graph) Node
		want   string
	}{
		{"zero step", func(g *Graph) Node { return g.NewConst("ctor", u32, 0) }, "positive constant"},
		{"negative step", func(g *Graph) Node { return g.NewConst("ctor", u32, -1) }, "positive constant"},
		{"non-constant step", func(g *Graph) Node {
			one := g.NewConst("ctor", u32, 1)
			return g.NewBinOp("ctor", BinOpAdd, one, one)
		}, "non-constant step"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, loop := buildStep(tc.stepOf)
			vs := Verify(g)
			found := false
			for _, v := range vs {
				if v.Invariant == invLoopStep && v.NodeID == loop.BasePtr().ID() &&
					strings.Contains(v.Msg, tc.want) {
					found = true
				}
			}
			if !found {
				t.Fatalf("verify must report a loop-step violation (%q) naming loop %d, got %v",
					tc.want, loop.BasePtr().ID(), vs)
			}
		})
	}

	t.Run("positive constant step is clean", func(t *testing.T) {
		g, _ := buildStep(func(g *Graph) Node { return g.NewConst("ctor", u32, 32) })
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("a positive constant step must verify clean, got %v", vs)
		}
	})
}
