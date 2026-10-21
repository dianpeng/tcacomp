package ir

import (
	"fmt"
	"strings"
	"testing"
)

// verify_loop_test.go pins the two LIR loop-structure invariants
// (quick-261021-o61 T4): the loop-IV scoping invariant (the
// CR-02/HI-01/HI-02/ME-03 leakage class of the 261021-mlx review — a fill or
// access reading a loop dimvar from outside that loop's body operand
// subtree) and the positive-step invariant (HI-04's IR-level half).

// loopScopeViolation fails unless Verify reports a loop-scope violation
// naming the loop, its dimvar, and the offending user node.
func loopScopeViolation(t *testing.T, vs []Violation, loop, iv, user Node) Violation {
	t.Helper()
	var got []Violation
	for _, v := range vs {
		if v.Invariant != invLoopScope {
			continue
		}
		got = append(got, v)
		if v.NodeID == user.BasePtr().ID() &&
			strings.Contains(v.Msg, fmt.Sprintf("for-loop node %d", loop.BasePtr().ID())) &&
			strings.Contains(v.Msg, fmt.Sprintf("dimvar node %d", iv.BasePtr().ID())) &&
			strings.Contains(v.Msg, fmt.Sprintf("by node %d", user.BasePtr().ID())) {
			return v
		}
	}
	t.Fatalf("verify must report one loop-scope violation naming loop %d, dimvar %d, user %d, got %v",
		loop.BasePtr().ID(), iv.BasePtr().ID(), user.BasePtr().ID(), got)
	return Violation{}
}

// dimVarOf1 builds a (0, step) u32 dimvar over a fresh dim param.
func dimVarOf1(g *Graph, step float64) Node {
	u32 := InternScalar(TU32)
	dim := g.NewInputParam("ctor", RoleDim, "d", nil)
	return g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, step))
}

// TestForLoopScopeOutlawsTopLevelIVUse pins rejection case (a): the loop's
// IV used OUTSIDE every loop — a top-level expression reading an induction
// value no enclosing loop defines at that position.
func TestForLoopScopeOutlawsTopLevelIVUse(t *testing.T) {
	g := NewGraph("scope-top")
	u32 := InternScalar(TU32)
	one := g.NewConst("ctor", u32, 1)
	iv := dimVarOf1(g, 1)
	loop := g.NewForLoop("ctor", iv,
		g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 64)),
		g.NewConst("ctor", u32, 0)) // body: no IV user inside
	leak := g.NewBinOp("ctor", BinOpAdd, iv, one)
	g.NewReturn("ctor", loop, leak) // anchor both subtrees at the roots

	v := loopScopeViolation(t, Verify(g), loop, iv, leak)
	if !strings.Contains(v.Msg, "operand slot 0") {
		t.Fatalf("the diagnostic must name the offending use's slot, got: %s", v.Msg)
	}
}

// TestForLoopScopeOutlawsSiblingSubtreeUse pins rejection case (b): the IV
// used from a SIBLING loop's body subtree — the cross-nest leakage shape
// (ME-03's rewrite leakage: a node inside one nest referencing another
// nest's loop variables). The sibling loop itself stays clean.
func TestForLoopScopeOutlawsSiblingSubtreeUse(t *testing.T) {
	g := NewGraph("scope-sibling")
	u32 := InternScalar(TU32)
	one := g.NewConst("ctor", u32, 1)
	ivA := dimVarOf1(g, 1)
	ivB := dimVarOf1(g, 1)
	loopA := g.NewForLoop("ctor", ivA,
		g.NewBoundedDimVar("ctor", ivA, g.NewConst("ctor", u32, 64)),
		g.NewConst("ctor", u32, 0))
	leak := g.NewBinOp("ctor", BinOpAdd, ivA, one) // inside B's body, reads A's IV
	loopB := g.NewForLoop("ctor", ivB,
		g.NewBoundedDimVar("ctor", ivB, g.NewConst("ctor", u32, 64)),
		leak)
	g.NewReturn("ctor", loopA, loopB)

	vs := Verify(g)
	loopScopeViolation(t, vs, loopA, ivA, leak)
	for _, v := range vs {
		if v.Invariant == invLoopScope && v.NodeID != leak.BasePtr().ID() {
			t.Fatalf("the sibling loop's own IV use is in scope and must not be flagged: %v", v)
		}
	}
}

// TestForLoopScopeOutlawsNonEnclosingUse pins rejection case (c): the INNER
// loop's IV read from a position inside the outer loop's body subtree but
// OUTSIDE the inner loop (the HI-01 shape — a fill hosted at the between
// level reading a tile-inside dimvar that is undefined there).
func TestForLoopScopeOutlawsNonEnclosingUse(t *testing.T) {
	g := NewGraph("scope-nonenclosing")
	u32 := InternScalar(TU32)
	one := g.NewConst("ctor", u32, 1)
	outerIV := dimVarOf1(g, 4)
	innerIV := dimVarOf1(g, 1)
	inner := g.NewForLoop("ctor", innerIV,
		g.NewBoundedDimVar("ctor", innerIV, g.NewConst("ctor", u32, 8)),
		g.NewConst("ctor", u32, 0))
	// The leak sits in the OUTER body subtree but is NOT a descendant of the
	// inner loop's body operand: the inner dimvar is not defined there.
	leak := g.NewBinOp("ctor", BinOpAdd, innerIV, one)
	combined := g.NewBinOp("ctor", BinOpAdd, inner, leak)
	outer := g.NewForLoop("ctor", outerIV,
		g.NewBoundedDimVar("ctor", outerIV, g.NewConst("ctor", u32, 64)),
		combined)
	g.NewReturn("ctor", outer)

	vs := Verify(g)
	loopScopeViolation(t, vs, inner, innerIV, leak)
	for _, v := range vs {
		if v.Invariant == invLoopScope && v.NodeID != leak.BasePtr().ID() {
			t.Fatalf("the outer loop's own IV use is in scope and must not be flagged: %v", v)
		}
	}
}

// TestForLoopScopeAcceptsLegitimateShapes pins the patterns that MUST pass:
// the loop's own structure (its iv/bound edges and the BoundedDimVar
// wrapper), nested loops' bounds referencing outer IVs, the body-value chain
// reading the IV through its BoundedDimVar, and the tile-lower F8 fill-splice
// shape — a buffer access inside the host loop's body whose per-buffer
// effect chain (ImpactBy) reaches a fill reading an ENCLOSING loop's dimvar
// as its round base.
func TestForLoopScopeAcceptsLegitimateShapes(t *testing.T) {
	t.Run("own structure and nested bound references", func(t *testing.T) {
		g := NewGraph("scope-ok-nest")
		u32 := InternScalar(TU32)
		one := g.NewConst("ctor", u32, 1)
		outerIV := dimVarOf1(g, 4)
		innerIV := dimVarOf1(g, 1)
		// The inner loop's BOUND references the outer IV — legal: the inner
		// loop sits inside the outer's body subtree, bounds included.
		innerBound := g.NewBinOp("ctor", BinOpAdd, outerIV, one)
		inner := g.NewForLoop("ctor", innerIV,
			g.NewBoundedDimVar("ctor", innerIV, innerBound),
			g.NewConst("ctor", u32, 0))
		outer := g.NewForLoop("ctor", outerIV,
			g.NewBoundedDimVar("ctor", outerIV, g.NewConst("ctor", u32, 64)),
			inner)
		g.NewReturn("ctor", outer)
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("nested bound references are in scope and must verify clean, got %v", vs)
		}
	})

	t.Run("body reads the IV through its bounded wrapper", func(t *testing.T) {
		g := NewGraph("scope-ok-body")
		u32 := InternScalar(TU32)
		one := g.NewConst("ctor", u32, 1)
		iv := dimVarOf1(g, 1)
		biv := g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 64))
		body := g.NewBinOp("ctor", BinOpAdd, biv, one) // biv -> iv, inside the body
		loop := g.NewForLoop("ctor", iv, g.NewConst("ctor", u32, 64), body)
		g.NewReturn("ctor", loop)
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("a body reading the IV through its wrapper must verify clean, got %v", vs)
		}
	})

	t.Run("tile-lower fill-splice effect-chain shape", func(t *testing.T) {
		g := NewGraph("scope-ok-fill")
		u32 := InternScalar(TU32)
		outerIV := dimVarOf1(g, 4)
		innerIV := dimVarOf1(g, 1)
		tensor := g.NewInputParam("ctor", RoleTensor, "t", InternTensor(TF32, []DimRef{{Name: "d"}}))
		buf := g.NewFragNew("ctor", g.NewConst("ctor", u32, 32))
		// The fill's source axis reads the OUTER IV; its trailing round base
		// is the outer dimvar itself (the cooperative fill's round base).
		src := g.NewMemLoad("ctor", tensor, []Node{outerIV})
		fill := g.NewFragStore("ctor", src, buf, outerIV)
		// The access sits inside the INNER loop's body and chains after the
		// fill on the buffer's per-tensor effect chain — the F8 splice.
		bufEnv := NewEnv()
		bufEnv.NoteStore(buf, fill)
		access := g.NewFragLoadChained("ctor", bufEnv, buf, innerIV)
		inner := g.NewForLoop("ctor", innerIV,
			g.NewBoundedDimVar("ctor", innerIV, g.NewConst("ctor", u32, 8)),
			access)
		outer := g.NewForLoop("ctor", outerIV,
			g.NewBoundedDimVar("ctor", outerIV, g.NewConst("ctor", u32, 64)),
			inner)
		g.NewReturn("ctor", outer)
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("the fill-splice shape must verify clean (the fill is reachable from the host body through the access's effect chain), got %v", vs)
		}
	})
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
			g.NewBoundedDimVar("ctor", iv, g.NewConst("ctor", u32, 64)),
			g.NewConst("ctor", u32, 0))
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
