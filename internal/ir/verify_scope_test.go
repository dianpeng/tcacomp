package ir

import (
	"strings"
	"testing"
)

// TestScopeDominance pins the R-I scope invariant as a VERIFIER check
// (fix3 D-l): a hand-stranded consumer — an item whose operand closure
// references a dimvar whose loop does not dominate the item's block —
// fails VERIFY (so the pass that strands it fails), never only the
// backend emitter (whose fail-closed gate stays as the backstop). The
// check is POST-SCHEDULE only: it reads SchedBlock stamps over the
// attached AST; a graph without an AST is out of scope.
func TestScopeDominance(t *testing.T) {
	u32 := InternScalar(TU32)

	// fixture builds the stranded shape: a loop block L (IV dimvar d)
	// nested in the body Group, and a binop item referencing d placed
	// EITHER inside L (well-scoped) or in the sibling tail of the body
	// Group (stranded).
	fixture := func(stranded bool) *Graph {
		g := NewGraph("scope-dom")
		dp := g.NewInputParam("ctor", RoleDim, "d", nil)
		d := g.NewDimVar("ctor", dp, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		bd := g.NewBoundedDimVar("ctor", d, g.NewConst("ctor", u32, 64))
		loop := g.NewForLoop("ctor", d, bd,
			g.NewKeyedAttribute("ctor", "soc", g.NewConst("ctor", u32, 0)),
			g.NewKeyedAttribute("ctor", "loc", g.NewConst("ctor", u32, 1)))
		lblk, err := NewASTLoop(loop)
		if err != nil {
			t.Fatalf("fixture: NewASTLoop: %v", err)
		}
		loop.BasePtr().SetSched(1, lblk)
		d.BasePtr().SetSched(1, lblk)
		bd.BasePtr().SetSched(1, lblk)

		// The consumer: a binop reading the dimvar (a coordinate-shaped
		// expression — the emitter's dimvar case).
		consumer := g.NewBinOp("ctor", BinOpAdd, d, g.NewConst("ctor", u32, 1))

		body := NewASTGroup()
		body.AppendItem(ASTItem{Block: lblk})
		if stranded {
			// Stranded: the consumer sits in the body Group, OUTSIDE the
			// loop that defines its coordinate.
			body.AppendItem(ASTItem{Node: consumer})
			consumer.BasePtr().SetSched(0, body)
		} else {
			lblk.AppendItem(ASTItem{Node: consumer})
			consumer.BasePtr().SetSched(1, lblk)
		}

		proto := NewASTPrototype()
		proto.AppendItem(ASTItem{Node: dp})
		dp.BasePtr().SetSched(PrototypeSN, proto)
		if err := g.AttachAST(&ASTKernel{Prototype: proto, Body: body}); err != nil {
			t.Fatalf("fixture: AttachAST: %v", err)
		}
		return g
	}

	t.Run("stranded consumer fails the verifier", func(t *testing.T) {
		g := fixture(true)
		vs := Verify(g)
		found := false
		for _, v := range vs {
			if v.Invariant == invScopeDominance &&
				strings.Contains(v.Msg, "references dimvar node") {
				found = true
			}
		}
		if !found {
			t.Fatalf("the stranded consumer must fail the scope-dominance check, got %v", vs)
		}
	})

	t.Run("well-scoped consumer verifies clean", func(t *testing.T) {
		g := fixture(false)
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("the well-scoped consumer must verify clean, got %v", vs)
		}
	})

	t.Run("no AST, no check", func(t *testing.T) {
		// The same stranded data shape WITHOUT an attached AST: the sea's
		// membership is IV-reference based, never positional — nothing to
		// check (fix1.md D-c).
		g := NewGraph("scope-dom-sea")
		dp := g.NewInputParam("ctor", RoleDim, "d", nil)
		d := g.NewDimVar("ctor", dp, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		g.NewBinOp("ctor", BinOpAdd, d, g.NewConst("ctor", u32, 1))
		if vs := Verify(g); len(vs) != 0 {
			t.Fatalf("a sea graph carries no positions; the check must be inactive, got %v", vs)
		}
	})
}
