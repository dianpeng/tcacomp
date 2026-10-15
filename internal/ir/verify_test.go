package ir

import (
	"fmt"
	"strings"
	"testing"
)

func TestVerifierCatchesCorruption(t *testing.T) {
	g := NewGraph("corrupt")
	f32 := InternScalar(TF32)
	d1 := g.NewConst("ctor", f32, 1.0)
	d2 := g.NewConst("ctor", f32, 2.0)
	x := g.NewBinOp("ctor", BinOpAdd, d1, d1)
	g.AddStatement(x)

	// Corrupt ONE direction of the chain by hand (white-box): point x's
	// first operand at d2 WITHOUT splicing the use into d2's use-list.
	// The use-def side now claims d2; the def-use side knows nothing.
	xn := x.(*binOpNode)
	xn.ops[0].Val = d2

	// Stamp the corrupted node through a legal choke point so the
	// diagnostic can name the last-touched-by pass structurally (D-17).
	ctx := &MutContext{Pass: "test-corrupt"}
	if err := ctx.SetDef(&xn.ops[1], d1); err != nil {
		t.Fatalf("SetDef: %v", err)
	}

	vs := Verify(g)
	if len(vs) == 0 {
		t.Fatal("verifier must catch the deliberately corrupted chain")
	}
	found := false
	for _, v := range vs {
		diag := v.String()
		if v.NodeID == x.BasePtr().ID() &&
			strings.Contains(diag, "test-corrupt") &&
			strings.Contains(diag, v.Invariant) &&
			strings.Contains(diag, fmt.Sprintf("node %d", x.BasePtr().ID())) {
			found = true
		}
	}
	if !found {
		t.Fatalf("no violation names node %d, its last-touched-by pass, and the invariant: %v",
			x.BasePtr().ID(), vs)
	}
}

func TestVerifierFlagsUnwiredOperand(t *testing.T) {
	g := NewGraph("unwired")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)

	// Constructors deliberately do not validate: a nil operand leaves an
	// unwired slot. This is the violation class plan 01-04's
	// TestBuildVerifiesThroughPM relies on.
	x := g.NewNode(KindBinOp, "ctor", a, nil)
	g.AddStatement(x)

	vs := Verify(g)
	found := false
	for _, v := range vs {
		if v.NodeID == x.BasePtr().ID() && v.Invariant == "use-def" && strings.Contains(v.Msg, "nil") {
			found = true
		}
	}
	if !found {
		t.Fatalf("verifier must flag the nil-def operand slot, got %v", vs)
	}
}
