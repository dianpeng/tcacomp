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
	g.NewReturn("ctor", x)

	// Corrupt ONE direction of the chain by hand (white-box): retarget x's
	// first operand record at d2 WITHOUT moving it into d2's ref list.
	// The use-def side now claims d2; the def-use side knows nothing.
	xn := x.(*binOpNode)
	xn.ops[0].rec.def = d2.BasePtr().ssa

	// Stamp the corrupted node through a legal choke point so the
	// diagnostic can name the last-touched-by pass structurally (D-17).
	ctx := &MutContext{Pass: "test-corrupt"}
	if err := ctx.SetDef(xn.ops[1], d1); err != nil {
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

func TestVerifierFlagsSurplusOperands(t *testing.T) {
	g := NewGraph("surplus")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)

	// A third operand of a fixed-arity binop cannot be wired, but the
	// construction deliberately does not fail: instead the surplus is
	// recorded and the verifier flags it (fail-closed, WR-01 — never a
	// silent drop).
	x := g.NewNode(KindBinOp, "ctor", a, b, a)
	g.NewReturn("ctor", x)

	if got := len(x.(*binOpNode).ops); got != 2 {
		t.Fatalf("binop wired %d slots, want exactly 2 (surplus never wired)", got)
	}
	vs := Verify(g)
	found := false
	for _, v := range vs {
		if v.NodeID == x.BasePtr().ID() && v.Invariant == "use-def" && strings.Contains(v.Msg, "surplus") {
			found = true
		}
	}
	if !found {
		t.Fatalf("verifier must flag the surplus operands, got %v", vs)
	}

	// Sanity: the exact-arity construction next to it stays clean.
	y := g.NewNode(KindBinOp, "ctor", a, b)
	g.NewReturn("ctor", y)
	foundY := false
	for _, v := range Verify(g) {
		if v.NodeID == y.BasePtr().ID() {
			foundY = true
		}
	}
	if foundY {
		t.Fatal("exact-arity construction must not be flagged")
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
	g.NewReturn("ctor", x)

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

// TestVerifierFlagsKilledUse: a killed ref record still referenced from an
// operand slot is the new use-def corruption class of the handle model
// (directive 2: killed states are representable but checkable). The
// diagnostic must name the node and the slot.
func TestVerifierFlagsKilledUse(t *testing.T) {
	g := NewGraph("killed")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.NewReturn("ctor", sum)

	// White-box corruption: kill slot 1's record in place — no choke point —
	// as if a pass had detached it without dropping the slot.
	sum.(*binOpNode).ops[1].rec.killed = true

	vs := Verify(g)
	found := false
	for _, v := range vs {
		if v.NodeID == sum.BasePtr().ID() && v.Invariant == "use-def" &&
			strings.Contains(v.Msg, "killed") && strings.Contains(v.Msg, "slot 1") {
			found = true
		}
	}
	if !found {
		t.Fatalf("verifier must flag the killed record by node and slot, got %v", vs)
	}
}

// TestVerifierFlagsZeroHandleSlot: a zero handle sitting in an operand slot
// (white-box) must report the SAME "nil def (unwired operand)" message as a
// construction-time unwired slot (D-6 boundary contract).
func TestVerifierFlagsZeroHandleSlot(t *testing.T) {
	g := NewGraph("zerohandle")
	f32 := InternScalar(TF32)
	a := g.NewConst("ctor", f32, 1.0)
	b := g.NewConst("ctor", f32, 2.0)
	sum := g.NewBinOp("ctor", BinOpAdd, a, b)
	g.NewReturn("ctor", sum)

	sum.(*binOpNode).ops[1] = SSAUse{} // white-box: clear the handle

	vs := Verify(g)
	found := false
	for _, v := range vs {
		if v.NodeID == sum.BasePtr().ID() && v.Invariant == "use-def" &&
			strings.Contains(v.Msg, "nil def (unwired operand)") {
			found = true
		}
	}
	if !found {
		t.Fatalf("verifier must flag the zero-handle slot as an unwired operand, got %v", vs)
	}
}
