package schedule

// order_net_internal_test.go pins the two package-private fail-closed nets
// directly (quick-261022-b2m task 3, ME-01): topoRank's dependency-cycle
// refusal (including a genuine self-dependency) and verifyBlockOrder's
// cross-block inversion + leftover refusals. Internal on purpose — the
// functions under test are not part of the package's public face.

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
)

// TestRunDependencyCycle pins topoRank's cycle refusal through Run: a
// chained store made SELF-DEPENDENT (its own ImpactBy def re-pointed at
// itself via the MutContext choke point) is unschedulable — the cycle
// remainder fails closed naming the stuck node, and no AST is attached.
func TestRunDependencyCycle(t *testing.T) {
	g := ir.NewGraph("cyc")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("ctor", ir.RoleDim, "m", nil)
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	env := ir.NewEnv()
	g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 1)) // chain head
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{m}, g.NewConst("ctor", f32, 2))
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)

	ins := ir.ImpactBy(st)
	if len(ins) == 0 {
		t.Fatal("the chained store must carry an ImpactBy use to re-point")
	}
	mctx := &ir.MutContext{Pass: "t"}
	if err := mctx.SetDef(ins[0], st); err != nil {
		t.Fatalf("self-dependency wiring = %v", err)
	}
	err := (Scheduler{}).Run(nil, g)
	if err == nil || !strings.Contains(err.Error(), "dependency cycle") {
		t.Fatalf("Run on a self-dependent store = %v, want the dependency-cycle refusal", err)
	}
	if g.AST() != nil {
		t.Fatal("the refused run must not attach an AST")
	}
}

// TestVerifyBlockOrderInversion pins the ORDER half of the net on a
// hand-built kernel: a node placed in an EARLIER group whose live dep is
// placed in a LATER group is a cross-block inversion — the error names both
// IDs and kinds. This is the check that would have caught CR-01/CR-02.
func TestVerifyBlockOrderInversion(t *testing.T) {
	g := ir.NewGraph("inv")
	f32 := ir.InternScalar(ir.TF32)
	d := g.NewConst("ctor", f32, 1)                                     // the dependency: lands in g2
	n := g.NewBinOp("ctor", ir.BinOpMul, d, g.NewConst("ctor", f32, 2)) // the user: lands in g1

	body := ir.NewASTGroup()
	g1, g2 := ir.NewASTGroup(), ir.NewASTGroup()
	body.AppendItem(ir.ASTItem{Block: g1})
	body.AppendItem(ir.ASTItem{Block: g2})
	g1.AppendItem(ir.ASTItem{Node: n})
	n.BasePtr().SetSched(0, g1)
	g2.AppendItem(ir.ASTItem{Node: d})
	d.BasePtr().SetSched(0, g2)

	nodes := []ir.Node{n, d}
	live := map[ir.Node]bool{n: true, d: true}
	err := verifyBlockOrder(&ir.ASTKernel{Body: body}, nodes, live)
	if err == nil {
		t.Fatal("an inversion (user block before dep block) must fail closed")
	}
	msg := err.Error()
	if !strings.Contains(msg, "cross-block order violated") ||
		!strings.Contains(msg, "before its dependency") {
		t.Fatalf("error = %q, want the cross-block-order message", msg)
	}
	if !strings.Contains(msg, fmt.Sprintf("dependency node %d", d.BasePtr().ID())) {
		t.Fatalf("error = %q, want it to name the dependency's ID", msg)
	}
	if !strings.Contains(msg, n.KindOf().String()) || !strings.Contains(msg, d.KindOf().String()) {
		t.Fatalf("error = %q, want it to name both kinds", msg)
	}
}

// TestVerifyBlockOrderLeftover pins the LEFTOVER half of the net: a live
// node with no landing block fails closed with the sorted-ID message shape
// (the old net's contract, subsumed).
func TestVerifyBlockOrderLeftover(t *testing.T) {
	g := ir.NewGraph("left")
	f32 := ir.InternScalar(ir.TF32)
	placed := g.NewConst("ctor", f32, 1)
	loose := g.NewConst("ctor", f32, 2)

	body := ir.NewASTGroup()
	body.AppendItem(ir.ASTItem{Node: placed})
	placed.BasePtr().SetSched(0, body)
	// loose: live but never placed (SchedBlock nil).

	nodes := []ir.Node{placed, loose}
	live := map[ir.Node]bool{placed: true, loose: true}
	err := verifyBlockOrder(&ir.ASTKernel{Body: body}, nodes, live)
	if err == nil || !strings.Contains(err.Error(), "landed in no block") {
		t.Fatalf("leftover verification = %v, want the landed-in-no-block refusal", err)
	}
	if !strings.Contains(err.Error(), "ids:") {
		t.Fatalf("error = %q, want the sorted id list", err.Error())
	}
}
