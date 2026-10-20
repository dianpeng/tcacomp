package ir

import (
	"fmt"
	"strings"
	"testing"
)

// mutate_test.go pins the MutContext.RemoveImpactBy choke point (quick-
// 261020-ejf condition C3): dropping an effect-input (ImpactBy) edge from
// its bearer. The fixtures follow effect_test.go's in-package chained-store
// construction; the choke point itself is the chain-splicing half dce's
// hinge-6 documented as missing ("removing it would need chain splicing").
//
// RemoveImpactBy only DROPS an edge (the chain-first removal case);
// re-pointing a successor at the removed node's predecessor stays a plain
// SetDef splice — the four tests below pin exactly that division of labor.

// removeImpactFixture builds the C3 graph: two chained stores on one tensor
// with the end node wired at the chain head (the second store), so the
// second store's ImpactBy edge is the droppable middle link and the end
// node's edge is the droppable tail observation.
func removeImpactFixture() (g *Graph, m, out, c1, c2, s1, s2, end Node) {
	g = NewGraph("removeimpact")
	f32 := InternScalar(TF32)
	m = g.NewInputParam("ctor", RoleDim, "m", nil)
	out = g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	c1 = g.NewConst("ctor", f32, 1)
	c2 = g.NewConst("ctor", f32, 2)
	env := NewEnv()
	s1 = g.NewMemStoreChained("ctor", env, out, []Node{m}, c1)
	s2 = g.NewMemStoreChained("ctor", env, out, []Node{m}, c2)
	end = g.NewReturnChained("ctor", env, []Node{out}, s2)
	return g, m, out, c1, c2, s1, s2, end
}

// TestMutateRemoveImpactBySplicesChain: the happy path — dropping the
// successor's edge empties its ImpactBy list, kills the record, removes it
// from the predecessor's ref list, stamps both nodes, and leaves Verify
// green. Nothing else moves: node count is unchanged, both stores stay.
func TestMutateRemoveImpactBySplicesChain(t *testing.T) {
	g, _, _, _, _, s1, s2, _ := removeImpactFixture()
	before := len(g.nodes)

	edge := ImpactBy(s2)
	if len(edge) != 1 {
		t.Fatalf("second store must carry exactly 1 ImpactBy edge, got %d", len(edge))
	}
	mctx := &MutContext{Pass: "test-rib"}
	if err := mctx.RemoveImpactBy(edge[0]); err != nil {
		t.Fatalf("RemoveImpactBy: %v", err)
	}

	if got := ImpactBy(s2); len(got) != 0 {
		t.Fatalf("successor must lose its whole ImpactBy list, got %d edges", len(got))
	}
	if !edge[0].IsKilled() {
		t.Fatal("the removed record must read as killed to every handle copy")
	}
	if uses := UsesOf(s1); len(uses) != 0 {
		t.Fatalf("predecessor must lose the record from its ref list, %d use(s) remain", len(uses))
	}
	for name, n := range map[string]Node{"user": s2, "def": s1} {
		if got := n.BasePtr().LastTouchedBy(); got != "test-rib" {
			t.Fatalf("%s node not stamped by the mutation: LastTouchedBy = %q", name, got)
		}
	}
	if got := len(g.nodes); got != before {
		t.Fatalf("RemoveImpactBy drops an edge, never a node: %d nodes before, %d after", before, got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the edge drop: %v", vs)
	}
}

// TestMutateRemoveImpactByNilUse: a zero handle is a deterministic error.
func TestMutateRemoveImpactByNilUse(t *testing.T) {
	mctx := &MutContext{Pass: "test-rib"}
	err := mctx.RemoveImpactBy(SSAUse{})
	if err == nil {
		t.Fatal("RemoveImpactBy(zero handle) must return an error")
	}
	if err.Error() != "ir: RemoveImpactBy on nil use" {
		t.Fatalf("error = %q, want %q", err.Error(), "ir: RemoveImpactBy on nil use")
	}
}

// TestMutateRemoveImpactByWrongSlotClass: a use held in an ordinary OPERAND
// slot is rejected fail-closed with an error naming the user's node ID and
// kind — both for a non-bearer user (a BinOp input) and for an effect
// bearer whose use lives in an operand slot (a store's value operand). The
// graph is untouched in both cases.
func TestMutateRemoveImpactByWrongSlotClass(t *testing.T) {
	g, _, _, c1, c2, _, s2, _ := removeImpactFixture()
	bin := g.NewBinOp("ctor", BinOpAdd, c1, c2)
	mctx := &MutContext{Pass: "test-rib"}

	// The BinOp's input use — an ordinary operand record of a non-bearer.
	var binUse SSAUse
	for _, u := range UsesOf(c1) {
		if u.User() == bin {
			binUse = u
		}
	}
	if binUse.rec == nil {
		t.Fatal("fixture must wire a BinOp input use of c1")
	}
	err := mctx.RemoveImpactBy(binUse)
	if err == nil {
		t.Fatal("RemoveImpactBy on an ordinary operand use must fail")
	}
	for _, want := range []string{
		fmt.Sprintf("node %d", bin.BasePtr().ID()),
		bin.KindOf().String(),
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q must name %q (fail-closed diagnostics)", err.Error(), want)
		}
	}

	// The store's value operand — an operand record of an effect BEARER:
	// the hook must not accept it either (it scans effectIns only).
	var valUse SSAUse
	for _, u := range UsesOf(c2) {
		if u.User() == s2 {
			valUse = u
		}
	}
	if valUse.rec == nil {
		t.Fatal("fixture must wire the store's value operand use of c2")
	}
	if err := mctx.RemoveImpactBy(valUse); err == nil {
		t.Fatal("RemoveImpactBy on a bearer's operand-slot use must fail")
	}
	if got := OperandsOf(s2); len(got) != 3 || got[0] != c2 {
		t.Fatalf("the refused call must leave the operand slot intact, got %v", got)
	}

	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("refused calls must leave the graph verifying: %v", vs)
	}
}

// TestMutateRemoveImpactByReturnEdge: the end node's ImpactBy edge — the
// region's final-state observation — is removable the same way (pins
// returnNode coverage of the uniform hook). The end node here also holds
// the head store as a DATA operand; only the effect edge dies.
func TestMutateRemoveImpactByReturnEdge(t *testing.T) {
	g, _, _, _, _, _, s2, end := removeImpactFixture()

	edges := ImpactBy(end)
	if len(edges) != 1 || edges[0].Def() != s2 {
		t.Fatalf("end node must carry one ImpactBy edge at the chain head, got %v", edges)
	}
	mctx := &MutContext{Pass: "test-rib"}
	if err := mctx.RemoveImpactBy(edges[0]); err != nil {
		t.Fatalf("RemoveImpactBy on the return edge: %v", err)
	}
	if got := ImpactBy(end); len(got) != 0 {
		t.Fatalf("end node must lose its ImpactBy edge, %d remain", len(got))
	}
	// The data operand survives: the head store keeps exactly the return's
	// value use.
	uses := UsesOf(s2)
	if len(uses) != 1 || uses[0].User() != end {
		t.Fatalf("head store must keep exactly the return's data use, got %v", uses)
	}
	if got := end.BasePtr().LastTouchedBy(); got != "test-rib" {
		t.Fatalf("end node not stamped: LastTouchedBy = %q", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("graph must verify after the return-edge drop: %v", vs)
	}
}
