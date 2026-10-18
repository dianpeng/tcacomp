package pass_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
)

// cleanOracle builds a graph of only Phase-1 kinds through the typed
// constructors: two consts, a binop over them, an env-chained store, and an
// end node carrying the store as its returned value — the whole tree is
// reachable from the sea roots (RoleTensor params + end node, R2/R3).
func cleanOracle() *ir.Graph {
	g := ir.NewGraph("clean")
	f32 := ir.InternScalar(ir.TF32)
	m := g.NewInputParam("test", ir.RoleDim, "m", nil)
	tensor := g.NewInputParam("test", ir.RoleTensor, "A", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	c1 := g.NewConst("test", f32, 1)
	c2 := g.NewConst("test", f32, 2)
	sum := g.NewBinOp("test", ir.BinOpAdd, c1, c2)
	env := ir.NewEnv()
	st := g.NewMemStoreChained("test", env, tensor, []ir.Node{m}, sum)
	g.NewReturnChained("test", env, []ir.Node{tensor}, st)
	return g
}

func TestLegalityRejectsUnknownKind(t *testing.T) {
	// A kind >= KindEnd is rejected even in StageConstructed: no window can
	// know it, and map presence fails closed.
	g := ir.NewGraph("bad")
	bad := g.NewNode(ir.Kind(1<<20), "test")
	g.NewReturn("test", bad)
	pm := pass.NewManager([]pass.Pass{&stubPass{id: "noop", produces: pass.StageConstructed}})
	err := pm.Run(g)
	if err == nil {
		t.Fatal("unknown kind: expected legality error, got nil")
	}
	for _, want := range []string{"stage constructed", fmt.Sprintf("node %d", bad.BasePtr().ID())} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("unknown-kind error %q missing %q", err.Error(), want)
		}
	}

	// Every Phase-1 kind passes StageConstructed.
	pm = pass.NewManager([]pass.Pass{&stubPass{id: "noop", produces: pass.StageConstructed}})
	if err := pm.Run(cleanOracle()); err != nil {
		t.Fatalf("clean oracle at StageConstructed: expected nil, got %v", err)
	}

	// A REAL Phase-1 kind is rejected at an UNSET stage: WindowFor fails
	// closed with an empty window, so absence of the kind's entry from the
	// Actions map rejects it — absence fails closed even for kinds the
	// pipeline fully knows.
	unset := pass.WindowFor(pass.StageLinearized)
	if unset == nil || len(unset.Actions) != 0 {
		t.Fatalf("WindowFor(StageLinearized) must be a fail-closed empty window, got %+v", unset)
	}
	built := cleanOracle()
	var paramID uint64
	ir.Walk(built, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		if paramID == 0 && n.KindOf() == ir.KindInputParam {
			paramID = n.BasePtr().ID()
		}
		return ir.Descend
	}))
	pm = pass.NewManager([]pass.Pass{&stubPass{id: "noop", produces: pass.StageLinearized}})
	err = pm.Run(built)
	if err == nil {
		t.Fatal("known kind at unset stage: expected fail-closed legality error, got nil")
	}
	// Walk roots graph params first, so the FIRST param InputParam is the
	// first illegal node the unset stage's window rejects.
	for _, want := range []string{"stage linearized", fmt.Sprintf("node %d", paramID)} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("fail-closed error %q missing %q", err.Error(), want)
		}
	}

	// The constructed window names every Phase-1 kind Legal explicitly.
	constructed := pass.WindowFor(pass.StageConstructed)
	for k := ir.KindIter; k <= ir.KindForLoop; k++ {
		if constructed.Actions[k] != pass.Legal {
			t.Errorf("StageConstructed window missing explicit Legal entry for %s", k)
		}
	}
}
