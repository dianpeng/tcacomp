package pass_test

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
)

// cleanOracle builds a graph of only Phase-1 kinds through the typed
// constructors: two consts, a binop over them, a store, and a return.
func cleanOracle() *ir.Graph {
	g := ir.NewGraph("clean")
	f32 := ir.InternScalar(ir.TF32)
	tensor := g.AddTensorParam("A", ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}}))
	c1 := g.NewConst("test", f32, 1)
	c2 := g.NewConst("test", f32, 2)
	sum := g.NewBinOp("test", ir.BinOpAdd, c1, c2)
	g.AddStatement(sum)
	g.AddStatement(g.NewStore("test", ir.Access{Tensor: tensor}, sum))
	g.AddStatement(g.NewReturn("test", sum))
	return g
}

func TestLegalityRejectsUnknownKind(t *testing.T) {
	// A kind >= KindEnd is rejected even in StageConstructed: no window can
	// know it, and map presence fails closed.
	g := ir.NewGraph("bad")
	bad := g.NewNode(ir.Kind(1<<20), "test")
	g.AddStatement(bad)
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
	var binopID uint64
	ir.Visit(built, func(n ir.Node) {
		if n.KindOf() == ir.KindBinOp {
			binopID = n.BasePtr().ID()
		}
	})
	pm = pass.NewManager([]pass.Pass{&stubPass{id: "noop", produces: pass.StageLinearized}})
	err = pm.Run(built)
	if err == nil {
		t.Fatal("known kind at unset stage: expected fail-closed legality error, got nil")
	}
	for _, want := range []string{"stage linearized", fmt.Sprintf("node %d", binopID)} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("fail-closed error %q missing %q", err.Error(), want)
		}
	}

	// The constructed window names every Phase-1 kind Legal explicitly.
	constructed := pass.WindowFor(pass.StageConstructed)
	for k := ir.KindIter; k <= ir.KindCall; k++ {
		if constructed.Actions[k] != pass.Legal {
			t.Errorf("StageConstructed window missing explicit Legal entry for %s", k)
		}
	}
}
