package pass_test

import (
	"testing"

	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"
	"tcacomp/internal/pass"
)

type stubAnalysis struct{ name string }

func (s stubAnalysis) Name() string { return s.name }

var _ analysis.Analysis = stubAnalysis{}

func TestAnalysisInvalidation(t *testing.T) {
	kept := stubAnalysis{name: "kept"}
	dropped := stubAnalysis{name: "dropped"}

	// Direct cache: invalidate-all default empties everything.
	c := pass.NewCache()
	c.Put(kept)
	c.Put(dropped)
	c.Invalidate(nil)
	if c.Get("kept") != nil || c.Get("dropped") != nil {
		t.Fatalf("Invalidate(nil) must empty the cache, got kept=%v dropped=%v",
			c.Get("kept"), c.Get("dropped"))
	}

	// Preservation whitelist re-keeps only the named analyses.
	c.Put(kept)
	c.Put(dropped)
	c.Invalidate([]string{"kept"})
	if c.Get("kept") == nil {
		t.Fatal("whitelisted analysis must survive Invalidate")
	}
	if c.Get("dropped") != nil {
		t.Fatal("non-whitelisted analysis must be dropped by Invalidate")
	}

	// Through Manager.Run: a pass with Preserved() == ["kept"] keeps it; a
	// pass with an empty whitelist clears everything.
	pm := pass.NewManager([]pass.Pass{&stubPass{id: "preserve", preserved: []string{"kept"}}})
	pm.Cache().Put(kept)
	pm.Cache().Put(dropped)
	if err := pm.Run(ir.NewGraph("t")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if pm.Cache().Get("kept") == nil {
		t.Fatal("preserved analysis must survive Manager.Run")
	}
	if pm.Cache().Get("dropped") != nil {
		t.Fatal("unpreserved analysis must be invalidated by Manager.Run")
	}

	pmClear := pass.NewManager([]pass.Pass{
		&stubPass{id: "keep-pass", preserved: []string{"kept"}},
		&stubPass{id: "clear-pass"},
	})
	pmClear.Cache().Put(kept)
	if err := pmClear.Run(ir.NewGraph("t")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if pmClear.Cache().Get("kept") != nil {
		t.Fatal("a pass with empty Preserved must clear even whitelisted analyses")
	}
}
