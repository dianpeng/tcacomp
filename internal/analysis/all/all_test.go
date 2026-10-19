package all_test

import (
	"testing"

	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"

	// The blank import of analysis/all is the act under test: it fires
	// every analysis sub-module's func init() self-registration.
	_ "tcacomp/internal/analysis/all"
)

// TestAllRegistersEveryAnalysis pins the composition root: importing
// analysis/all is the single act that makes the full analysis set
// registry-visible. The registry is process-global, so this asserts
// MEMBERSHIP of specific names only — never exact registry size.
func TestAllRegistersEveryAnalysis(t *testing.T) {
	if _, ok := analysis.Lookup("side-effect"); !ok {
		t.Error("Lookup(side-effect) failed — analysis/all must register every analysis sub-module")
	}
}

// TestBuildAllConstructsSideEffect pins the registered factory's product:
// BuildAll hands back the side-effect analysis under its stable name, its
// value implements the ir.SideEffectAnalysis query face, and that face
// fail-closes on a nil node.
func TestBuildAllConstructsSideEffect(t *testing.T) {
	all := analysis.BuildAll(ir.NewGraph("t"))
	a, ok := all["side-effect"]
	if !ok {
		t.Fatal("BuildAll missing the side-effect analysis under its stable name")
	}
	sea, ok := a.(ir.SideEffectAnalysis)
	if !ok {
		t.Fatalf("the side-effect analysis must implement ir.SideEffectAnalysis, got %T", a)
	}
	if !sea.HasSideEffect(nil) {
		t.Error("HasSideEffect(nil) must be true (fail-closed)")
	}
}
