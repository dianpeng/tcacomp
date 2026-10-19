package all_test

import (
	"testing"

	"tcacomp/internal/opt"

	// The blank import of opt/all is the act under test: it fires every pass
	// sub-module's func init() self-registration.
	_ "tcacomp/internal/opt/all"
)

// TestAllRegistersEveryPass pins the composition root: importing opt/all is
// the single act that makes BuildPipeline see the full default pipeline. The
// registry is process-global, so this asserts MEMBERSHIP of specific IDs
// only — never exact pipeline length.
func TestAllRegistersEveryPass(t *testing.T) {
	for _, id := range []string{"dce", "dce-verify", "simplify-lower", "simplify-inline", "simplify-verify", "gvn", "gvn-verify", "schedule"} {
		if _, ok := opt.Lookup(id); !ok {
			t.Errorf("Lookup(%q) failed — opt/all must register every pass sub-module", id)
		}
	}
}

// TestBuildPipelineOrdersDCEBeforeSchedule pins the default order: early
// cleanup (dce, priority 100) runs before late scheduling (schedule, 900).
// Relative index only — the registry is process-global and other tests
// register fakes.
func TestBuildPipelineOrdersDCEBeforeSchedule(t *testing.T) {
	pipe := opt.BuildPipeline()
	dceIdx, schIdx := -1, -1
	for i, p := range pipe {
		switch p.ID() {
		case "dce":
			dceIdx = i
		case "schedule":
			schIdx = i
		}
	}
	if dceIdx < 0 || schIdx < 0 {
		t.Fatalf("default pipeline missing dce (%d) or schedule (%d)", dceIdx, schIdx)
	}
	if dceIdx >= schIdx {
		t.Errorf("dce (index %d) must precede schedule (index %d) in the default pipeline", dceIdx, schIdx)
	}
}
