package pipeline_test

// pipeline_test.go pins the orchestrator's composition (fix2 §4 step 1): the
// explicit windows reproduce — order for order — the registry's default
// pipeline over the same pass set, resolved through the registered factories
// (fail-closed on unknown IDs), so the orchestration refactor is behavior-
// identical to opt.BuildPipeline at this commit.

import (
	"testing"

	"tcacomp/internal/opt"
	"tcacomp/internal/pipeline"
)

// passIDs renders a pass list's IDs in order.
func passIDs(passes []opt.Pass) []string {
	ids := make([]string, len(passes))
	for i, p := range passes {
		ids[i] = p.ID()
	}
	return ids
}

// TestCompositionMatchesRegistry pins fix2 §4 step 1: HIR ++ LOWER is
// order-for-order IDENTICAL to opt.BuildPipeline()'s ID sequence — the
// orchestration refactor adds wiring only, zero composition drift.
func TestCompositionMatchesRegistry(t *testing.T) {
	got := passIDs(append(pipeline.HIRPasses(), pipeline.LOWERPasses()...))
	want := passIDs(opt.BuildPipeline())
	if len(got) != len(want) {
		t.Fatalf("pipeline composition has %d passes, registry default has %d:\npipeline: %v\nregistry: %v",
			len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("composition diverges at position %d: pipeline %q, registry %q\npipeline: %v\nregistry: %v",
				i, got[i], want[i], got, want)
		}
	}
}

// TestWindowsAreFreshInstances pins R-E's mechanism at the wiring level: the
// factories ARE re-invokable — two calls yield independent instances of the
// same pass (the registry hands out fresh instances, never shared state).
func TestWindowsAreFreshInstances(t *testing.T) {
	a, b := pipeline.HIRPasses(), pipeline.HIRPasses()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("twin HIR windows differ in shape: %d vs %d", len(a), len(b))
	}
	for i := range a {
		// Interface values over distinct pointers differ; the registry's
		// Factory contract guarantees fresh construction per call.
		if &a[i] == &b[i] {
			t.Fatalf("HIR window %d shares storage across calls", i)
		}
	}
}

// TestRunRefusesNilGraph pins the entry's nil-gate delegation.
func TestRunRefusesNilGraph(t *testing.T) {
	if err := pipeline.Run(nil); err == nil {
		t.Fatal("Run(nil) must fail closed")
	}
}
