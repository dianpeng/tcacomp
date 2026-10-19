package analysis_test

import (
	"strings"
	"testing"

	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"
)

// The registry pins (quick-261019-jck task 1), mirroring
// internal/opt/registry_test.go's contract tests. The registry is
// PROCESS-GLOBAL: these tests never assert exact registry membership or
// absolute size — only membership of specific names and known relative
// order. Fake names are prefixed "jck-fake-" so they never collide with
// the real analysis name ("side-effect").

// stubAnalysis is a local Analysis whose Name is fixed at construction;
// the graph argument is ignored (the registry contract only cares that
// factories construct something with the right Name).
type stubAnalysis struct{ name string }

func (s stubAnalysis) Name() string { return s.name }

// fakeSpec builds a self-consistent spec for name (Factory(nil).Name()
// equals name), the shape Register accepts.
func fakeSpec(name string) analysis.AnalysisSpec {
	return analysis.AnalysisSpec{
		Name:    name,
		Factory: func(g *ir.Graph) analysis.Analysis { return stubAnalysis{name} },
	}
}

// mustRegister registers spec, failing the test on panic with the message.
func mustRegister(t *testing.T, spec analysis.AnalysisSpec) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Register(%q) panicked: %v", spec.Name, r)
		}
	}()
	analysis.Register(spec)
}

// TestRegistryLookupRoundTrip pins the happy path: Register makes a spec
// visible to Lookup under its Name, and unknown names report absent.
func TestRegistryLookupRoundTrip(t *testing.T) {
	mustRegister(t, fakeSpec("jck-fake-a"))
	spec, ok := analysis.Lookup("jck-fake-a")
	if !ok {
		t.Fatal("Lookup(jck-fake-a) not found — Register must make the spec visible")
	}
	if spec.Name != "jck-fake-a" || spec.Factory == nil {
		t.Fatalf("Lookup(jck-fake-a) returned %+v, want the registered spec", spec)
	}
	if _, ok := analysis.Lookup("no-such-analysis"); ok {
		t.Fatal("Lookup(no-such-analysis) found a spec that was never registered")
	}
}

// TestRegistryRegisteredOrderFreshCopy pins the ordering and ownership
// contract: Registered() keeps REGISTRATION order (analyses have no
// priority to sort by) and returns a fresh copy — caller mutations must
// not leak into the registry.
func TestRegistryRegisteredOrderFreshCopy(t *testing.T) {
	mustRegister(t, fakeSpec("jck-fake-1"))
	mustRegister(t, fakeSpec("jck-fake-2"))
	mustRegister(t, fakeSpec("jck-fake-3"))

	specs := analysis.Registered()
	pos := make(map[string]int, len(specs))
	for i, s := range specs {
		pos[s.Name] = i
	}
	if !(pos["jck-fake-1"] < pos["jck-fake-2"] && pos["jck-fake-2"] < pos["jck-fake-3"]) {
		t.Errorf("Registered() must keep registration order, got 1=%d 2=%d 3=%d",
			pos["jck-fake-1"], pos["jck-fake-2"], pos["jck-fake-3"])
	}

	// Fresh copy: swap the returned slice's extremes, re-fetch, and demand
	// every name sits exactly where it sat before the mutation.
	specs[0], specs[len(specs)-1] = specs[len(specs)-1], specs[0]
	again := analysis.Registered()
	if len(again) != len(specs) {
		t.Fatalf("Registered() length changed after caller mutation: %d vs %d", len(again), len(specs))
	}
	for i, s := range again {
		if pos[s.Name] != i {
			t.Fatalf("Registered() is not a fresh copy: %q moved %d -> %d after caller mutation",
				s.Name, pos[s.Name], i)
		}
	}
}

// TestRegistryPanics pins the registration discipline (the opt.Register
// philosophy: every malformed registration PANICS at init with a message
// naming the offender).
func TestRegistryPanics(t *testing.T) {
	expectPanic := func(name, wantSub string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: Register did not panic", name)
			}
			msg, ok := r.(string)
			if !ok {
				t.Fatalf("%s: panic value is %T (%v), want string", name, r, r)
			}
			if !strings.Contains(msg, wantSub) {
				t.Fatalf("%s: panic %q does not mention %q", name, msg, wantSub)
			}
		}()
		fn()
	}

	expectPanic("empty name", `empty analysis name`, func() {
		analysis.Register(analysis.AnalysisSpec{
			Factory: func(g *ir.Graph) analysis.Analysis { return stubAnalysis{"x"} },
		})
	})
	expectPanic("nil Factory", `nil Factory`, func() {
		analysis.Register(analysis.AnalysisSpec{Name: "jck-fake-nil"})
	})
	expectPanic("duplicate name", `jck-fake-dup`, func() {
		mustRegister(t, fakeSpec("jck-fake-dup"))
		analysis.Register(fakeSpec("jck-fake-dup"))
	})
	expectPanic("factory-name mismatch", `jck-fake-wrong`, func() {
		analysis.Register(analysis.AnalysisSpec{
			Name:    "jck-fake-wrong",
			Factory: func(g *ir.Graph) analysis.Analysis { return stubAnalysis{"jck-fake-other"} },
		})
	})
}

// TestBuildAllConstructsEveryRegisteredAnalysis pins the abstract factory:
// BuildAll constructs every registered analysis over the given snapshot,
// keyed by Name(), hands ITS graph to each factory, and tolerates a nil
// graph (one entry per registered spec — factories default conservative).
func TestBuildAllConstructsEveryRegisteredAnalysis(t *testing.T) {
	var got *ir.Graph
	mustRegister(t, analysis.AnalysisSpec{
		Name: "jck-fake-graph",
		Factory: func(g *ir.Graph) analysis.Analysis {
			got = g
			return stubAnalysis{"jck-fake-graph"}
		},
	})

	g := ir.NewGraph("t")
	all := analysis.BuildAll(g)
	a, ok := all["jck-fake-graph"]
	if !ok {
		t.Fatal("BuildAll missing jck-fake-graph — it must construct EVERY registered analysis")
	}
	if a.Name() != "jck-fake-graph" {
		t.Fatalf("BuildAll[jck-fake-graph].Name() = %q, want %q", a.Name(), "jck-fake-graph")
	}
	if got != g {
		t.Fatal("BuildAll must hand its graph to each factory")
	}
	if len(all) != len(analysis.Registered()) {
		t.Fatalf("BuildAll returned %d entries, Registered() has %d", len(all), len(analysis.Registered()))
	}

	// A nil graph is legal: nil passes through, one entry per spec.
	nilAll := analysis.BuildAll(nil)
	if len(nilAll) != len(analysis.Registered()) {
		t.Fatalf("BuildAll(nil) returned %d entries, want one per registered spec (%d)",
			len(nilAll), len(analysis.Registered()))
	}
	if _, ok := nilAll["jck-fake-graph"]; !ok {
		t.Fatal("BuildAll(nil) missing jck-fake-graph — nil-graph factories must still construct")
	}
}
