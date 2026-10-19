package analysis

import (
	"fmt"
	"sync"

	"tcacomp/internal/ir"
)

// registry.go is the global analysis registry (quick-261019-jck),
// mirroring internal/opt/registry.go's discipline. Every analysis
// sub-module (sideeffect, ...) self-registers its AnalysisSpec from a
// func init(), so importing the sub-module — directly or through
// internal/analysis/all — is the single act that makes the analysis
// available to Lookup/Registered/BuildAll. The database/sql.Register
// precedent applies: registration happens at program init, so every
// failure mode is a PANIC with a clear message — deterministic, surfaced
// at startup, never a silently missing analysis.
//
// Differences from the opt pass registry, by design (each documented per
// quick-261019-jck):
//
//   - AnalysisSpec has NO Priority and NO Requires: analyses are
//     independent read-only computations, not ordered pipeline steps —
//     registration order is the only stable order, and Registered()
//     returns it unsorted.
//   - The Factory takes the graph: `Factory func(g *ir.Graph) Analysis` —
//     constructing an analysis IS computing it over a snapshot. Factories
//     must tolerate a NIL graph (all-conservative defaults) because
//     Register cross-checks Factory(nil).Name() against spec.Name,
//     exactly as opt.Register cross-checks Factory().ID() against spec.ID.
//   - New import edge analysis → ir: safe, ir never imports analysis.

// AnalysisSpec is one registry entry: everything BuildAll needs to
// construct every analysis without importing a single analysis
// sub-module.
type AnalysisSpec struct {
	// Name is the analysis's identity; it must equal Factory(nil).Name().
	// It is the opt.Cache key, so it is a STABLE public identifier.
	Name string
	// Factory constructs the analysis over g. Read-only per the
	// analysis-package contract. Must tolerate a nil graph (Register
	// cross-checks with Factory(nil); nil graph = all-conservative
	// defaults).
	Factory func(g *ir.Graph) Analysis
}

// registry is the process-global spec table. Registration happens in func
// init()s, but tests may register later, so every access takes the mutex.
var registry = struct {
	sync.Mutex
	specs []AnalysisSpec
}{}

// Register adds spec to the global registry. It panics — with the
// offending name in the message where one exists — on:
//
//   - an empty Name,
//   - a nil Factory,
//   - a Name that is already registered (duplicate registration is almost
//     always a copy-paste of another analysis's init, and failing at init
//     is the cheapest possible diagnosis),
//   - a Factory whose Factory(nil).Name() does not equal spec.Name (the
//     registry would hand out analyses under the wrong name).
func Register(spec AnalysisSpec) {
	registry.Lock()
	defer registry.Unlock()
	if spec.Name == "" {
		panic("analysis: Register called with an empty analysis name")
	}
	if spec.Factory == nil {
		panic(fmt.Sprintf("analysis: Register(%q) called with a nil Factory", spec.Name))
	}
	for _, s := range registry.specs {
		if s.Name == spec.Name {
			panic(fmt.Sprintf("analysis: Register called twice for analysis %q", spec.Name))
		}
	}
	if got := spec.Factory(nil).Name(); got != spec.Name {
		panic(fmt.Sprintf("analysis: Register(%q): Factory(nil).Name() = %q, must equal the spec name", spec.Name, got))
	}
	registry.specs = append(registry.specs, spec)
}

// Lookup returns the registered spec for name, and whether one exists.
func Lookup(name string) (AnalysisSpec, bool) {
	registry.Lock()
	defer registry.Unlock()
	for _, s := range registry.specs {
		if s.Name == name {
			return s, true
		}
	}
	return AnalysisSpec{}, false
}

// Registered returns every registered spec in registration order
// (analyses have no priority — registration order is the only stable
// order). The returned slice is a fresh copy, so callers may reorder it
// freely.
func Registered() []AnalysisSpec {
	registry.Lock()
	defer registry.Unlock()
	return append([]AnalysisSpec(nil), registry.specs...)
}
