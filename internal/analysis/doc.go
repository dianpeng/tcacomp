// Package analysis holds read-only analyses over graph snapshots: the
// shared Analysis contract, the global analysis registry, and the
// abstract factory that constructs every registered analysis
// (quick-261019-jck).
//
// Layout:
//
//   - This top level (internal/analysis) holds the machinery SHARED by
//     all analyses: the Analysis contract (analysis.go) and the registry
//   - construct-everything factory (registry.go, factory.go).
//   - Each analysis lives in its own sub-module folder
//     (internal/analysis/sideeffect, ...) and self-registers via a func
//     init() calling analysis.Register with its AnalysisSpec.
//   - internal/analysis/all blank-imports every analysis sub-module;
//     importing it is what fires the registrations, and
//     analysis.BuildAll then returns every analysis over one snapshot.
//   - Sub-modules import internal/analysis and internal/ir, NEVER each
//     other; the top level NEVER imports a sub-module (that is all/'s
//     job) — so the import graph stays acyclic by construction.
package analysis
