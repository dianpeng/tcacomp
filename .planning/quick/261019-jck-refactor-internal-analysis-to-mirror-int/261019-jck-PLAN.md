---
phase: quick-261019-jck
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/analysis/analysis.go
  - internal/analysis/doc.go
  - internal/analysis/registry.go
  - internal/analysis/registry_test.go
  - internal/analysis/factory.go
  - internal/analysis/sideeffect/sideeffect.go
  - internal/analysis/sideeffect/sideeffect_test.go
  - internal/analysis/all/all.go
  - internal/analysis/all/all_test.go
  - internal/opt/gvn/gvn.go
  - internal/opt/gvn/verify.go
autonomous: true
requirements: ["QUICK-261019-jck"]
tags: [analysis, refactor, registry, factory, infrastructure]

must_haves:
  truths:
    - "Every analysis pass lives in its own sub-module folder under internal/analysis/ and self-registers into the global analysis registry from a func init() through an abstract factory (today: sideeffect; the layout makes the next analysis a single new folder + one blank import)"
    - "The top-level internal/analysis package contains ONLY shared infrastructure — the Analysis interface, the registry (Register/Lookup/Registered), the abstract factory (BuildAll), and package docs — and never imports a pass sub-module (that is all/'s job), so the import graph stays acyclic by construction"
    - "Importing tcacomp/internal/analysis/all is the single act that makes every analysis available through analysis.Lookup / analysis.Registered / analysis.BuildAll — no other package needs to import an analysis sub-module to get registry visibility"
    - "The registered side-effect analysis keeps its stable name \"side-effect\" (the opt.Cache key), and analysis.Register cross-checks Factory(nil).Name() == spec.Name at init, panicking on empty name, nil Factory, duplicate registration, or name mismatch"
    - "All existing consumers compile and behave identically: opt.Cache/opt.Cache tests use the unchanged top-level analysis.Analysis interface; gvn and gvn-verify construct the side-effect analysis through the moved package (sideeffect.Analyze) with zero behavioral change"
    - "go build ./..., go vet ./..., and go test ./... are all green after the refactor, with no test assertion weakened"
  artifacts:
    - path: "internal/analysis/registry.go"
      provides: "AnalysisSpec (Name + Factory), process-global registry with Register/Lookup/Registered mirroring internal/opt/registry.go's discipline"
      exports: ["AnalysisSpec", "Register", "Lookup", "Registered"]
    - path: "internal/analysis/factory.go"
      provides: "BuildAll(g) — the abstract-factory half: every registered analysis constructed over one graph snapshot, keyed by Name()"
      exports: ["BuildAll"]
    - path: "internal/analysis/sideeffect/sideeffect.go"
      provides: "The accumulated side-effect analysis, moved verbatim from internal/analysis/sideeffect.go, package sideeffect, plus its func init() self-registration"
      exports: ["Analyze", "SideEffects"]
    - path: "internal/analysis/all/all.go"
      provides: "Composition root of the analysis layer: blank-imports every analysis sub-module so init registrations fire"
    - path: "internal/analysis/all/all_test.go"
      provides: "Pin that importing analysis/all registers every analysis sub-module"
    - path: "internal/analysis/registry_test.go"
      provides: "Registry contract pins mirroring internal/opt/registry_test.go"
  key_links:
    - from: "internal/analysis/sideeffect/sideeffect.go"
      to: "internal/analysis registry"
      via: "func init() { analysis.Register(analysis.AnalysisSpec{...}) }"
      pattern: "analysis\\.Register\\(analysis\\.AnalysisSpec"
    - from: "internal/analysis/all/all.go"
      to: "internal/analysis/sideeffect"
      via: "blank import _ \"tcacomp/internal/analysis/sideeffect\""
      pattern: "_ \"tcacomp/internal/analysis/sideeffect\""
    - from: "internal/opt/gvn/gvn.go"
      to: "internal/analysis/sideeffect"
      via: "sideeffect.Analyze(g) consumed as ir.SideEffectAnalysis"
      pattern: "sideeffect\\.Analyze\\(g\\)"
    - from: "internal/opt/gvn/verify.go"
      to: "internal/analysis/sideeffect"
      via: "sideeffect.Analyze(g) and sideeffect.SideEffects parameter type"
      pattern: "sideeffect\\.(Analyze|SideEffects)"
    - from: "internal/analysis/factory.go"
      to: "internal/analysis registry"
      via: "BuildAll calls Registered() then spec.Factory(g) per spec"
      pattern: "spec\\.Factory\\(g\\)"
---

<objective>
Reorganize internal/analysis to mirror internal/opt's organization exactly: each analysis pass in
its own sub-module folder with func-init self-registration and an abstract factory, shared
infrastructure (interface, registry, factory) at the top level, and an `all` pseudo-module that
blank-imports every pass so one import makes the full analysis set registry-visible.

Purpose: the analysis tree currently has one pass (sideeffect.go) sitting loose in the top-level
package, while internal/opt already established the sub-module + registry + all/ pattern
(quick-261018-l36 requirement 9). Making the two trees read identically means the next analysis
lands as "new folder + init + one blank import in all/" with no further plumbing.

Output: refactored internal/analysis (registry.go, factory.go, sideeffect/, all/), atomically
updated consumers in internal/opt/gvn, all tests green.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@internal/opt/registry.go
@internal/opt/factory.go
@internal/opt/doc.go
@internal/opt/all/all.go
@internal/opt/all/all_test.go
@internal/analysis/analysis.go
@internal/analysis/sideeffect.go
@internal/opt/gvn/gvn.go
@internal/opt/gvn/verify.go
@internal/opt/cache.go

<interfaces>
<!-- Exact contracts this refactor creates or moves. Mirror sources are the
     internal/opt files referenced above; read them before writing code. -->

From internal/analysis/analysis.go (UNCHANGED — stays top level):
```go
// An Analysis is a named, read-only computation over a graph snapshot.
type Analysis interface {
	Name() string
}
```

From internal/opt/registry.go (MIRROR SOURCE for discipline, comments, panic
messages — not for fields: analyses have no pipeline Priority and no Requires,
they are independent computations, not ordered pipeline steps):
```go
type PassSpec struct {
	ID       string
	Priority int
	Requires []string
	Factory  func() Pass
}
func Register(spec PassSpec)   // panics: empty ID, nil Factory, duplicate ID, Factory().ID() != spec.ID
func Lookup(id string) (PassSpec, bool)
func Registered() []PassSpec   // priority-ascending fresh copy
```

NEW: internal/analysis/registry.go contract (this plan creates it):
```go
package analysis

import (
	"sync"
	"tcacomp/internal/ir"
)

type AnalysisSpec struct {
	// Name is the analysis's identity; it must equal Factory(nil).Name().
	// It is the opt.Cache key, so it is a STABLE public identifier.
	Name string
	// Factory constructs the analysis over g. Read-only per the
	// analysis-package contract. Must tolerate a nil graph (Register
	// cross-checks with Factory(nil); nil graph = all-conservative defaults).
	Factory func(g *ir.Graph) Analysis
}

func Register(spec AnalysisSpec)                    // same panic philosophy as opt.Register
func Lookup(name string) (AnalysisSpec, bool)
func Registered() []AnalysisSpec                    // registration-order fresh copy (no priority to sort by)
```

NEW: internal/analysis/factory.go contract:
```go
// BuildAll constructs every registered analysis over g, keyed by Name().
func BuildAll(g *ir.Graph) map[string]Analysis
```

MOVED: internal/analysis/sideeffect/sideeffect.go — package sideeffect, contents
of the current internal/analysis/sideeffect.go VERBATIM (zero behavioral change:
SideEffects, Analyze, state, effectfulOf, hasEffectfulChild, baseVerdict,
orderSensitive all unchanged), plus:
```go
func init() {
	analysis.Register(analysis.AnalysisSpec{
		Name:    "side-effect", // MUST stay "side-effect": opt.Cache key + SideEffects.Name()
		Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
	})
}
```

Consumers to update atomically (exhaustive — verified by repo-wide grep):
- internal/opt/cache.go, internal/opt/cache_test.go — use analysis.Analysis only; NO CHANGE (interface stays top level)
- internal/opt/gvn/gvn.go:101 — analysis.Analyze(g) -> sideeffect.Analyze(g)
- internal/opt/gvn/verify.go:85,179 — analysis.Analyze(g) -> sideeffect.Analyze(g); analysis.SideEffects -> sideeffect.SideEffects
- No other .go file imports tcacomp/internal/analysis; no doc/ file references it

Import-cycle proof: analysis(top) -> ir; analysis/sideeffect -> analysis + ir;
analysis/all -> sideeffect; opt/gvn -> analysis/sideeffect + opt + ir. ir never
imports analysis (established), top-level analysis never imports sub-modules.
Acyclic.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create top-level analysis infrastructure — registry, abstract factory, docs</name>
  <files>internal/analysis/registry.go, internal/analysis/registry_test.go, internal/analysis/factory.go, internal/analysis/doc.go, internal/analysis/analysis.go</files>
  <behavior>
    Pinned by internal/analysis/registry_test.go (write it FIRST, mirroring internal/opt/registry_test.go's contract tests with a local stub analysis):
    - Test 1: Register then Lookup(name) returns the spec, true; Lookup of an unknown name returns false.
    - Test 2: Register with an empty Name panics.
    - Test 3: Register with a nil Factory panics.
    - Test 4: Register twice with the same Name panics.
    - Test 5: Register whose Factory(nil).Name() != spec.Name panics (wrong-name factory).
    - Test 6: Registered() returns specs in registration order, and mutating the returned slice does not affect the registry (fresh copy).
    - Test 7: BuildAll(g) over ir.NewGraph("t") with one registered stub returns a map keyed by the stub's Name() containing the constructed analysis; BuildAll over a nil graph still returns one entry per registered spec (factories must tolerate nil).
  </behavior>
  <action>
    Write registry_test.go first (RED), then implement (GREEN). This task implements locked decision [2].

    1. internal/analysis/registry.go — mirror internal/opt/registry.go's structure and comment voice (database/sql.Register precedent: registration happens at init, every failure mode PANICS with a clear message — deterministic, surfaced at startup). Copy its mutex-guarded anonymous-struct registry shape exactly. Differences from opt, each documented in the file header:
       - AnalysisSpec has NO Priority and NO Requires: analyses are independent read-only computations, not ordered pipeline steps — registration order is the only stable order.
       - The Factory takes the graph: `Factory func(g *ir.Graph) Analysis` — constructing an analysis IS computing it over a snapshot. Document the nil-graph contract (Factory must tolerate nil; Register cross-checks Factory(nil).Name() against spec.Name, exactly as opt.Register cross-checks Factory().ID()).
       - Registered() returns a fresh copy in registration order (no sort — nothing to sort by).
       - New import edge analysis -> ir: safe, ir never imports analysis.
    2. internal/analysis/factory.go — mirror internal/opt/factory.go (the abstract-factory half, per locked decision [1]/[2]): `BuildAll(g *ir.Graph) map[string]Analysis` iterates Registered() and calls spec.Factory(g), keyed by spec.Name(). A nil g is legal — it passes nil through to each factory. Document that map-key-by-Name matches opt.Cache.Put's bookkeeping, so a manager can seed its cache from BuildAll in one loop.
    3. internal/analysis/doc.go — rewrite the package doc to mirror internal/opt/doc.go's layout block: top level holds the shared machinery (Analysis contract in analysis.go, registry + factory in registry.go/factory.go); each analysis lives in its own sub-module folder (internal/analysis/sideeffect, ...) and self-registers via a func init() calling analysis.Register with its AnalysisSpec; internal/analysis/all blank-imports every analysis sub-module; sub-modules import internal/analysis and internal/ir, NEVER each other; the top level NEVER imports sub-modules (all/'s job) — acyclic by construction.
    4. internal/analysis/analysis.go — keep the Analysis interface byte-identical (opt/cache.go depends on it); refresh only its leading comment to note the interface is the shared contract all sub-modules implement and the registry/BuildAll construct against.
    Do NOT move or touch sideeffect.go in this task.
  <verify>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go test ./internal/analysis/</verify>
  <done>internal/analysis has Register/Lookup/Registered/BuildAll with the full opt-style panic discipline pinned by tests; the tree builds; existing sideeffect tests still pass untouched.</done>
</task>

<task type="auto">
  <name>Task 2: Move sideeffect into its own sub-module with init self-registration; update gvn consumers atomically</name>
  <files>internal/analysis/sideeffect/sideeffect.go, internal/analysis/sideeffect/sideeffect_test.go, internal/opt/gvn/gvn.go, internal/opt/gvn/verify.go</files>
  <action>
    This task implements locked decision [1] (own sub-module folder + func init registration + abstract factory). Preserve file history with git mv:

    1. `git mv internal/analysis/sideeffect.go internal/analysis/sideeffect/sideeffect.go` and `git mv internal/analysis/sideeffect_test.go internal/analysis/sideeffect/sideeffect_test.go`.
    2. sideeffect.go: change the package clause to `package sideeffect` with a package doc comment ("Package sideeffect implements the accumulated side-effect analysis..."). Add the import `"tcacomp/internal/analysis"` and the func init() from the <interfaces> block — Name MUST be "side-effect" (it is SideEffects.Name() and the opt.Cache key; Register's cross-check enforces the match). Rewrite ONLY the stale header prose that said "It lives here, in the EXISTING internal/analysis package (amendment F)": it now lives in its own sub-module, still implementing the analysis.Analysis contract the internal/opt cache builds on; keep the import-edge/no-cycle note, updated to analysis/sideeffect -> {analysis, ir}. Every function, verdict, and comment below the header stays VERBATIM — zero behavioral change.
    3. sideeffect_test.go: change `package analysis_test` to `package sideeffect_test` and the import `"tcacomp/internal/analysis"` to `"tcacomp/internal/analysis/sideeffect"`, updating the `analysis.` qualifiers to `sideeffect.`. Do not weaken any assertion; the fixture and all cases move unchanged.
    4. internal/opt/gvn/gvn.go: swap the import to `"tcacomp/internal/analysis/sideeffect"`; line 101 `se := analysis.Analyze(g)` becomes `se := sideeffect.Analyze(g)`. (se is consumed only through HasSideEffect — ir.SideEffectAnalysis — so nothing else changes.)
    5. internal/opt/gvn/verify.go: same import swap; line 85 `analysis.Analyze(g)` -> `sideeffect.Analyze(g)`; line 179 `func hasEffectfulChildMirror(n ir.Node, se analysis.SideEffects)` -> `se sideeffect.SideEffects`; update the line-43 comment's path "internal/analysis/sideeffect.go's" to "internal/analysis/sideeffect/sideeffect.go's".
    6. Leave internal/ir comments (value.go, node.go, effect.go) and internal/opt/cache.go untouched — they reference the analysis package/interface, which still exists at the top level.
  <verify>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./internal/... && go test ./internal/analysis/... ./internal/opt/... && ! test -f internal/analysis/sideeffect.go && ! test -f internal/analysis/sideeffect_test.go</verify>
  <done>The side-effect analysis is a self-registering sub-module; the old top-level pass files are gone; gvn and gvn-verify compile against the moved package; every analysis and opt test passes with assertions unchanged.</done>
</task>

<task type="auto">
  <name>Task 3: Create the all/ composition root and verify the whole module</name>
  <files>internal/analysis/all/all.go, internal/analysis/all/all_test.go</files>
  <action>
    This task implements locked decision [3], mirroring internal/opt/all exactly.

    1. internal/analysis/all/all.go — copy the shape of internal/opt/all/all.go: package doc naming it "the composition root of the analysis layer" (blank-imports every analysis sub-module so their func init() self-registrations fire; importing tcacomp/internal/analysis/all is the single act that makes analysis.Lookup/Registered/BuildAll see the full analysis set), package all, blank import `_ "tcacomp/internal/analysis/sideeffect"` with the note that every new analysis sub-module's blank import accumulates here.
    2. internal/analysis/all/all_test.go — mirror internal/opt/all/all_test.go: package all_test importing opt's counterpart style — here `"tcacomp/internal/analysis"` plus the blank import `_ "tcacomp/internal/analysis/all"` as the act under test. Assert (a) `analysis.Lookup("side-effect")` succeeds, (b) `analysis.BuildAll(ir.NewGraph("t"))` contains the "side-effect" key whose value type-asserts to `ir.SideEffectAnalysis` and whose HasSideEffect(nil) is true (fail-closed). Assert MEMBERSHIP only, never registry size — the registry is process-global.
    3. Final sweep: `go vet ./...` and `go test ./...` over the entire module; confirm no stale references with `grep -rn "analysis/sideeffect\.go" --include="*.go" .` returning only the updated `sideeffect/sideeffect.go` path matches.
  <verify>cd /home/ps/Workspace/cuda/project/tcacomp && go vet ./... && go test ./...</verify>
  <done>Importing analysis/all registers every analysis sub-module (pinned by test); go vet and the full test suite are green across the whole module.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | Pure internal refactor: no new inputs, outputs, or external interfaces cross a trust boundary in this change |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-261019-jck-01 | Tampering | analysis registry (Register panics) | accept | Panic-on-bad-registration is the opt-established pattern: init-time failures are deterministic and surface at startup; no untrusted input reaches Register (all callers are in-repo init functions) |
| T-261019-jck-02 | Denial of Service | BuildAll(g) | accept | Constructs every registered analysis over one snapshot — bounded by in-repo registry contents, same cost class as the existing per-pass analysis.Analyze calls |
</threat_model>

<verification>
- go build ./... — whole module compiles
- go vet ./... — clean
- go test ./... — full suite green, no assertions weakened
- Layout mirrors opt: sideeffect/ and all/ subfolders exist; top-level analysis/ contains only analysis.go, doc.go, registry.go, factory.go, registry_test.go
- Registry contract: Register/Lookup/Registered with opt-identical panic discipline, pinned by registry_test.go
- Consumers: opt/cache.go and opt/cache_test.go compile unchanged; gvn/gvn.go and gvn/verify.go use sideeffect.Analyze / sideeffect.SideEffects
</verification>

<success_criteria>
- internal/analysis tree is organized exactly like internal/opt: shared infra top-level, one folder per analysis pass with func-init self-registration through an abstract factory (AnalysisSpec.Factory), and an all/ pseudo-module whose single import makes every analysis registry-visible
- The stable name "side-effect" is preserved (opt.Cache key continuity)
- All existing tests pass unweakened; go vet clean; import graph acyclic (top-level analysis imports only ir; sub-modules import analysis + ir, never each other)
</success_criteria>

<output>
After completion, create `.planning/quick/261019-jck-refactor-internal-analysis-to-mirror-int/261019-jck-SUMMARY.md`
</output>
