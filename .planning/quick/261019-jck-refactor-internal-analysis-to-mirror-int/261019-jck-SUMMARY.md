---
phase: quick-261019-jck
plan: 01
subsystem: analysis-infrastructure
tags: [analysis, refactor, registry, factory, infrastructure]
requires:
  - "internal/opt registry/factory/all pattern (quick-261018-l36)"
  - "accumulated side-effect analysis (quick-261018-nga)"
provides:
  - "analysis.Register/Lookup/Registered + AnalysisSpec (graph-taking factory, opt-mirror panic discipline)"
  - "analysis.BuildAll(g) — every registered analysis over one snapshot, keyed by Name()"
  - "internal/analysis/sideeffect self-registering sub-module (stable name \"side-effect\")"
  - "internal/analysis/all composition root — one blank import makes the full analysis set registry-visible"
affects:
  - "internal/opt/gvn (gvn.go, verify.go) — now consume analysis/sideeffect"
tech-stack:
  added: []
  patterns:
    - "sub-module + func init() self-registration + abstract factory + all/ blank-import composition root (mirrors internal/opt)"
key-files:
  created:
    - internal/analysis/registry.go
    - internal/analysis/registry_test.go
    - internal/analysis/factory.go
    - internal/analysis/all/all.go
    - internal/analysis/all/all_test.go
  modified:
    - internal/analysis/doc.go
    - internal/analysis/analysis.go
    - internal/analysis/sideeffect/sideeffect.go   # git mv from internal/analysis/sideeffect.go
    - internal/analysis/sideeffect/sideeffect_test.go  # git mv from internal/analysis/sideeffect_test.go
    - internal/opt/gvn/gvn.go
    - internal/opt/gvn/verify.go
decisions:
  - "AnalysisSpec has no Priority/Requires — analyses are independent read-only computations; registration order is the only stable order"
  - "Factory takes the graph and must tolerate nil (Register cross-checks Factory(nil).Name() == spec.Name); \"side-effect\" kept as the stable opt.Cache key"
  - "sideeffect_test.go keeps a dual import: top-level analysis stays imported solely for the Analysis-interface satisfaction pin (assertion unweakened)"
metrics:
  duration: "~7 min"
  completed: "2026-10-19"
---

# Phase quick-261019-jck Plan 01: Refactor internal/analysis to mirror internal/opt Summary

**One-liner:** internal/analysis reorganized to the opt pattern — top-level AnalysisSpec registry with opt-identical panic discipline + BuildAll abstract factory, sideeffect moved (git mv) into a self-registering sub-module keeping the stable "side-effect" name, and an all/ blank-import composition root — with gvn consumers updated atomically, zero behavioral change, full module green.

## What Was Built

1. **Top-level analysis infrastructure** (Task 1, TDD):
   - `internal/analysis/registry.go` — `AnalysisSpec{Name, Factory func(g *ir.Graph) Analysis}`, process-global `Register`/`Lookup`/`Registered` in the database/sql.Register panic philosophy: empty name, nil Factory, duplicate name, and `Factory(nil).Name() != spec.Name` all panic at init naming the offender. `Registered()` returns registration-order fresh copies (analyses carry no Priority/Requires by design — documented in the file header).
   - `internal/analysis/factory.go` — `BuildAll(g)` constructs every registered analysis over one snapshot keyed by `Name()` (opt.Cache.Put-compatible so a manager can seed its cache in one loop); nil g passes through to factories.
   - `internal/analysis/registry_test.go` — all seven contract behaviors from the plan pinned (round-trip, order + fresh copy, four panic modes, BuildAll membership/graph-hand-off/nil tolerance). Fake names prefixed `jck-fake-` to never collide with real names; duplicate-panic and order tests are self-contained (no inter-test ordering dependence).
   - `internal/analysis/doc.go` — layout block mirroring opt/doc.go; `analysis.go` — Analysis interface byte-identical, leading comment refreshed.

2. **sideeffect sub-module move** (Task 2, history preserved via git mv, 83%/95% similarity):
   - `internal/analysis/sideeffect/sideeffect.go` — package `sideeffect`; header rewritten into the package doc (own sub-module, import edges analysis/sideeffect → {analysis, ir}); `func init()` registers `AnalysisSpec{Name: "side-effect", Factory: Analyze}`. Every verdict, function, and comment below the header verbatim — zero behavioral change.
   - `internal/analysis/sideeffect/sideeffect_test.go` — package `sideeffect_test`, qualifiers swapped to `sideeffect.`; fixture and every assertion unchanged.
   - Consumers: `internal/opt/gvn/gvn.go` (`sideeffect.Analyze(g)` at the gate) and `internal/opt/gvn/verify.go` (`sideeffect.Analyze(g)`, `se sideeffect.SideEffects`, mirror-path comment updated). `opt/cache.go` / `opt/cache_test.go` untouched and compiling unchanged.

3. **all/ composition root** (Task 3): `internal/analysis/all/all.go` blank-imports the sideeffect sub-module; `all_test.go` pins `Lookup("side-effect")` membership and that BuildAll's product type-asserts to `ir.SideEffectAnalysis` with fail-closed `HasSideEffect(nil)`. Membership-only assertions (process-global registry).

## Verification Results

- `go build ./...` — PASS (whole module)
- `go vet ./...` — PASS (clean)
- `go test ./...` — PASS (full suite, all packages; no assertion weakened)
- Layout — top-level `internal/analysis/` contains exactly: `analysis.go`, `doc.go`, `factory.go`, `registry.go`, `registry_test.go` + `all/` + `sideeffect/` (mirrors internal/opt)
- Old files gone — `internal/analysis/sideeffect.go` / `sideeffect_test.go` no longer exist
- Stale-reference sweep — `grep -rn "analysis/sideeffect\.go" --include="*.go" .` returns nothing
- Import graph verified via `go list -deps`: analysis→ir; sideeffect→{analysis, ir}; all→{sideeffect, analysis, ir}; top level never imports sub-modules — acyclic by construction; `go list -m all` shows stdlib-only still holds

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - consistency] sideeffect_test.go keeps a dual import (plan's mechanical swap was incomplete)**
- **Found during:** Task 2, step 3
- **Issue:** The plan said "change the import to sideeffect, updating the `analysis.` qualifiers to `sideeffect.`" — but the test's `var a analysis.Analysis = mustAnalyze(...)` (the Analysis-interface satisfaction pin) requires the top-level package; a wholesale swap would not compile, and deleting the pin would weaken an assertion, which the same plan forbids.
- **Fix:** Import both `tcacomp/internal/analysis` and `tcacomp/internal/analysis/sideeffect`; all other qualifiers moved to `sideeffect.`. The interface pin survives unweakened.
- **Files modified:** internal/analysis/sideeffect/sideeffect_test.go
- **Commit:** 6e31dc3

**2. [Rule 1 - doc accuracy] gvn.go gate-comment path updated to the new location**
- **Found during:** Task 2, step 4
- **Issue:** The plan said "nothing else changes" in gvn.go, but its comment "the accumulated side-effect analysis (internal/analysis, amendment F)" would have pointed at the old location after the move.
- **Fix:** One-word path update to `internal/analysis/sideeffect, amendment F` (verify.go's mirror-path comment update was already in the plan).
- **Files modified:** internal/opt/gvn/gvn.go
- **Commit:** 6e31dc3

Otherwise the plan executed exactly as written (including the TDD RED-first flow for Task 1).

## Known Stubs

None — no placeholders, no unwired data paths; all infrastructure is live and pinned by tests.

## Threat Flags

None. Pure internal refactor; both registered threats (T-261019-jck-01 registry panics, T-261019-jck-02 BuildAll cost) were `accept` dispositions and introduced no surface beyond the plan.

## Commits

| Task | Commit | Type | Message |
|------|--------|------|---------|
| 1 (RED) | 7fde21d | test | test(quick-261019-jck): pin analysis registry + BuildAll contracts (RED) |
| 1 (GREEN) | b8e6d1f | feat | feat(quick-261019-jck): top-level analysis infrastructure — registry, abstract factory, docs |
| 2 | 6e31dc3 | refactor | refactor(quick-261019-jck): move sideeffect into sub-module with init self-registration |
| 3 | e451b2d | feat | feat(quick-261019-jck): analysis/all composition root blank-importing every analysis sub-module |

## Self-Check: PASSED

All 11 created/modified files exist on disk (including the two git-mv'd sideeffect files at their new paths; old top-level `sideeffect.go`/`sideeffect_test.go` confirmed removed). All four task commits (7fde21d, b8e6d1f, 6e31dc3, e451b2d) present in worktree branch history, each on top of the verified base a450391.
