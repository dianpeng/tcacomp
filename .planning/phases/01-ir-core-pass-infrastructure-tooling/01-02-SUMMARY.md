---
phase: 01-ir-core-pass-infrastructure-tooling
plan: 02
subsystem: compiler-pass-manager
tags: [go, pass-manager, legality-windows, mlir-patterns, instrumentation, analysis-cache, tdd]

# Dependency graph
requires:
  - phase: 01-01-ir-core
    provides: "internal/ir construction surface, Verify, Kind enum, deterministic BB/operand walks"
provides:
  - "internal/pass: flat Manager with enforced preconditions, abort-on-error, per-pass legality checks, cache invalidation, pipeline-end error propagation"
  - "internal/pass: stage-centered legality window table (StageConstructed row populated; unset stages fail closed) — IR-07 substrate"
  - "internal/pass: Instrumentation + PipelineEnd interfaces with VerifyEach/IRDump/PassTiming standard instrumentations gated by flags.F"
  - "internal/pass: analysis cache with invalidate-all default + preservation whitelist (PASS-03)"
  - "internal/flags: pinned global compiler constants (VerifyEach/DumpAfter/DumpGraphviz/PassTiming)"
  - "internal/analysis: read-only Analysis interface skeleton (PASS-05)"
  - "internal/ir: exported ir.Visit deterministic node enumerator (walk substrate for external packages)"
affects: [01-03-printer-tooling, 01-04-pkg-builder, 01-05-visualization, phase-03-parallelization, phase-04-optimizations]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flat PM verified loop: preconditions -> BeforePass (registration order) -> run -> AfterPass (REVERSE order) -> abort-on-error -> legality -> invalidate"
    - "Fail-closed legality: map PRESENCE decides; WindowFor never returns nil, unset stages get an empty window"
    - "Instrumentation as PM callbacks only — verify/dump/timing never inside passes; flag-gated via flags.F"
    - "Deterministic enumeration via ir.Visit: BB statement order + operand order, visited set, no map iteration"
    - "PassTiming sorts IDs with slices.Sort before reporting — never iterate maps for output"

key-files:
  created:
    - internal/flags/flags.go
    - internal/analysis/analysis.go
    - internal/ir/walk.go
    - internal/pass/context.go
    - internal/pass/manager.go
    - internal/pass/legality.go
    - internal/pass/instrument.go
    - internal/pass/cache.go
    - internal/pass/manager_test.go
    - internal/pass/legality_test.go
    - internal/pass/instrument_test.go
    - internal/pass/cache_test.go
  modified: []

key-decisions:
  - "checkLegality tests drive it through Manager.Run with stub passes (produces StageConstructed / StageLinearized) rather than exporting it — the pinned unexported checker stays unexported and the acceptance greps still match manager.go"
  - "Manager exposes Cache() so tests and callers can pre-seed/observe analyses; NewCache + lazy map init in Put make both &Cache{} and NewCache() usable"
  - "VerifyEach.AfterPipeline error message embeds the first violation's node ID/last-touched-by/invariant so the pipeline-end error names the node (pinned 'naming the node' test)"
  - "IRDump parses DumpAfter with a hand-rolled comma split instead of strings.Split + trim to avoid empty-element edge cases; 'all' short-circuits first"
  - "Stage grew a String() method because the pinned error format 'stage %s' needs a name, not a number"

patterns-established:
  - "Test doubles: stubPass/stubAnalysis/recording/failingEnd live in pass_test files; graphs are built exclusively through the exported ir construction surface"
  - "Flag mutation in tests always pairs with t.Cleanup restore (setVerifyEach/setFlag/setBoolFlag helpers)"
  - "Unwired-graph fixtures use g.NewNode(ir.KindBinOp, \"test\", nil, nil) — constructors deliberately don't validate (PASS-06 owns wiring)"

requirements-completed: [IR-03, IR-07, PASS-01, PASS-02, PASS-03, PASS-04, PASS-05, PASS-06, TOOL-06]

# Metrics
duration: 6min
completed: 2026-10-15
---

# Phase 1 Plan 02: Pass Manager + Legality Windows + Instrumentation Summary

**Flat pass manager enforcing pipeline preconditions, checking stage-centered fail-closed legality windows after every pass, and running verify/dump/timing as PM callbacks — with the conservative analysis cache and pinned global flags.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-10-15T16:35:29Z
- **Completed:** 2026-10-15T16:42Z
- **Tasks:** 3
- **Files modified:** 12 created, 0 modified

## Accomplishments

- Pass-management core (PASS-01/02/03, IR-07): the `Pass` interface exactly as pinned (ID/Requires/Produces/Preserved/Run), `Manager.Run` implementing the MLIR-verified flat loop — precondition check naming both passes, BeforePass in registration order, AfterPass in reverse stack order, abort-on-pass-error ("pass %s: %w"), per-pass `checkLegality`, preservation-whitelist cache invalidation, and pipeline-end `AfterPipeline` dispatch whose first error aborts with "pipeline end: %w" — even for an empty pass list (the D-10 substrate plan 01-04's Build relies on).
- Stage-centered legality windows (IR-07, IR-03): `Stage` enum mirroring the ROADMAP pipeline order, `LegalityWindow{Actions, Predicates}`, central `WindowFor` table with an explicit Legal entry for every Phase-1 kind in StageConstructed and a pinned fail-closed EMPTY window for any unset stage — rejection is map-presence based, so future kinds below KindEnd fail closed until their row lands.
- Structural instrumentation (PASS-02, TOOL-06): `Instrumentation`/`PipelineEnd` interfaces plus `VerifyEach` (per-pass + final verify, renders pinned `verify: node %d (last touched by %q): %s: %s` lines, `--verify-each=0` gate), `IRDump` (DumpAfter comma-list/"all" selection, DumpGraphviz routing, pinned placeholders while emitters stay nil for plan 01-05 to wire), and `PassTiming` (sorted-ID report at pipeline end).
- Conservative analysis cache (PASS-03) keyed by `analysis.Analysis`, invalidate-all default with whitelist re-keep; global flags pinned defaults (PASS-04); read-only `Analysis` skeleton documenting the no-MutContext rule (PASS-05).
- All five fundamental PM behaviors proven by tests: order enforcement, unknown-kind rejection (both >= KindEnd and fail-closed-at-unset-stage), callback order + flag gates, pipeline-end/empty-pipeline verification, preservation invalidation.

## Task Commits

Each task was committed atomically:

1. **Task 1: Global flags defaults + read-only analysis skeleton** - `24db17a` (feat)
2. **Task 2: Pass manager + legality windows + instrumentation + analysis cache** - `ea6b7e4` (feat; includes the Rule 3 `internal/ir/walk.go` addition)
3. **Task 3: PM fundamental tests** - `55cd65a` (test)

## Self-Check: PASSED

- 12/12 claimed files exist on disk
- All 3 task commits present on HEAD: 24db17a, ea6b7e4, 55cd65a
- `go test ./... -count=1` and `go vet ./...` green at HEAD

## Files Created/Modified

- `internal/flags/flags.go` - global F struct with pinned defaults: VerifyEach=true, DumpAfter="", DumpGraphviz=false, PassTiming=false
- `internal/analysis/analysis.go` - read-only Analysis interface (Name()) with the no-MutContext rule in the doc comment
- `internal/ir/walk.go` - exported ir.Visit: deterministic BFS node enumeration (statement order, operand order, lambda bodies as edges, visited set, iterative) — Rule 3 addition, the missing traversal substrate
- `internal/pass/context.go` - pass-local Context{Pass}
- `internal/pass/manager.go` - Pass interface, Manager (NewManager/AddInstrumentation/Cache/Run), checkPreconditions, pipeline-end dispatch
- `internal/pass/legality.go` - Stage enum + String, LegalityAction (Legal/Illegal/Dynamic), LegalityWindow, WindowFor central table (fail-closed empty for unset stages), checkLegality walk
- `internal/pass/instrument.go` - Instrumentation + PipelineEnd interfaces; VerifyEach, IRDump, PassTiming standard instrumentations
- `internal/pass/cache.go` - analysis cache with Get/Put/Invalidate (invalidate-all default)
- `internal/pass/manager_test.go` - stubPass/recording/failingEnd doubles; TestPreconditionRejectsOrder, TestPipelineEndErrorSurfaces
- `internal/pass/legality_test.go` - TestLegalityRejectsUnknownKind (unknown kind, clean oracle, fail-closed unset stage, explicit-Legal table audit)
- `internal/pass/instrument_test.go` - TestInstrumentation (callback order, verify-each gate, IRDump selection, sorted PassTiming)
- `internal/pass/cache_test.go` - TestAnalysisInvalidation (direct + through Manager.Run)

## Decisions Made

- **checkLegality stays unexported, tested through Manager.Run.** The plan's tests reference `checkLegality` by name but the pinned API surface is the Manager; stub passes producing the stage under test exercise the real code path (preconditions → run → legality). The acceptance greps match manager.go either way.
- **Manager.Cache() accessor.** The pinned acceptance test ("a stub pass with Preserved()==['kept'] keeps it through Manager.Run") requires observing the manager-owned cache; an accessor is the minimal additive surface.
- **Stage.String() added.** The pinned error format "stage %s: node %d kind %s illegal in this window" needs a human stage name; the seven-stage enum renders lowercase canonical names ("constructed".."final").
- **VerifyEach pipeline-end error embeds the first violation.** Format: "ir verify failed with %d violation(s), first: node %d (last touched by %q): %s: %s" — satisfies the pinned "naming the node" requirement while Manager.Run wraps it with "pipeline end: %w".

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added internal/ir/walk.go with exported ir.Visit**
- **Found during:** Task 2 (pass manager implementation)
- **Issue:** The plan's behavior pins `checkLegality` walking "each statement's RHS value graph ... operand-wise" from package pass, but plan 01-01's merged IR exports no operand accessor — `nodeImpl.forEachOperand` is unexported and `ir.Use` fields only expose def→user direction (`UsesOf`). Package pass physically cannot enumerate operand nodes without a new exported substrate; the `<interfaces>` block asserted such a traversal exists, but it does not.
- **Fix:** Added `ir.Visit(g, func(Node))` — a 40-line deterministic enumerator (BB statement order, operand order, lambda bodies as edges, visited set, iterative so cyclic graphs can't overflow the stack), deliberately a *pure enumeration* primitive with no checking logic (legality policy stays in package pass; chain checking stays in ir.Verify). New file, additive only; no 01-01 file modified.
- **Files modified:** internal/ir/walk.go (new)
- **Verification:** `go build ./...`, `go vet ./...`, `go test ./... -count=1` all green; legality tests exercise both accept and reject paths through it.
- **Committed in:** ea6b7e4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking, missing referenced API)
**Impact on plan:** The deviation is the minimal substrate the plan assumed; no behavior, signature, or file-ownership change beyond the one additive IR file. Merge note for the orchestrator: parallel wave-2 plans (01-03 printer, 01-05 dot) will likely want the same enumeration — `ir.Visit` is the shared primitive; if their worktrees add a competing helper, keep `Visit` and reconcile call sites at merge.

### Execution adaptations (no Rule 1-4 triggers)

1. **TDD ordering**: Both Task 2 and Task 3 carry `tdd="true"`, but the plan's own task structure puts all test files in Task 3 *after* Task 2's implementation, and Task 2's `<verify>` is build+vet only (no test command). Literal RED-first was therefore impossible without violating the plan's file ownership; tests were written in Task 3 against the pinned behavior block and passed on first run (five behavior tests, four subtests). Recorded here per the 01-01 precedent.
2. **Interface parameter naming**: Go forbids mixing named and unnamed parameters in one list; the pinned `AfterPass(Pass, *ir.Graph, err error)` interface signature is written `AfterPass(p Pass, g *ir.Graph, err error)` — signature-compatible, names are documentation only.
3. **Test construction of lambda-carrying nodes**: legality fixtures use const/binop/store/return (payload-light kinds); `Lambda` payloads need no accessor since `ir.Visit` descends lambda bodies internally.

## Issues Encountered

None beyond the Rule 3 gap above. Full suite green on first run of the tests; only compile-level fixes during implementation (the parameter-naming rule) and gofmt normalization of the two new pass files (the pre-existing `internal/ir/node.go` gofmt complaint from 01-01 was left untouched as out of scope).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-04 (pkg builder) can run `NewManager(nil)` + `VerifyEach` as the single authoritative post-build verification (D-10): `Manager.Run` dispatches `AfterPipeline` on empty pipelines, and `TestPipelineEndErrorSurfaces` proves broken graphs fail with the node named.
- Plan 01-03 (printer) and 01-05 (dot): wire `IRDump.TextEmitter`/`DotEmitter` post-merge; until then the pinned placeholder comments stand in. `ir.Visit` is available for deterministic emission walks (statement order + operand order, lambda bodies included).
- Later-phase lowering passes: declare `Produces()` for the stage whose window should hold after they run; add the stage's window row in the same phase that lands the pass (windows fail closed until the row exists — a pass producing an unset stage is rejected loudly, by design).
- Known intentional stubs: `IRDump` emitters are nil (plan 01-05 wires them); window rows exist only for StageConstructed (Phases 3-4 add rows with their lowering passes); `Context` carries only the pass ID (pass-local scratch arrives with real passes).
- No blockers.

---
*Phase: 01-ir-core-pass-infrastructure-tooling*
*Completed: 2026-10-15*
