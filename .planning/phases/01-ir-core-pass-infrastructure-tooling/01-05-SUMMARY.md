---
phase: 01-ir-core-pass-infrastructure-tooling
plan: 05
subsystem: tooling
tags: [pass-manager, instrumentation, ir-dump, graphviz, flags]

# Dependency graph
requires:
  - phase: 01-02
    provides: "pass.Manager/Instrumentation contract, IRDump TextEmitter/DotEmitter callback fields (landed nil), VerifyEach, PassTiming, flags.F gates"
  - phase: 01-03
    provides: "ir.Print canonical text emitter and visualization.EmitDot graphviz emitter"
provides:
  - "pkg.DefaultInstrumentations(w) — the one-constructor standard instrumentation set (VerifyEach + IRDump + PassTiming) with real dump emitters wired"
  - "End-to-end proof that --dump-after emits real ir.Print content and --dump-graphviz emits real EmitDot digraphs through Manager.Run (TOOL-06 closed)"
affects: [phase-05-cli, tooling]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Public pkg/ constructor composes internal/ instrumentations and binds their callback fields (emitters) — the wiring layer pattern for cross-wave post-merge integration"

key-files:
  created:
    - pkg/instrument.go
    - pkg/instrument_test.go
  modified: []

key-decisions:
  - "DotEmitter adapter ignores EmitDot's error and returns the partial buffer: EmitDot is emit-only (nil graph returns nil) and cannot fail against a strings.Builder; a dump must never panic or abort the pipeline (T-1-05-03)"
  - "Statement header asserted dynamically via stmt.ID() instead of the plan's literal `statement #1`: node and statement IDs share one monotonic graph counter (IR-05), so the number depends on construction order; the load-bearing assertion is real ir.Print content (`return 1.0 + 2.0`) after the header"

patterns-established:
  - "Instrumentation tests set flags.F fields and restore via t.Cleanup helpers (setDumpAfter/setDumpGraphviz/setPassTiming/setVerifyEach)"
  - "Placeholder absence asserted via the actual placeholder line prefix (// [ir dump: / // [dot dump:) so the placeholder phrasing never appears in pkg/ (acceptance grep hygiene)"

requirements-completed: [TOOL-06]

# Metrics
duration: 4min
completed: 2026-10-15
---

# Phase 01 Plan 05: Instrumentation Wiring Summary

**pkg.DefaultInstrumentations wires IRDump's TextEmitter to ir.Print and DotEmitter to visualization.EmitDot, so --dump-after/--dump-graphviz emit real IR content through Manager.Run — closing TOOL-06 and giving Phase 5's CLI its ready-made instrumentation set**

## Performance

- **Duration:** 4 min
- **Started:** 2026-10-15T17:03:16Z
- **Completed:** 2026-10-15T17:07:04Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- `pkg.DefaultInstrumentations(w io.Writer) []pass.Instrumentation` returns the standard VerifyEach + IRDump + PassTiming set in one constructor; IRDump's previously-nil callback fields are bound to the real 01-03 emitters (`TextEmitter = ir.Print`, `DotEmitter` = EmitDot-into-strings.Builder adapter)
- TOOL-06 closed: text dumps emit the pass header plus real canonical IR (`statement #N:` header + stage text `return 1.0 + 2.0`), graphviz dumps emit real `digraph`/`subgraph cluster_` output — no placeholder path remains reachable through the public API
- End-to-end test through `pass.NewManager(...).Run` proves all three instrumentations behaviorally live: verify gate fails pipeline end on a broken graph, dump fires under `DumpAfter=all`, PassTiming reports a sorted `pass nop: <duration>` line

## Task Commits

Each task was committed atomically:

1. **Task 1: DefaultInstrumentations wiring + real-dump-output test (RED)** - `daa143e` (test)
2. **Task 1: DefaultInstrumentations wiring + real-dump-output test (GREEN)** - `302c4c8` (feat)

**Plan metadata:** pending (docs commit follows this summary)

_Note: TDD task — test commit (RED) precedes implementation commit (GREEN); test refinements from GREEN debugging ride in the feat commit_

## Files Created/Modified
- `pkg/instrument.go` - `DefaultInstrumentations` constructor; wires IRDump.TextEmitter→ir.Print and IRDump.DotEmitter→visualization.EmitDot adapter; composes VerifyEach + IRDump + PassTiming
- `pkg/instrument_test.go` - `TestInstrumentationsRealDumpOutput` (real text/dot content end-to-end, placeholder-absent) and `TestDefaultInstrumentationsComposition` (3-set type composition + behavioral liveness of verify gate, dump firing, timing report); nopPass stub; flags.F t.Cleanup helpers

## Decisions Made
- **DotEmitter error handling:** the adapter returns the partial buffer and never propagates EmitDot's error. EmitDot is emit-only (returns nil for nil graph, only error source is a writer flush failure, impossible for strings.Builder). This satisfies T-1-05-03 (no dump-path panic/abort) and the acceptance grep for `panic(` in pkg/instrument.go.
- **Dynamic statement-ID assertion** instead of the plan's literal `statement #1` — see Deviations.
- **Placeholder absence checked by line prefix** (`// [ir dump:`, `// [dot dump:`) rather than the placeholder phrasing, keeping `grep -rn "emitter wired" pkg/` clean per the acceptance criterion while asserting strictly more (the whole placeholder line shape, not one phrase).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test asserted a literal `statement #1` header that the as-built IR can never produce for a multi-node statement**
- **Found during:** Task 1 (GREEN — first test run failed with `dump missing real ir.Print content (statement #1): "// ---- after nop ----\nstatement #4:\n\n\n"`)
- **Issue:** Two plan assumptions collided with the as-built 01-01 IR: (a) node IDs and statement IDs share one monotonic counter (`Graph.NewID`, IR-05), so a statement whose rhs needs N operand nodes always gets ID > N (got `statement #4`); (b) a non-stage statement root (bare binop) renders only the header — `chainOf` walks stage kinds only, so no body content was printed at all, making even a correct-ID assertion content-free.
- **Fix:** Built the minimal graph as a stage-rooted statement (`return 1.0 + 2.0` — const, const, binop, return) and asserted the statement header dynamically via `g.Block().Statements()[0].ID()`, plus the real rendered stage content `return 1.0 + 2.0`. The must-have truth ("real ir.Print output with statement headers, not a placeholder") is asserted strictly stronger than the plan's literal.
- **Files modified:** pkg/instrument_test.go
- **Verification:** `go test ./pkg/ -count=1` passes; header + content assertions both fire against real ir.Print output
- **Committed in:** 302c4c8 (Task 1 GREEN commit)

**2. [Rule 2 - Missing Critical] Placeholder phrasing leaked into pkg/ via the test's negative assertions, violating the plan's own acceptance grep**
- **Found during:** Task 1 (acceptance-criteria sweep: `grep -rn "emitter wired" pkg/` must return nothing)
- **Issue:** The plan's test behavior requires asserting the placeholder strings are absent, while the acceptance criterion requires those exact strings to appear nowhere under pkg/ — direct contradiction if the test names them literally.
- **Fix:** Assert absence of the actual placeholder line prefixes emitted by `IRDump.AfterPass` (`// [ir dump:` / `// [dot dump:` from internal/pass/instrument.go) — a stricter check than the phrasing alone — and reworded comments so the placeholder text appears nowhere in pkg/.
- **Files modified:** pkg/instrument_test.go
- **Verification:** `grep -rn "emitter wired" pkg/` returns nothing; negative assertions still fail the test if the unwired placeholder path ever regresses
- **Committed in:** 302c4c8 (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing-critical/consistency)
**Impact on plan:** Both fixes are test-shape corrections against as-built (01-01/01-02/01-03) behavior; the shipped API matches the plan's action block exactly. No scope creep.

## Issues Encountered
- None beyond the deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5's CLI consumes `pkg.DefaultInstrumentations(os.Stderr)` and gets --verify-each/--dump-after/--dump-graphviz/--pass-timing behavior gated by flags.F for free.
- Sibling wave-3 plans (pkg builder/chain/expr) touch disjoint files; no integration dependency on this plan beyond shared package `pkg` compiles (verified: `go test ./...` green).

## Threat Flags

None — no new network endpoints, auth paths, file access, or schema changes. Threat-register mitigations T-1-05-01/02/03 all verified by the shipped tests (real-content assertion, caller-provided writer, no panic path).

---
*Phase: 01-ir-core-pass-infrastructure-tooling*
*Completed: 2026-10-15*

## Self-Check: PASSED
- pkg/instrument.go, pkg/instrument_test.go, 01-05-SUMMARY.md all exist on disk
- Commits daa143e (test RED) and 302c4c8 (feat GREEN) verified in `git log`
