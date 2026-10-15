---
phase: 01-ir-core-pass-infrastructure-tooling
plan: 03
subsystem: compiler-tooling
tags: [go, diagnostics, slog, golden-test, canonical-printer, graphviz-dot, determinism]

# Dependency graph
requires:
  - phase: 01-01
    provides: "internal/ir construction surface, per-graph monotonic IDs, deterministic BB/operand walks, Kind/opcode String() tables, Verify"
provides:
  - "internal/diagnostic: 3-severity model (Error/Warning/Remark), Span, structured Diagnostic, Bag with Err() (TOOL-01)"
  - "internal/log: slog facade with Debug/Info/Warn/Error + context-scoped WithGroup, stderr default (TOOL-02)"
  - "internal/ir/print.go: canonical deterministic printer with Print/PrintOpts/PrintOptions/FormatConst plus shared emission helpers ComputeNumbering/Numbering.Num/StmtNodes/OperandsOf/LambdaBodies/Label (TOOL-04, D-01..D-04)"
  - "internal/golden: one generic -update golden harness with exact byte compare and first-differing-line report (TOOL-05, D-11..D-14)"
  - "internal/visualization: EmitDot cluster-per-statement DOT emission mirroring text numbering (TOOL-03, D-05)"
  - "pkg/testdata/matmul_scale.ir: doc/tca.md §5.1 oracle golden, byte-exact, guarded by TestOracleGoldenBytes until plan 01-04's dynamic comparison"
  - "ir.NewLambdaParam: exported lambda-param constructor completing the external construction surface"
affects: [01-04-pkg-builder, 01-05, phase-02-codegen-runtime, phase-05-frontend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Canonical printer: chain stages joined by ' |', stage-local %0 pipe inputs decided by chain membership, %name stored bindings, %N numbering recomputed per print"
    - "Generic golden harness: single package-level -update flag, normalize to one trailing newline, exact byte compare, first-diff locator with 3 lines context"
    - "DOT emission reuses the printer's walk/render/numbering via exported ir helpers so text and graph can never drift (D-05)"
    - "Lambda.Body edges emitted into DOT as def-to-user edges alongside operand Use edges"

key-files:
  created:
    - internal/diagnostic/diagnostic.go
    - internal/diagnostic/diagnostic_test.go
    - internal/log/log.go
    - internal/golden/golden.go
    - internal/ir/print.go
    - internal/ir/print_test.go
    - internal/ir/testdata/golden_smoke.txt
    - pkg/testdata/matmul_scale.ir
    - internal/visualization/dot.go
    - internal/visualization/dot_test.go
    - internal/visualization/testdata/matmul_scale.dot
  modified:
    - internal/ir/node.go
    - internal/ir/construct.go

key-decisions:
  - "Chain membership decides %0: a stage's pipe slot renders %0 only when its def IS the previous chain stage (defensive — a miswired slot renders its value instead of lying)"
  - "Statement boundaries: a chain walk stops at any node that is another statement's RHS; Statement carries an unexported owning-graph back-pointer (set by AddStatement) to make that check possible"
  - "FormatConst owns the float rule per plan: strconv.FormatFloat 'g' -1 32, append .0 when no radix/exponent, NO f suffix — IR text is algebra notation (overrides the RESEARCH sketch's f-suffix note; the §5.1 oracle prints 0.0)"
  - "golden.Check (error-returning) is exported alongside Assert so the harness can self-test its mismatch report without intercepting t.Fatal"
  - "DOT labels render stage text with the same numbering as the text printer; the pipe input edge is emitted explicitly even though the label shows the stage-local %0"
  - "Lambda params get an exported constructor (ir.NewLambdaParam) — the 01-01 construction surface could build Lambda payloads only from inside package ir"

patterns-established:
  - "Diagnostics format: <file>:<line>:<col>: <severity>: <message> + 2-space-indented context lines (zero Span omits the location)"
  - "Golden fixture hygiene: no trailing whitespace, LF only, exactly one trailing newline — enforced by TestOracleGoldenBytes"
  - "Emission determinism rule: BB statement order + chain stage order + operand slot order + node ID only; maps are lookup-only, never iterated (Pitfall 1)"

requirements-completed: [IR-05, TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05]

# Metrics
duration: 18min
completed: 2026-10-15
---

# Phase 1 Plan 03: Printer + Tooling Backbone Summary

**Three-severity diagnostics with structured context, an slog facade, the canonical deterministic IR printer with byte-exact §5.1 oracle golden, one generic -update golden harness, and cluster-per-statement DOT emission that reuses the printer's numbering.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-10-15T16:38:56Z
- **Completed:** 2026-10-15T16:56:44Z
- **Tasks:** 3
- **Files modified:** 11 created, 2 modified, 4 skeletons deleted

## Accomplishments

- **Diagnostic model (TOOL-01):** `Error`/`Warning`/`Remark` severities in the REQUIREMENTS order, `Span` (zero = unknown location), `Diagnostic` with ordered structured context lines (`expected:`/`actual:` style per syntax.md §10), and an insertion-ordered `Bag` whose `Err()` is non-nil only on Error severity. `TestBagErrNonNilOnlyWithErrorSeverity` and `TestStringRendersSpanMessageAndContext` pin the behavior.
- **Log facade (TOOL-02):** `Debug`/`Info`/`Warn`/`Error` over a swappable package-default `slog.Logger` (stderr text handler; stdout never touched, Pitfall 6) plus context-scoped `WithGroup` for per-pass/per-node grouped records. Pass code never imports slog directly.
- **Canonical printer (TOOL-04, IR-05, D-01..D-04):** `statement #N:` headers with construction-time IDs, chain stages joined by ` |` at end of line with 2-space continuation indent, stage-local `%0` pipe inputs (doc §2.6 — chain-local, which is why both the reduce and the scale-map inputs print `%0`), bare kernel params (`scale`, `lhs`), `%name` stored bindings (`%output`), `%N` definition-order numbering recomputed per print (Pitfall 3), lambda bodies re-rendered bottom-up with parens decided only by the syntax.md §7.6 13-level table, and `FormatConst` as the single owner of the float spelling (`0.0`, no `f` suffix — Pitfall 9). `TestCanonicalPrint` proves two structurally identical graphs print byte-identically and that repeated prints are stable.
- **Golden harness (TOOL-05, D-11..D-14):** one generic `internal/golden` package — exactly one package-level `-update` flag (Pitfall 7; acceptance-verified as the sole `"update"` registration in the repo), trailing-newline normalization, exact byte compare (D-13), and a hand-rolled first-differing-line report with 3 lines of context. `TestGoldenHarness` exercises match, mismatch, and report content.
- **Oracle golden:** `pkg/testdata/matmul_scale.ir` committed byte-exactly to the plan's `<canonical_golden>` block (verified: no trailing spaces, LF only, single trailing newline) and guarded against silent drift by `TestOracleGoldenBytes` until plan 01-04 replaces the static pin with the dynamic printer comparison.
- **DOT emission (TOOL-03, D-05):** `EmitDot(w io.Writer, g *ir.Graph) error` — `digraph <name> {`, one `subgraph cluster_<stmtID>` per statement (verified Graphviz convention), `n<id> [label=<canonical>]` nodes using the text printer's own rendering and numbering (shared via exported `ir.StmtNodes`/`ir.Label`/`ir.ComputeNumbering`), and explicit def-to-user edges in operand order plus lambda-body edges. Text-only, no graphviz library or binary. `TestDotGolden` pins `testdata/matmul_scale.dot` and proves byte-identical repeat emission.

## Task Commits

Each task was committed atomically:

1. **Task 1: Diagnostic severity model + slog log facade** - `87b9508` (feat)
   - *follow-up:* `06064e2` (chore: drop superseded doc.go skeletons)
2. **Task 2 RED: failing printer + golden harness tests** - `0488ddd` (test)
   **Task 2 GREEN: canonical printer + golden harness + oracle golden** - `a8dd6a1` (feat)
3. **Task 3 RED: failing dot emission test** - `419ece7` (test)
   **Task 3 GREEN: graphviz dot emission** - `12d7ddd` (feat)

_Note: TDD tasks have two commits each (test → feat) per the tdd execution flow._

## Self-Check: PASSED

- 13/13 claimed files verified on disk (11 created + node.go/construct.go modified + this SUMMARY)
- All 6 task commits present on HEAD: 87b9508, 06064e2, 0488ddd, a8dd6a1, 419ece7, 12d7ddd
- Wave merge gate green: `go test ./... -count=1`, `go vet ./...`, `go list -m all | wc -l` == 1

## Files Created/Modified

- `internal/diagnostic/diagnostic.go` - Severity enum, Span, Diagnostic, Bag (TOOL-01)
- `internal/diagnostic/diagnostic_test.go` - severity names, Bag.Err semantics, insertion-order copy, String rendering
- `internal/log/log.go` - slog facade: SetDefault/Debug/Info/Warn/Error/WithGroup (TOOL-02)
- `internal/golden/golden.go` - generic golden harness: Assert/Check, -update, firstDiff (TOOL-05)
- `internal/ir/print.go` - Print/PrintOpts/PrintOptions/FormatConst + ComputeNumbering/Numbering/StmtNodes/OperandsOf/LambdaBodies/Label (TOOL-04)
- `internal/ir/print_test.go` - TestCanonicalPrint, TestFormatConst, TestGoldenHarness, TestOracleGoldenBytes
- `internal/ir/testdata/golden_smoke.txt` - golden-harness smoke fixture
- `pkg/testdata/matmul_scale.ir` - doc/tca.md §5.1 oracle golden (byte-exact)
- `internal/visualization/dot.go` - EmitDot cluster-per-statement DOT emission (TOOL-03)
- `internal/visualization/dot_test.go` - TestDotGolden + determinism double-emission check
- `internal/visualization/testdata/matmul_scale.dot` - DOT golden (pinned via -update, CI-verified)
- `internal/ir/node.go` - Statement gains unexported owning-graph back-pointer
- `internal/ir/construct.go` - AddStatement sets the back-pointer; NewLambdaParam exported

## Decisions Made

- **`%0` by chain membership:** a stage's pipe slot renders `%0` only when its def is exactly the previous chain stage; anything else renders as a value. This keeps the golden's stage-local pipe semantics while staying honest about miswired slots.
- **Statement boundary via RHS set:** `chainOf` stops at any stage node that is another statement's RHS (e.g. `return` of a stored binding stays its own statement). This required an unexported `graph` field on `Statement` — additive, no API change.
- **FormatConst without `f` suffix:** the plan explicitly overrides the RESEARCH sketch's "`f` suffix" note — IR text is algebra notation, and the §5.1 oracle prints `0.0`. The rule (`FormatFloat 'g' -1 32`, append `.0` when no `.`/`e`) is owned by this one function and tripwired by goldens.
- **Exported `golden.Check`:** `Assert` is the t.Fatal wrapper; `Check` returns the mismatch error so the harness's own report format is testable.
- **DOT lambda-body edges:** `Lambda.Body` is the IR's single non-Use graph edge (the verifier walks it as one); DOT emits it as a def-to-user edge so the dump is a faithful dependency graph, not just the operand spine.

## §5.1 vs §9.3 Reconciliation Note

Per the plan's `<canonical_golden>` reconciliation block: `pkg/testdata/matmul_scale.ir` pins **doc/tca.md §5.1** as the canonical print format (D-01), with the doc's alignment padding removed. **§9.3 differs because it is POST-canonicalization notation** — factored `iter(@m/M) * iter(@k/K) * iter(@n/N)` form, `reduce<@n>` annotation, `%lhs`/`%scale` name sigils, and `0.0f` source-style suffixes. D-01 pins §5.1 and D-06 makes Phase 1 print-only, so §5.1 form wins; §9.3's alignment is a Phase 5 parser-target concern. Plan 01-04's `TestMatmulScaleGolden` holds the printer to this reconciliation byte-for-byte.

## Known Stubs / Forward Reconciliation (for plan 01-04)

These are intentional Phase-1 boundaries, each with a named owner; none blocks this plan's success criteria:

1. **Map tensor-access operands** render as bare param names (`lhs`, not `lhs[@m][@n]`): the v0 Kind suite has no access node kind — `Access` exists only as the store-target payload (plan 01-01). The dot test documents this and uses a paramRef operand. Plan 01-04's builder + the printer/builder byte reconciliation ("the diff is the spec") owns the final encoding; the printer's `value()` and dot paths are the only places that need to learn it.
2. **`ir.NewLambdaParam` added** (Rule 2 — the exported surface had no way to construct `Lambda.Params` from outside package ir; plan 01-04's builder needs it too).
3. **iterNode init/step are not rendered** in the minimal default (Phase 1 graphs use bare standard induction variables via cluster dims); the bounded form `@x/M` renders when a bound param is set. Verbose mode and Phase 2+ kinds can extend `stageText`/`value` without touching the emission-order machinery.
4. **`value()` fallback `%<id>`** marks nodes unreachable from the block — deterministic, unreachable for verified graphs (the verifier owns rejection of those).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added `ir.NewLambdaParam`**
- **Found during:** Task 3 (RED dot test construction)
- **Issue:** the exported construction surface (plan 01-01 `<interfaces>`) could build `Lambda{Params: ...}` payloads only from inside package ir — no exported constructor for lambda-parameter symbols; plan 01-04's builder hits the same wall.
- **Fix:** added `func NewLambdaParam(name string, t *Type) Param` to construct.go; not registered in the graph's kernel param list, documented as lambda-only.
- **Files modified:** internal/ir/construct.go
- **Committed in:** `12d7ddd` (Task 3 GREEN)

**2. [Rule 1 - Bug] Iteration-cluster body printed as a duplicate chain stage**
- **Found during:** Task 2 GREEN (TestCanonicalPrint first run)
- **Issue:** `chainOf` collected both an iteration cluster and its body map as separate stages, printing `map(...) |\n  iter(@m, @k) -> map(...)` — the body must render inline inside its cluster (one printed stage, matching the golden's `iter(@m, @k, @n) -> map(...) |`).
- **Fix:** `chainOf` now absorbs a stage whose consumer (the already-collected head) is an iterCluster with that stage as its body, continuing the walk from the body's own pipe input.
- **Files modified:** internal/ir/print.go
- **Committed in:** `a8dd6a1` (Task 2 GREEN)

### Execution Adaptations (no Rule 1-4 triggers)

1. **`-update` scope accident:** an early `-update` run rewrote `internal/ir/testdata/golden_smoke.txt` with the mismatch-fixture bytes; the file was restored via `git checkout` and the dot golden regenerated with package-scoped `-update` only. Final tree verified clean.
2. **Verify command paths:** automated commands ran in this worktree (`agent-a2c3b3944b23a3321`), not the plan's literal main-checkout path, per worktree isolation. Identical commands and gates.
3. **Pre-existing `gofmt` drift** in `internal/ir/node.go` (from plan 01-01) left untouched per the scope boundary; all lines changed by this plan are gofmt-clean.

**Total deviations:** 2 auto-fixed (1 missing-critical, 1 bug); 3 execution adaptations
**Impact on plan:** Both auto-fixes are required for the exported API to function and for the golden's stage notation; no scope creep. The 01-03 `<interfaces>` contract (`func Print(*Graph) string`, `func EmitDot(io.Writer, *ir.Graph) error`, `func Assert(t, path, got)`) is implemented exactly as pinned for plan 01-02's `pass.IRDump` wiring.

## Issues Encountered

- None beyond the two auto-fixes above. TDD flow held: both RED commits failed for the expected reason (undefined implementation), both GREEN commits passed all behavior tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Plan 01-04** can: build lambda params via `ir.NewLambdaParam`; print via `ir.Print` and compare via `golden.Assert` against the committed `pkg/testdata/matmul_scale.ir`; rely on `TestOracleGoldenBytes` to catch any golden drift. Open reconciliation item (theirs by plan): tensor-access operand encoding in map position — printer touchpoints are `value()` in print.go only.
- **Plan 01-02** (same wave) can wire `pass.IRDump` callbacks to `ir.Print` (TextEmitter: `func(*ir.Graph) string`) and `visualization.EmitDot` (DotEmitter shape) as one-liners.
- No blockers. Wave merge gate green: full suite, vet, stdlib-only module count all verified.

---
*Phase: 01-ir-core-pass-infrastructure-tooling*
*Completed: 2026-10-15*
