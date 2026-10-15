---
phase: 01-ir-core-pass-infrastructure-tooling
plan: 04
subsystem: api
tags: [go, builder-api, typestate, fluent-interface, ssa-ir, pass-manager, golden-testing, determinism]

# Dependency graph
requires:
  - phase: 01-01 (ir-core)
    provides: exported IR constructors (NewGraph/AddParam/NewMap/NewReduce/NewStore/NewReturn), statement IDs at construction (D-04), the verifier and its unwired-operand check
  - phase: 01-02 (pass-manager)
    provides: pass.Manager, VerifyEach instrumentation implementing PipelineEnd (AfterPipeline runs the final verify even for an empty pass list; Manager.Run returns its error)
  - phase: 01-03 (printer-tooling)
    provides: canonical deterministic printer (ir.Print) and the committed doc-pinned golden pkg/testdata/matmul_scale.ir
provides:
  - Public pkg/ graph-builder API: Builder root with typed Dim/Tensor/Operand/Expr handles (D-08)
  - Typestate statement chain StmtIter -> RootMap -> RootAs -> MidChain -> AfterReduce -> AfterMap -> Tail -> Stored where illegal pipe orders are compile-time errors (D-07)
  - Expr-builder DSL mirroring doc/syntax.md §6.2 operators 1:1 with eager type checks (D-09)
  - Eager call-site checks returning errors, never panics (D-10)
  - Build() verification exclusively through pass.NewManager(nil) + VerifyEach (D-10 single authoritative path)
  - Programmatic matmul_scale oracle byte-matching the doc-pinned golden (FE-01 + TOOL-05, phase Success Criterion 1)
  - TestDeterminism / TestDeterminismDot byte-compare gates (VER-08)
affects: [phase-2-parser-lowering, any phase constructing test oracles through pkg]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Typestate chain: distinct Go struct per pipe rung, error-returning rungs return (rung, error); chains unrolled into per-rung statements (Go forbids method calls on multi-valued returns)"
    - "Lazy handle materialization: ScalarParam Exprs materialize their paramRef node on first use so declarations never consume graph IDs ahead of Stmt's statement-ID reservation"
    - "Reserve-then-append statement IDs: ir.Graph.ReserveStatementID + AddStatementAs give the builder dense statement numbering (doc-pinned `statement #1`/`#2`)"
    - "Any-typed lambda parameter + runtime arity dispatch (coerceMapFn): Go assignability forbids pinning fn func(...Expr) Expr while accepting func(l, r Expr) Expr literals"

key-files:
  created:
    - pkg/builder.go
    - pkg/chain.go
    - pkg/expr.go
    - pkg/builder_test.go
    - pkg/oracle_test.go
    - pkg/determinism_test.go
  modified:
    - internal/ir/construct.go
    - internal/ir/print.go
    - internal/visualization/testdata/matmul_scale.dot

key-decisions:
  - "Map/Reduce lambda parameter typed as `fn any` with a type-switch arity dispatch: the plan's pinned variadic signature cannot accept the plan's own pinned oracle lambda literal under Go assignability rules"
  - "Statement chains unrolled into intermediate rung variables: Go forbids chaining a method call off a (rung, error) multi-valued return; all plan acceptance greps still match"
  - "Printer reconciled to the doc-pinned golden (never the reverse): tensor-typed paramRefs render lhs[@m][@n] from type dims; iter-cluster bodies receive the previous stage so piped inputs render stage-local %0"
  - "Stmt() reserves both the statement ID and its return-statement ID before any chain nodes exist, giving dense doc-pinned numbering; ScalarParam handles are lazy so pre-Stmt declarations do not disturb it"
  - "v0 cast registry lives in pkg (arithmetic-to-arithmetic only, no identity casts) until the shared frontend/IR registry lands with the parser"

patterns-established:
  - "Builder-owned eager checks: declaration-time violations are recorded via noteDeclErr and surfaced by the next error-returning rung or Build (pinned no-error signatures cannot return errors)"
  - "Poisoned Expr error propagation: operator type violations produce an Expr carrying the error, surfaced at the next rung or Build — the library never panics"
  - "Build() emits kernel `return` statements for declared output bindings via AddStatementAs with the reserved return IDs"

requirements-completed: [FE-01, TOOL-05, VER-08]

# Metrics
duration: ~90min
completed: 2026-10-16
---

# Phase 01 Plan 04: Public Builder API + matmul_scale Oracle Summary

**Fluent typestate graph-builder in pkg/ (8 compile-time rung types, typed handles, §6.2 expr DSL) whose programmatic matmul_scale oracle byte-matches the doc-pinned golden, with Build() verifying exclusively through the empty-pipeline pass-manager path and -count=10 determinism gates.**

## Performance

- **Duration:** ~90 min (est.; first commit 2026-10-15T17:14:50Z, session included pre-implementation printer/golden diagnosis)
- **Started:** 2026-10-15 (session; UTC)
- **Completed:** 2026-10-15T17:38:00Z
- **Tasks:** 3
- **Files modified:** 9 (6 created, 3 modified)

## Accomplishments
- matmul_scale constructed programmatically through pkg/ byte-matches the committed doc-pinned golden pkg/testdata/matmul_scale.ir — the phase's flagship Success Criterion 1 (FE-01 + TOOL-05)
- Illegal pipe orders are compile-time errors via 8 distinct typestate rung types; every runtime-checkable violation (arity, axis coverage/permutation, homogeneous reducer, subscript count, undeclared names, unknown builtins) returns an error at the call site (D-07, D-10)
- Build() runs the single authoritative verification path — pass.NewManager(nil) + VerifyEach, empty pass list — and surfaces verification failures as Build errors; proven on both the positive (oracle) and negative (unwired operand) paths (D-10)
- VER-08 proven: two fresh in-process builds print byte-identical text and byte-identical .dot; `go test ./pkg/ -run TestDeterminism -count=10` green; committed golden is the cross-process anchor

## Task Commits

Each task was committed atomically (--no-verify per parallel-executor protocol; orchestrator runs hooks after the wave):

1. **Task 1: Builder API — handles, expr DSL, typestate chain, eager checks** - `609b927` (test, RED) + `bb7e1b2` (feat, GREEN)
2. **Task 2: matmul_scale oracle graph + PM-path verification + golden test** - `21d18e1` (test, RED) + `b92b9eb` (feat, GREEN)
3. **Task 3: Determinism byte-compare (VER-08)** - `59cab0b` (test)

**Plan metadata:** pending (this SUMMARY's commit)

_Note: Task 3 tests passed on first run — RED was not achievable because determinism was already established by Task 2's implementation; documented under Deviations._

## Files Created/Modified
- `pkg/builder.go` - Builder root: NewBuilder, Dim/Tensor/ScalarParam/At handles, Stmt with dual ID reservation, Build() with PM-path verification and automatic kernel-return emission
- `pkg/chain.go` - Typestate chain: 8 rung types (StmtIter/RootMap/RootAs/MidChain/AfterReduce/AfterMap/Tail/Stored), eager checks (arity, as-permutation, piped as-exactness, homogeneous reducer, structural subscripts), lambda construction with doc-style parameter names
- `pkg/expr.go` - Expr DSL mirroring syntax.md §6.2: arithmetic/comparison/bool/bitwise operators, literals, Select/Cast/Call package functions, pure-builtin registry, poisoned-Expr error propagation
- `pkg/builder_test.go` - TestBuilderEagerChecks (9 negative cases) + TestBuilderTypestate (legal sequences)
- `pkg/oracle_test.go` - buildMatmulScale shared helper, TestMatmulScaleGolden, TestBuildVerifiesThroughPM (positive + negative paths)
- `pkg/determinism_test.go` - TestDeterminism (print byte-compare) + TestDeterminismDot (dot byte-compare)
- `internal/ir/construct.go` - Graph.ReserveStatementID + Graph.AddStatementAs (reserve-then-append statement IDs for the builder)
- `internal/ir/print.go` - value() renders tensor-typed paramRefs as lhs[@m][@n] from type dims; iter-cluster stageText passes prev to the body so piped inputs render stage-local %0
- `internal/visualization/testdata/matmul_scale.dot` - derived dot golden regenerated after the printer reconciliation (sanctioned by dot_test.go)

## Decisions Made
- **fn-any lambda dispatch** (`fn any` + coerceMapFn type switch): the plan's pinned variadic `fn func(...Expr) Expr` cannot accept the plan's own pinned oracle literal `func(l, r pkg.Expr) pkg.Expr` — Go assignability forbids assigning a non-variadic func to a variadic func type. Arity mismatches return errors containing "arity".
- **Unrolled chains**: Go forbids a method call on a multi-valued return, so `Map(...).As(...)` chains were unrolled into per-rung statements with intermediate variables. Every plan acceptance grep (`.Iter(m, k, n)`, `root.As(m, k, n)`, `.Reduce(`, `tail.Store(out, m, k)`) still matches, and the typestate compile-time guarantees are unchanged (only legal transitions are expressible at each step).
- **Printer reconciled to the golden, never the reverse**: the 01-03 printer could not emit the doc-pinned golden bytes (tensor paramRefs printed bare; piped-through-iter inputs printed as numbered defs instead of stage-local `%0`). The plan's Task 2 behavior explicitly authorizes "fix the printer/builder to emit the pinned golden form — do NOT re-run with -update". Both fixes are in internal/ir/print.go; all existing 01-03 printer tests still pass (their fixtures use scalar paramRefs and first-stage iter clusters).
- **Dense statement numbering via reservation**: Stmt() reserves the statement ID and the return-statement ID before any chain nodes exist; ScalarParam handles are lazy (Expr.materialize) so pre-Stmt declarations never consume graph IDs. This yields the doc-pinned `statement #1` (store chain) and `statement #2` (return).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Pinned variadic lambda signature is unusable with the pinned oracle literal**
- **Found during:** Task 1 (chain.go rung signatures)
- **Issue:** The plan pins `Map(ops []Operand, fn func(...Expr) Expr)`, but the plan's own pinned oracle passes `func(l, r pkg.Expr) pkg.Expr` — Go forbids assigning a non-variadic function to a variadic function type, so the pinned chain cannot compile as written.
- **Fix:** Rung `fn` parameters are typed `any` and dispatched at runtime through `coerceMapFn`, which type-switches over `func(Expr) Expr` … `func(Expr, Expr, Expr, Expr) Expr` and `func(...Expr) Expr`, returning an "arity" error on mismatch (test-greppable). Pinned receiver shapes (`Map(ops []Operand, fn any)`, `Reduce(init Expr, fn func(acc, v Expr) Expr)`) otherwise unchanged.
- **Files modified:** pkg/chain.go, pkg/builder_test.go, pkg/oracle_test.go
- **Verification:** all wrong-arity subtests fail with errors containing "arity"; full suite green
- **Committed in:** bb7e1b2 (Task 1 GREEN)

**2. [Rule 3 - Blocking] Chained multi-value rung calls do not compile in Go**
- **Found during:** Task 1 (test compilation), re-surfaced twice during test rewrites
- **Issue:** `Map(...).As(...)` — a method call on a `(rung, error)` multi-valued return — is a compile error in Go, so the plan's fluent one-liner chains are inexpressible verbatim.
- **Fix:** All chains unrolled into per-rung statements (`root, err := b.Stmt().Iter(...).Map(...)`; `mid := root.As(...)`; `ar, err := mid.Reduce(...)`; …). Typestate ordering guarantees are identical; every plan acceptance grep still matches.
- **Files modified:** pkg/oracle_test.go, pkg/builder_test.go
- **Verification:** acceptance greps; `go build ./pkg/` exit 0
- **Committed in:** bb7e1b2, b92b9eb

**3. [Rule 1 - Bug] Eager ScalarParam broke the doc-pinned statement numbering**
- **Found during:** Task 2 (golden mismatch: `statement #2`/`#1` and return numbered too late)
- **Issue:** `ScalarParam` eagerly created its paramRef node, consuming graph ID 1 before Stmt's reservation, and the kernel return's statement ID was allocated after all chain nodes — so the print showed wrong statement numbers vs the doc-pinned golden.
- **Fix:** (a) ScalarParam returns a lazy Expr whose paramRef materializes on first use (`Expr.materialize`; every node consumer updated); (b) `Stmt()` reserves TWO IDs up front (statement + its return) via new `ir.Graph.ReserveStatementID`; (c) Build emits kernel returns through `AddStatementAs(id, …)` with the reserved IDs.
- **Files modified:** pkg/builder.go, pkg/expr.go, pkg/chain.go, internal/ir/construct.go
- **Verification:** statement IDs are exactly #1 (store chain) and #2 (return); `ir.Print` byte-identical to the golden
- **Committed in:** bb7e1b2 (ir API), b92b9eb (lazy materialization + reservation)

**4. [Rule 1 - Bug] Printer could not emit the doc-pinned golden bytes**
- **Found during:** Task 2 (golden diff diagnosis, before coding)
- **Issue:** Two rendering gaps vs doc/tca.md §5.1: tensor-typed paramRefs printed bare (`lhs`) instead of structurally (`lhs[@m][@n]`), and a value piped through an `iter()` wrapper printed as a numbered SSA def (`%1`) instead of the stage-local `%0` (doc §2.6 pipe semantics).
- **Fix:** (1) `value()` renders tensor-typed paramRefs from the tensor type's dims (`lhs[@m][@n]`); scalar/lambda params stay bare (`scale`, `v`). (2) `iterCluster` stageText passes `prev` into the body so a piped body input renders `%0`. The derived dot golden was regenerated (explicitly sanctioned by dot_test.go's comment for 01-04 reconciliation).
- **Files modified:** internal/ir/print.go, internal/visualization/testdata/matmul_scale.dot
- **Verification:** TestMatmulScaleGolden byte-matches with pkg/testdata/matmul_scale.ir UNCHANGED (`git diff --stat a5704e4 HEAD -- pkg/testdata/matmul_scale.ir` empty); all 01-03 printer/visualization tests still pass
- **Committed in:** b92b9eb (Task 2 GREEN)

### Process Deviations

**5. Task 3 RED not achievable (documented, no code impact)**
- **Found during:** Task 3
- **Issue:** TDD RED requires the new test to fail first, but determinism was already established by Task 2's implementation (the byte-matched golden is itself proof of cross-build stability).
- **Fix:** Tests committed in a single passing commit after verification with `-count=10`. No production code changed in Task 3, matching the plan's "No production code changes expected".
- **Committed in:** 59cab0b

---

**Total deviations:** 4 auto-fixed (2 blocking, 2 bug) + 1 documented process deviation
**Impact on plan:** All auto-fixes were required to satisfy the plan's own pinned artifacts (oracle chain literal, doc-pinned golden bytes) under Go language rules. No scope creep; the golden was never regenerated; the typestate and D-10 contracts are fully preserved.

## Issues Encountered
- Two Go language limitations (variadic assignability, multi-value chaining) contradicted the plan's pinned code shapes; resolved via the deviations above without weakening any acceptance criterion.
- A mid-task test bug (`store: @n is not a live axis` in the PM-path subtest) — the output tensor was declared rank 2 but only `@m` is live after `As(m, n)` + Reduce; corrected the test fixture to a rank-1 output.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- pkg/ is the oracle-construction and parser-lowering target for Phase 2; buildMatmulScale in pkg/oracle_test.go is the reference construction sequence
- The v0 cast registry and pure-builtin registry currently live in pkg/expr.go; the parser phase should hoist them into a shared frontend/IR registry
- The derived dot golden now reflects the reconciled printer; any future printer change must regenerate it (it is derived, -update is permitted)

## Self-Check: PASSED
- All 6 created files exist on disk and are committed (609b927, bb7e1b2, 21d18e1, b92b9eb, 59cab0b)
- All acceptance greps verified: builder signatures, 8 rung types, `func Select/Cast/Call`, expr count >= 15, no `panic(` in pkg/, `ir.Verify` zero matches in pkg/builder.go, NewManager+VerifyEach in Build, chain greps in oracle_test.go, TestDeterminism/TestDeterminismDot present
- Golden unchanged vs base commit a5704e4; `go test ./... -count=1`, `go vet ./...`, `go list -m all | wc -l` == 1 all green

---
*Phase: 01-ir-core-pass-infrastructure-tooling*
*Completed: 2026-10-16*
