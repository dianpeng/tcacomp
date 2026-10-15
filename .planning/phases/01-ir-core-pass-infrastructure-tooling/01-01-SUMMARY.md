---
phase: 01-ir-core-pass-infrastructure-tooling
plan: 01
subsystem: compiler-ir
tags: [go, ssa-ir, intrusive-use-lists, rauw, interning, fnv, tdd, verifier]

# Dependency graph
requires: []
provides:
  - "tcacomp Go module (stdlib-only, `go list -m all | wc -l` == 1) with 9 package skeletons"
  - "internal/ir: versioned Kind enum + Families, interned Types, Graph/Block/Statement, intrusive Use records, exported construction surface (NewGraph + per-kind constructors)"
  - "internal/ir: MutContext choke-point mutations (SetDef/AddUse/RemoveUse/RAUW/Erase) with structural LastTouchedBy stamping"
  - "internal/ir: Verify(g) chain-consistency checkers (use-def + def-use) with Violation diagnostics naming node ID + last-touched-by + invariant"
affects: [01-02-pass-manager, 01-03-printer-tooling, 01-04-pkg-builder, phase-02-codegen-runtime, phase-03-parallelization, phase-04-optimizations]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mutable pointer SSA with intrusive use-lists (sanctioned exception to the global immutability rule per CLAUDE.md decision b)"
    - "Choke-point mutation: all chain edits via MutContext, construction exempt but PassOrigin-stamped"
    - "Deterministic walks: BB statement order + operand slot order only, no map iteration in emission/verification"
    - "FNV-1a content-hash intern table with full-content bucket confirmation (documented non-crypto)"

key-files:
  created:
    - go.mod
    - internal/ir/kind.go
    - internal/ir/types.go
    - internal/ir/node.go
    - internal/ir/use.go
    - internal/ir/construct.go
    - internal/ir/mutate.go
    - internal/ir/verify.go
    - internal/ir/construct_test.go
    - internal/ir/irx_test.go
    - internal/ir/use_test.go
    - internal/ir/verify_test.go
    - internal/{pass,analysis,flags,diagnostic,log,visualization,golden}/doc.go
    - pkg/doc.go
  modified: []

key-decisions:
  - "Every node-to-node edge is a Use slot in the node's embedded operandList; Lambda.Body is the only non-Use edge, reached via the unexported lambdaBearer interface, so the verifier has exactly two edge kinds to walk"
  - "Use-lists prepend at the head (LLVM discipline); UsesOf returns a freshly rebuilt snapshot in link order, safe for splice-during-iteration"
  - "NewNode switches on kind for fixed-arity concrete structs and falls back to a generic rawNode for KindInvalid / kinds >= KindEnd, keeping IR-03 rejection tests constructible from any package"
  - "Base carries an unexported owning-graph back-pointer so Erase can unlink its statement without changing the pinned five-mutation signature"
  - "Verify walks iteratively with a visited set (malformed/cyclic graphs cannot overflow the stack) and cross-checks use-list membership both ways, so corrupting one chain direction is caught by the opposite-direction checker"

patterns-established:
  - "Operand reference = *Use; value-producing node = the def; *Base = shared SSAVar cell (LLVM mapping, RESEARCH Pattern 1)"
  - "Per-graph monotonic IDs from Graph.NewID only; Erase tombstones, never reuses"
  - "Diagnostics format: verify: node <id> (last touched by '<pass>'): invariant '<name>': <msg>"
  - "White-box tests live in package ir (RESEARCH A3); the exported-surface contract test lives in package ir_test"

requirements-completed: [IR-01, IR-02, IR-03, IR-04, IR-05, IR-06, PASS-06]

# Metrics
duration: 12min
completed: 2026-10-15
---

# Phase 1 Plan 01: IR Core + Wave 0 Module Foundation Summary

**Stdlib-only `tcacomp` module with a mutable SSA IR core: versioned Kind enum, interned pointer-compared types, intrusive use-lists, the exported graph-construction surface, choke-point mutations stamping last-touched-by structurally, and a two-direction chain-consistency verifier with rich diagnostics.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-15T16:12:44Z
- **Completed:** 2026-10-15T16:24:54Z
- **Tasks:** 3
- **Files modified:** 21 created, 0 modified

## Accomplishments

- Wave 0 module foundation: `go.mod` (module tcacomp, go 1.26.2, zero requires) plus 9 package skeletons; `go list -m all | wc -l` == 1 mechanically proves the stdlib-only supply chain (T-1-01-03).
- IR foundations (IR-01/02/03/05/06, D-15/18): `Kind` enum versioned at 1 with append-before-`KindEnd` forward compatibility, Family query with Phase 3-4 family slots, `BinOpCode`/`UnaryOpCode` tables mirroring syntax.md §7.6, FNV-1a-keyed interned types, Graph with exactly one Block and per-graph monotonic IDs, and the full exported construction surface (`NewGraph`, param appenders, `AddStatement`, `NewNode` for any kind including `>= KindEnd`, and 14 typed per-kind wrappers).
- Mutation discipline (IR-04, D-16/17) and verification substrate (PASS-06): the five `MutContext` choke points with structural `LastTouchedBy` stamping, RAUW with next-snapshot splicing and self-RAUW no-op, Erase with live-use refusal + ID tombstoning, and `Verify` running both chain directions in deterministic order — a deliberately one-direction-corrupted chain is caught by the opposite-direction checker, naming node ID, `test-corrupt`, and the invariant.

## Task Commits

Each task was committed atomically:

1. **Task 1: Wave 0 — go.mod and package skeletons** - `fb4d617` (chore)
2. **Task 2: IR foundations — RED tests** - `249465b` (test)
3. **Task 2: IR foundations — implementation** - `555df59` (feat)
4. **Task 3: Choke-point mutation + verifier — RED tests** - `d26858d` (test)
5. **Task 3: Choke-point mutation + verifier — implementation** - `a006864` (feat)

**Plan metadata:** `a1c18c3` (docs: complete plan)

_Note: TDD tasks have two commits each (test → feat) per the tdd execution flow._

## Self-Check: PASSED

- 22/22 claimed files exist on disk (21 source/plan files + this SUMMARY)
- All 5 task commits present on HEAD: fb4d617, 249465b, 555df59, d26858d, a006864

## Files Created/Modified

- `go.mod` - module tcacomp, go 1.26.2, zero requires
- `internal/ir/kind.go` - versioned Kind enum (KindVersion 1), Family query, BinOp/UnaryOp opcode tables with String()
- `internal/ir/types.go` - ElemType, DimRef, interned Type with FNV-1a-keyed intern table (documented non-crypto, T-1-01-04)
- `internal/ir/node.go` - Span/Loc (span zero in Phase 1), Base SSAVar cell (no public mutators, D-15), Param/Statement/Block/Graph, Node + ScalarExprNode families, Value/SSADef aliases
- `internal/ir/use.go` - intrusive Use records, UsesOf snapshot, DefOf, shared link/unlink primitives
- `internal/ir/construct.go` - exported construction surface: NewGraph, Add{Dim,Tensor,Scalar}Param, AddStatement, NewNode (any kind), typed wrappers, Lambda/Access payloads; deliberately non-validating (IR-03/PASS-06 own acceptance)
- `internal/ir/mutate.go` - MutContext + SetDef/AddUse/RemoveUse/RAUW/Erase, all stamping LastTouchedBy = ctx.Pass
- `internal/ir/verify.go` - Violation + Verify: use-def checker (nil def / erased def / unknown kind) and def-use checker (use-list exactly matches pointing uses), deterministic BB-then-operand order
- `internal/ir/construct_test.go` - TestSingleBlock, TestNodeIdentity, TestUseListBasics (white-box)
- `internal/ir/irx_test.go` - TestConstructSurface (external `ir_test` package contract test)
- `internal/ir/use_test.go` - TestUseChain (RAUW + Erase + tombstone re-verify), TestRauwGuards (self-RAUW no-op, erase-with-uses refusal, nil rejection)
- `internal/ir/verify_test.go` - TestVerifierCatchesCorruption, TestVerifierFlagsUnwiredOperand
- `internal/{pass,analysis,flags,diagnostic,log,visualization,golden}/doc.go`, `pkg/doc.go` - Wave 0 package skeletons

## Decisions Made

- **Uniform operand storage**: every concrete node embeds `operandList` (`ops []Use`), so all node-to-node edges are Use slots; `Lambda.Body` is the single non-Use edge, exposed via the unexported `lambdaBearer` interface for the verifier. This keeps mutation choke points and both verifier directions uniform.
- **Prepend use-list discipline** (LLVM habit): new uses splice at the list head; `UsesOf` rebuilds a snapshot per call so RAUW can splice while iterating.
- **rawNode fallback in NewNode**: kinds without a dedicated struct (KindInvalid, `>= KindEnd`, future misuses) construct as a generic node with exactly the given operands, keeping out-of-window graphs buildable from any package for rejection tests.
- **Owning-graph back-pointer on Base** (unexported): lets `Erase` drop its statement from the block without a graph parameter, keeping the pinned five-mutation signatures intact.
- **Iterative verifier walk with visited set**: malformed (even cyclic) graphs are checked without recursion; membership is cross-checked both ways so single-direction corruption is always caught by the opposite-direction checker.

## Deviations from Plan

None requiring deviation rules — all work was planned. Execution adaptations (no Rule 1-4 triggers):

1. **Test file placement**: the plan's Task 2 file list named no test files; its four TDD behavior tests were placed in `internal/ir/construct_test.go` (white-box, package ir) and `internal/ir/irx_test.go` (external contract, package ir_test). Task 3's tests went to the plan-named `use_test.go`/`verify_test.go`. Additive only; the plan frontmatter `files_modified` list was not exhaustive about test files.
2. **Test accessor shape**: the pinned `Node` interface is exactly `{BasePtr(), KindOf()}`, so tests reach IDs via `n.BasePtr().ID()` (and `stmt.ID()` on *Statement). No interface widening.
3. **Verifier checker ordering**: `Verify` performs use-def checks during the single graph walk and def-use checks immediately after, both in deterministic BB-statement-then-operand order — equivalent to the plan's "BOTH checkers in deterministic order" with one walk instead of two.
4. **Verification command paths**: automated commands were run in this worktree (`agent-acb9c9884f9e44ecc`), not the plan's literal main-checkout path, per worktree isolation requirements. Identical commands and gates (`go test`, `go vet`, `go list -m all | wc -l`).

**Total deviations:** 0 auto-fixed; 4 execution adaptations (all cosmetic/organizational, contract-preserving)
**Impact on plan:** The `<interfaces>` contract is implemented exactly as pinned; waves 2-3 code compiles against it unchanged.

## Issues Encountered

None. TDD flow held: both RED commits failed for the expected reason (undefined implementation), both GREEN commits passed all behavior tests on first correct implementation; only compile-level test fixes were needed during RED (unused variables, `BasePtr()` accessor shape).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-02 (pass manager) can code against `internal/ir` immediately: `Graph`, `Node`, `Kind`, `Verify(g) []Violation`, and the construction surface are stable and tested.
- Plan 01-03 (printer/dot/golden) can rely on: per-graph monotonic IDs, deterministic BB statement order + operand order, `Kind.String()`, opcode `String()` tables, `ElemType.String()`, and `Loc.PassOrigin` (zero `Span` per D-03).
- Plan 01-04 (pkg builder) constructs exclusively through the exported surface; `TestVerifierFlagsUnwiredOperand` proves unwired graphs verify-fail, which `TestBuildVerifiesThroughPM` will rely on.
- No blockers. Known intentional stub: `Statement.def` stays nil until a pass wires it (the pinned `AddStatement(rhs)` has no def parameter); `Span` is zero-valued by design until the frontend arrives (IR-06).

---
*Phase: 01-ir-core-pass-infrastructure-tooling*
*Completed: 2026-10-15*
