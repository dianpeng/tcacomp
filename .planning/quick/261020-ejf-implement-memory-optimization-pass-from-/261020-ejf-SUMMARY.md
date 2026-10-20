---
phase: quick-261020-ejf
plan: 01
subsystem: opt-memopt
tags: [memopt, memory-optimization, store-load-forwarding, store-store-elimination, dead-store-elimination, ir-choke-point, effect-chains]
requires:
  - gvn (pass runs at 112, Requires ["gvn"])
  - MutContext choke-point surface (internal/ir/mutate.go)
provides:
  - "memopt pass (priority 112): C1-guarded store-to-load forwarding, store-store elimination, dead-store sweep over GMEM effect chains"
  - "memopt-verify pass (priority 113): independent re-derivation + fixpoint/integrity assertions"
  - "ir.MutContext.RemoveImpactBy choke point (C3): drop an ImpactBy effect edge by record identity"
  - "10-family property-test suite pinning INV-A..G"
affects:
  - internal/opt/all (new blank import — BuildPipeline now contains memopt/memopt-verify)
  - internal/opt/registry.go (priority-scale comment)
tech-stack:
  added: []   # stdlib-only Go; no new dependencies
  patterns:
    - "chain-totality eligibility guard (C1) — skip, never error"
    - "RAUW-then-repair splice: capture successor ImpactBy records before ReplaceAllUsesWith, re-point via SetDef / RemoveImpactBy after"
    - "shared deriveRegions helper re-run by the verifier (gvn's shared-code pattern)"
key-files:
  created:
    - internal/opt/memopt/memopt.go
    - internal/opt/memopt/verify.go
    - internal/opt/memopt/memopt_test.go
    - internal/ir/mutate_test.go
    - doc/optana/memory-opt.md
  modified:
    - internal/ir/mutate.go
    - internal/ir/effect.go
    - internal/ir/node_memory.go
    - internal/ir/node_end.go
    - internal/opt/all/all.go
    - internal/opt/registry.go
decisions:
  - "C2a implemented as zero-live-use + RAUW-to-value + fixpoint sweep (the corrected rule), not the doc's prose 'no MemLoad after'"
  - "Successor-edge identification by SSAUse record identity (handle ==) — (user, def) pair matching would wrongly splice the Return's data-use record"
  - "Walk-past comparison anchored at the ORIGIN store, never at the moving cursor"
  - "Region keying by tensor operand BasePtr (C2c); End edges into a region count toward the ≤1-successor eligibility test"
metrics:
  duration: ~15 min
  completed: 2026-10-20
  tasks: 3
  commits: 6
---

# Phase quick-261020-ejf Plan 01: Implement memory-optimization pass Summary

Implemented the memory-optimization pass from doc/optana/memory-opt.md exactly per the locked algorithm review: the `memopt` pass (store-to-load forwarding, store-store elimination, dead-store elimination over per-tensor GMEM effect chains), the sibling `memopt-verify` fixpoint verifier, the new `MutContext.RemoveImpactBy` IR choke point (C3), and the review's 10 property-test families — with all four binding conditions (C1-C4) and post-conditions INV-A..G implemented and pinned.

## What Was Built

### Task 1 — MutContext.RemoveImpactBy choke point (C3), TDD (9bd284e RED, 5aade71 GREEN)
- `internal/ir/effect.go`: `effectBearer` interface extended with `removeImpactUse(u SSAUse) bool`; shared `dropImpactHandle` helper (record-identity `slices.Delete` of a handle copy, mirroring `operandList.removeSlot` per CR-01/CR-02).
- `internal/ir/node_memory.go` + `node_end.go`: hook implemented on `memLoadNode`, `memStoreNode`, `returnNode` — removal only; constructors and `effectIns` layout untouched.
- `internal/ir/mutate.go`: exported `RemoveImpactBy` — nil record → `"ir: RemoveImpactBy on nil use"`; a record not held in the user's `effectIns` (ordinary operand slot, non-bearer user) → fail-closed error naming node ID + kind; success detaches the record from the def's ref list, kills it, drops the handle, stamps user and def node. `RemoveUse`/`Erase`/`SetDef` semantics untouched.
- `internal/ir/mutate_test.go`: four tests pin happy-path splice, nil use, wrong slot class (BinOp input AND a bearer's value operand), and the Return edge — whole `internal/ir` suite green.

### Task 2 — memopt (112) + memopt-verify (113) + registration (79d7f40)
- `internal/opt/memopt/memopt.go`: level gates first (`"memopt: nil graph"` / materialized refusal, untouched graph); `deriveRegions` groups live GMEM accesses by tensor operand node identity in first-access allocation order; `linearChain` enforces C1 (≤1 ImpactBy edge to an in-set pred, unique head, ≤1 effect-user successor in set ∪ {End}, single-path coverage) — failing regions skipped silently; fixpoint (rounds bounded by live-access count + 1) running rule (a) forward loads (most-recent-store discipline + proven path), rule (b) store-store elimination (stop at first load, walk past unequal stores, same-position cascades die in one walk), rule (c) dead-store sweep (no effect successor → RAUW data uses to value operand → erase); every mutation through one `MutContext{Pass: "memopt"}`.
- `internal/opt/memopt/verify.go`: mutates nothing; gates mirror; re-derives regions via the shared helper; asserts INTEGRITY (no killed/dangling ImpactBy records, no erased-def operands/users), FIXPOINT over eligible regions only (no forwardable load, no eliminable pair, no zero-use store), and SCOPED USERS (effect users ∈ region ∪ {End}).
- Wiring: `internal/opt/all/all.go` blank import; `internal/opt/registry.go` priority-scale comment gains the 112/113 anchors. Existing suites (including the BuildPipeline L1-prefix interop tests in simplify/gvn/typeinfer) stayed green unchanged.

### Task 3 — property tests + doc appendix (9e6d91a, 82eb552, 64f8413)
- `internal/opt/memopt/memopt_test.go` (26 tests, dce_test.go's style): the review's 10 families — gates; forwarding fires/blocked (position, intervening unequal store, no store); two same-position loads + hinge-6 middle-load chain splice; store-store (same-pos, load-blocked, diff-pos, walk-past-unequal, 3-store cascade); dead stores (unwired tail cascade, data-use RAUW, Return-wired tail never erased); C1 skip; head-removal edge drop; idempotence + Manager `[dce, gvn, memopt, memopt-verify]` + VerifyEach + registry/BuildPipeline 112/113 ordering; memopt-verify green sweep over all 15 fixtures, planted forwardable load flagged by node ID + kind, reversed pipeline rejected on "requires", own gates.
- `doc/optana/memory-opt.md`: appended "## Review verdict (implemented corrections)" — C2a/C2b/C2c corrections, C1/C3/C4 conditions, INV-A..G post-conditions, expectation note; original prose kept verbatim above it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Successor-edge enumeration matched (user, def) pairs, not records**
- **Found during:** Task 2 smoke check (before the Task 2 commit)
- **Issue:** `forwardLoad` collected successor edges by asking whether a use's user held *any* ImpactBy edge at the load — the Return's DATA-use record of the same load matched too (same user, same def), so a returned load was reported as having 2 effect successors and the pass errored on the canonical forwarding fixture.
- **Fix:** identify edge records by SSAUse identity (`e == u` compares the shared ref record), keeping data uses out of the splice set. Committed as part of 79d7f40 (caught pre-commit).

**2. [Rule 1 - Bug] Store-store walk-past compared the wrong pair**
- **Found during:** Task 3 (family 5, walk-past-unequal)
- **Issue:** the walk loop reassigned its cursor into the variable `sameProvenPath` read, so after one walk-past step the proven-path test compared (walked-past store vs candidate) instead of (origin store vs candidate) — S0(p)→S1(p')→S2(p) kept S0 alive.
- **Fix:** immutable `origin` anchor + separate `cur` cursor, in both `eliminateShadowedStores` and the verifier's `shadowingStoreBehind` mirror. Commit 82eb552.

**3. [Rule 3 - Blocking] doc/optana/memory-opt.md was not in git**
- **Found during:** Task 3 (doc appendix)
- **Issue:** the plan requires appending to `doc/optana/memory-opt.md`, but the file was repo-local untracked content — it did not exist in the worktree's committed history.
- **Fix:** brought the original prose into the worktree verbatim and appended the verdict section; the file enters git with commit 64f8413 so doc and implementation tell the same story in-repo.

Otherwise the plan executed exactly as written.

## TDD Gate Compliance

Task 1 followed RED→GREEN: `test(quick-261020-ejf)` commit 9bd284e (failing — `RemoveImpactBy` undefined) precedes `feat(quick-261020-ejf)` commit 5aade71 (green). Task 3's test-first discipline caught deviation 2 (the test commit lands before its fix commit).

## Verification

- `go test ./internal/ir/` — green (choke-point tests + zero regression).
- `go test ./internal/opt/` — green (registry/BuildPipeline ordering with memopt at 112, memopt-verify at 113).
- `go test ./internal/opt/memopt/ -v` — 26/26 green (all 10 families).
- `go test ./...` — full repo suite green; `go build ./...` + `go vet ./internal/opt/... ./internal/ir/` clean.
- Manual spot check: `opt.Lookup("memopt")` → priority 112; `opt.Lookup("memopt-verify")` → priority 113, Requires `[memopt]`; BuildPipeline places gvn-verify < memopt < memopt-verify.

## Commits

| Task | Commit | Type |
|------|--------|------|
| 1 (RED) | 9bd284e | test(quick-261020-ejf): pin MutContext.RemoveImpactBy choke-point semantics (C3) |
| 1 (GREEN) | 5aade71 | feat(quick-261020-ejf): add MutContext.RemoveImpactBy choke point (C3) |
| 2 | 79d7f40 | feat(quick-261020-ejf): add memopt (112) + memopt-verify (113) passes with registration |
| 3 | 9e6d91a | test(quick-261020-ejf): pin memopt's 10 review property-test families |
| 3 (fix) | 82eb552 | fix(quick-261020-ejf): anchor the store-store walk-past comparison at the origin store |
| 3 | 64f8413 | docs(quick-261020-ejf): append review-verdict appendix to memory-opt.md |

## Known Stubs

None — every rule fires on real fixtures; no placeholder logic anywhere in the pass.

## Threat Flags

None — no new trust-boundary surface (in-process transform; T-q-01/T-q-02 mitigations implemented as the C1 guard and the bounded fixpoint; T-q-03 accepted as-is with node-ID/kind-only diagnostics).

## Self-Check: PASSED

All 11 created/modified files exist on disk; all 6 task commits (9bd284e, 5aade71, 79d7f40, 9e6d91a, 82eb552, 64f8413) present on the branch; full `go test ./...` green at HEAD; working tree clean except this uncommitted SUMMARY (orchestrator owns the docs commit).
