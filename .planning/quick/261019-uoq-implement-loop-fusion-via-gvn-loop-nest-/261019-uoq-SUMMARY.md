---
phase: quick-261019-uoq
plan: 01
subsystem: opt-gvn-loop-fusion
tags: [gvn, cse, sideeffect, loopfusion, loopforest, analysis-registry]
requires:
  - quick-261018-nga (GVN pass + side-effect analysis + gvn-verify)
  - quick-261019-jck (analysis registry / composition-root pattern)
provides:
  - GVN loop fusion (bare Iter side-effect free; identical headers, CSE'd bounded headers, and whole parent-wired nests fuse in one run)
  - internal/analysis/loopforest ("loop-forest" registered analysis — the loop-nest forest loop tiling will consume)
affects:
  - internal/analysis/sideeffect
  - internal/opt/gvn
  - internal/analysis/all
tech-stack:
  added: []
  patterns:
    - lockstep mirror edit (sideeffect.baseVerdict and gvn baseVerdictMirror dropped KindIter in ONE commit so ANALYSIS-CLOSURE stays a fixpoint)
    - analysis registry self-registration via init() (loopforest mirrors sideeffect)
key-files:
  created:
    - internal/analysis/loopforest/loopforest.go
    - internal/analysis/loopforest/loopforest_test.go
  modified:
    - internal/analysis/sideeffect/sideeffect.go
    - internal/analysis/sideeffect/sideeffect_test.go
    - internal/opt/gvn/verify.go
    - internal/opt/gvn/gvn.go
    - internal/opt/gvn/gvn_test.go
    - internal/ir/value.go
    - internal/analysis/all/all.go
    - internal/analysis/all/all_test.go
decisions:
  - Bare Iter reclassified side-effect free per doc/optana/loop_fusion.md [1]; IterCluster/ForLoop/Return keep the [2.3] closure; orderSensitive unchanged (FamilyLoop taint is the fusion soundness gate)
  - gvn-verify baseVerdictMirror hand-edited in the same commit (mirror independence is the point — drift fails loudly)
  - Parent-wired iters use the parent-APPENDED-LAST operand layout ([dim, init, step, parent]); GVN keys on full slot identity and the forest scans all operands, so position is convention, not load-bearing — tests use one consistent layout
  - loopforest parent = FIRST live-Iter operand in slot order (doc's single-parent convention, deterministic); forest is a separate Tree structure, computed on demand, read-only
metrics:
  duration: ~6 min
  completed: 2026-10-19
---

# Phase quick-261019-uoq Plan 01: Loop Fusion via GVN + Loop-Nest Forest Summary

One-liner: Reclassified the bare Iter as side-effect free (baseVerdict + lockstep gvn-verify mirror drop) so GVN fuses identical loop headers, CSE'd bounded headers, and whole parent-wired nests in one run via the RAUW cascade, plus a new `internal/analysis/loopforest` package (registered "loop-forest") implementing the doc's reversed-edge forest algorithm for the future loop-tiling consumer.

## What Was Built

### Task 1 — KindIter reclassification = GVN loop fusion (TDD: RED 3a46d79, GREEN aa03d51)

- `internal/analysis/sideeffect/sideeffect.go`: `baseVerdict` drops `ir.KindIter` (case is now `KindIterCluster, KindForLoop, KindReturn`); package comment rewritten — the propagation clause is now LIVE (bare iter pure and GVN-eligible; iter with effectful operand taints via orderSensitive FamilyLoop). `orderSensitive` untouched.
- `internal/opt/gvn/verify.go`: `baseVerdictMirror` drops `KindIter` in the SAME commit — gvn-verify's ANALYSIS-CLOSURE stays a fixpoint; comment documents the hand-edit-in-lockstep rule.
- `internal/opt/gvn/gvn.go`: doc-comment only — iter value numbering is nested-aware (identity keying over kind, slot count, and every operand-slot def identity including parent-Iter operands; whole nests fuse through the parent-operand RAUW cascade; effectful-fed iters excluded via the taint). No algorithm change.
- `internal/ir/value.go`: comment only — stale "(two iters" one-of-a-kind example swapped to "(two stores".
- `internal/analysis/sideeffect/sideeffect_test.go`: bare Iter pinned PURE (split out of TestImplicitControlFlowEffectful into TestIterIsPure, unbounded + bounded forms); TestPureKindsNeverPropagate guard inverted; TestOrderSensitivePropagation formula drops KindIter; NEW TestIterTaintsThroughEffectfulOperand pins the taint gate (init = MemLoad ⇒ effectful; const init ⇒ pure).
- `internal/opt/gvn/gvn_test.go`: TestGVNIterArithmetic second half inverted (identical headers fuse, arithmetic cascades in one run); TestGVNNeverMergesSideEffects moves iters out of the exclusion set (count drops by exactly 1, store chain untouched); NEW TestGVNIterFusionBounded (equal-valued distinct const bounds CSE then fuse in one run; 3-op vs 4-op arity never fuses) and TestGVNIterFusionNestedAwareness (whole parent-wired nest fuses in one run; children of different parents never fuse).

### Task 2 — loopforest analysis + registry wiring (TDD: RED f113643, GREEN 87424cf, wiring 7fc95e6)

- `internal/analysis/loopforest/loopforest.go`: Tree/Forest types, `Analyze` implementing the doc's forest algorithm verbatim (collect live KindIter in allocation order → link REVERSED edges with first-live-Iter-operand parent convention and root-set removal → DFS depths from roots, cycle-safe by construction), `Roots()` (allocation order, fresh copy), `NodeOf` (fail-closed on nil/erased/foreign), `Name()` "loop-forest", init self-registration as AnalysisSpec "loop-forest". Separate structure, computed on demand, read-only.
- `internal/analysis/loopforest/loopforest_test.go`: 8 contract pins — TestName, TestFlatGraphAllSingletons (only KindIter collected), TestNestedChain (reversed edges, depths 0/1/2), TestTwoNestsTwoRoots, TestErasedIterSkipped, TestNilGraphEmpty, TestDeterministicAndReadOnly (fingerprint equality + no mutation), TestForestAfterGVNFusion (post-gvn single-survivor bridge pin).
- `internal/analysis/all/all.go` + `all_test.go`: blank import fires registration; Lookup("loop-forest") membership and BuildAll → loopforest.Forest construction pinned (membership-only assertions, never registry size).

## GVN Soundness Verdict

Established by the orchestrator before execution: SOUND — no GVN restructuring; gate reclassification only.

1. **One global hash table is the correct degenerate case of dominator-scoped GVN.** The Level-1 sea has exactly one implicit region (one BB per kernel, no control flow), and the pass refuses materialized/CFG-carrying graphs — so there is no dominance scope to partition by; a single table is exactly dominator-scoped GVN over a one-region graph.
2. **Identity keying (amendment D) implements the doc's nested-aware deep equality incrementally.** Def-before-use allocation order means operands CSE before their users, and RAUW rewires use records so users hash against their FINAL representatives — so whole parent-wired nests fuse level-by-level in one run (the doc's "a loop nest will be GVNed only when all NODEs are equal"). Pinned by TestGVNIterFusionNestedAwareness(a).
3. **The doc's verification requirement already exists.** "All iter nodes in Prefix(A) must have ZERO USE" is gvn.go's post-RAUW zero-use hard error, plus the Item-7 effect-replacement assert — both retained unchanged.
4. **Residual risks accepted.** Non-topological graphs (pass-mutated allocation order) may need a second gvn run — sound either way, since only PROVEN duplicates are removed. FamilyLoop remains order-sensitive, so effectful-fed iters stay out of GVN — pinned by TestIterTaintsThroughEffectfulOperand.

## Verification Results

- `go test ./...` fully green (unit + integration; e2e gate env-guarded, unaffected)
- `go vet ./...` clean
- gofmt: every file touched by this task is clean; three PRE-EXISTING dirty files at baseline 40c449e (internal/analysis/doc.go, internal/ir/effect_test.go, internal/ir/ssa.go) left untouched — logged in deferred-items.md
- Executor spot check (throwaway test, run and removed): two identical iters through the real L1 prefix pipeline (dce → dce-verify → gvn → gvn-verify via opt.BuildPipeline with VerifyEach) — exactly one iter survives, consumer rewired, ir.Verify clean
- `git status`: nothing under doc/optana/ modified; working tree clean after scratch removal

## Commits

| Commit | Type | Content |
|--------|------|---------|
| 3a46d79 | test (RED) | Invert iter side-effect pins to GVN fusion semantics (sideeffect_test + gvn_test) |
| aa03d51 | feat (GREEN) | Reclassify KindIter side-effect free — GVN loop fusion (sideeffect + gvn-verify mirror + doc comments) |
| f113643 | test (RED) | Pin loop-forest analysis contract (package absent = red state) |
| 87424cf | feat (GREEN) | Implement loopforest analysis (doc's forest algorithm) |
| 7fc95e6 | feat | Wire loopforest into analysis/all composition root |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] loopforest_test map key type**
- **Found during:** Task 2 GREEN (surfaced only once the package materialized and the test compiled)
- **Issue:** `usesBefore` map keyed `int` but `BasePtr().ID()` is `uint64` — compile error in the RED-authored test.
- **Fix:** Keyed the map `uint64`; included in the GREEN commit 87424cf.
- **Files modified:** internal/analysis/loopforest/loopforest_test.go

**2. [Rule 1 - Bug] gofmt comment-alignment drift in freshly edited tests**
- **Found during:** Task 1 RED and Task 2 RED commits (gofmt -l flagged my own just-edited/new files: comment alignment groups broken by reworded comments).
- **Fix:** `gofmt -w` on the affected files, folded into the same commits (RED commit amended before push-less finalize; loopforest test formatted before its RED commit).
- **Files modified:** internal/analysis/sideeffect/sideeffect_test.go, internal/analysis/loopforest/loopforest_test.go

### Out-of-scope (not fixed, logged)

- Three gofmt-dirty files pre-existing at baseline — see deferred-items.md in this directory.

Otherwise the plan executed exactly as written: no gvn.go algorithm change, no edits under doc/optana/, no changes to ir/verify.go / dce.go / effect.go / schedule.go / typeinfer.

## Auth Gates

None.

## Known Stubs

None — no placeholder values, no unwired data paths; all exported behavior is pinned by tests.

## Threat Flags

None. No security-relevant surface beyond the plan's threat model: T-uoq-01/02 mitigations verified in place (identity-keyed ValueEqual + Item-7 assert + post-RAUW zero-use assert + ANALYSIS-CLOSURE/UNIQUE-REPRESENTATIVE mirrors retained; taint pinned by TestIterTaintsThroughEffectfulOperand; mirror drop lockstep-committed), T-uoq-03 mitigated by the cycle-safe construction and fail-closed NodeOf.

## Self-Check: PASSED

- Created files exist: internal/analysis/loopforest/loopforest.go, internal/analysis/loopforest/loopforest_test.go — FOUND
- Modified files exist: sideeffect.go, sideeffect_test.go, verify.go, gvn.go, gvn_test.go, value.go, all.go, all_test.go — FOUND
- Commits exist: 3a46d79, aa03d51, f113643, 87424cf, 7fc95e6 — all verified in git log
