---
phase: quick-261022-f5o
plan: 261022-f5o
subsystem: compiler-ir
tags: [loopforest, loop-spec, category-interface, dedup, gsd-quick]

# Dependency graph
requires:
  - phase: 261022-b2m lineage
    provides: "the membership-edges-only linking constraint (D-04) discovered in CR-01's fix — now a documented, tested property of the shared analysis"
provides:
  - "ir.LoopSpec / ir.LoopNesting — the FamilyLoop category interface (IsLoop/ParentEdge/BodyRoot); only ir knows the operand layouts"
  - "ONE generic loopforest.Analyze over both loop dialects (HIR iter parent-operand links; LIR forloop body-containment), 0-based depths"
  - "schedule pass nest.go = thin adapter (forNest mirror, depth+1) + pass-local bodySetOf/reachSetOf placement derivations"
affects: ["future analyses over loop structure (one LFA to consume, category interface to extend)", "final_lower (unchanged — consumes the scheduler's products, not the forest)"]

key-files:
  created:
    - internal/ir/loopspec.go
    - internal/ir/loopspec_test.go
    - internal/analysis/loopforest/loopforest_lir_test.go
  modified:
    - internal/analysis/loopforest/loopforest.go
    - internal/analysis/loopforest/loopforest_test.go
    - internal/opt/schedule/nest.go

key-decisions:
  - "One spec, no stage detection: at any pipeline stage exactly one loop dialect is live (tile-lower retires iters), so ir.LoopNesting serves both; a mixed graph links coherently (pinned)"
  - "Membership-edges-only linking is THE reason the interface exists (D-04): completion edges (ImpactBy at a foreign loop root) order, never nest — pinned by TestImpactByNeverNests"
  - "Depth base root=0 (the historic iter convention; all tiling uses relative); the scheduler maps +1 at its single nest.go adapter point (D-02)"

requirements-completed: [QUICK-261022-f5o]

# Metrics
duration: ~25min (executed inline by the orchestrator after the user's model-switch kill of the dispatched executor)
completed: 2026-10-22
---

# Quick 261022-f5o: LFA unification behind ir.LoopSpec Summary

**One generic loop-nest forest analysis over the FamilyLoop category
interface serves both the HIR iter dialect (tiling) and the LIR forloop
dialect (schedule) — the scheduler's duplicated collect/link/depth
algorithm is gone.**

## Execution note

The dispatched executor was killed by the user mid-read (model switch); its
worktree was cleaned up with nothing partial. The orchestrator executed
both tasks inline per the plan.

## Task Commits

1. **T1 (71ef0b9, refactor)** — ir/loopspec.go (LoopSpec + LoopNesting),
   generic loopforest.Analyze, spec pins, LIR/cross-dialect/completion-edge
   pins.
2. **T2 (commit follows, refactor)** — nest.go adapter; scheduler.go zero
   changes; full repo green.

## Deviations from Plan

**1. [Rule 1] One existing pin amended (TestFlatGraphAllSingletons).**
The plan required existing loopforest tests "green UNMODIFIED where
possible", but this test explicitly pinned the analysis's OLD HIR-only
scope ("only KindIter is collected", live ForLoop excluded). The unified
spec deliberately changes exactly that behavior, so the pin was amended:
3 roots (it1, it2, the live forloop), forloop row moved from the exclusion
map to the membership check, IterCluster/DimVar exclusions intact. All
other loopforest tests and the whole tiling suite passed unmodified.

## Verification

- Full gate at both commits: build/vet green, 17 packages ok, gofmt clean.
- Plan checks: LoopNesting referenced 3× in the LFA; the old O(n²) link
  loop gone from nest.go; scheduler.go diff empty.
- Behavioral identity: tiling suites green unmodified (HIR forest
  byte-identical); schedule/pipeline suites green unmodified (LIR forest
  preserved through the adapter, depths +1).

## Self-Check: PASSED

---
*Plan: quick-261022-f5o*
*Completed: 2026-10-22*
