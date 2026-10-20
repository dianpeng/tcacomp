---
phase: quick-261020-b4f
plan: 01
subsystem: ir-iter-restructure
tags: [ir, iter, boundeddimvar, loopnest, simplify-lower, gvn, loopforest]
requires:
  - quick-261019-uoq (GVN loop fusion + loopforest analysis)
  - quick-261019-ktm (simplify-lower / IterCluster lowering)
provides:
  - Iter node in the sound 2-parameter formation: NewIter(origin, bdv, parent) — operand 0 a BoundedDimVar value object, operand 1 the parental Iter present only when nested; unbounded form deleted
  - Wired loop nests from real builder graphs: simplify-lower materializes DimVar → BoundedDimVar(TensorDimOf) → Iter(biv, parent) chains per as-order, interned per (dim, parent-chain) — the loop-nest forest shows real trees, not singletons
  - GVN prefix fusion permanently pinned (ABC ∩ ABCD → common prefix fused, D re-parented onto fused C; forest = one root, depths 0/1/2/3)
affects:
  - internal/ir (node_loop, catalog, verify, print + 3 test files)
  - internal/opt/simplify (lower + tests)
  - internal/opt/gvn, internal/opt/typeinfer, internal/analysis/loopforest, internal/analysis/sideeffect (test fixtures + doc comments)
key-files:
  created: []
  modified:
    - internal/ir/node_loop.go
    - internal/ir/catalog.go
    - internal/ir/verify.go
    - internal/ir/print.go
    - internal/ir/value_test.go
    - internal/ir/effect_test.go
    - internal/ir/node_convention_test.go
    - internal/opt/simplify/lower.go
    - internal/opt/simplify/simplify_test.go
    - internal/opt/gvn/gvn_test.go
    - internal/opt/typeinfer/typeinfer_test.go
    - internal/analysis/loopforest/loopforest_test.go
    - internal/analysis/sideeffect/sideeffect_test.go
decisions:
  - Iter = (BoundedDimVar, parentIter?) per user overrule of the parent-appended-last workaround — matches doc verbatim ("parental loop, if any, as its second input"); soundness fix: every Iter bounded by construction, BIV compared as ONE value object (SameBIV substrate), no half-formed loops
  - Catalog arity stays -1 (variadic) with the {1,2} operand-count invariant enforced by the verifier's shape check — fixed arity 2 would allocate a zero-handle slot on parentless iters, which checkOneUse flags as unwired (planner's mechanically-verified deviation)
  - Interning in simplify-lower keyed (dim, parent-chain): shared as-prefixes collapse to one chain, distinct nests stay distinct; MemLoad axis resolution is cluster-local
  - The old taint-premise test (Iter with MemLoad init operand) became structurally unreachable — an Iter's operands can no longer carry an effectful node — renamed to TestIterStaysPureOverValueObjects pinning the categorical truth ("Iter is itself side effect free" now holds by construction)
metrics:
  duration: ~10 min
  completed: 2026-10-20
---

# Phase quick-261020-b4f Plan 01: Iter (biv,parent?) Restructure + Nest Wiring + Prefix-Fusion Pins Summary

One-liner: Restructured KindIter to the sound 2-parameter formation — `NewIter(origin, bdv, parent)` with the BoundedDimVar value object as the header parameter and the parental Iter as the (optional) second input — deleted the unbounded form, made simplify-lower build wired DimVar→BIV→Iter nest chains per as-order so real kernel graphs produce forest trees instead of singletons, and permanently pinned GVN prefix fusion (common-prefix fusing with the extra level re-parented onto the fused prefix).

## What Was Built

### Task 1 — RED: fixtures rewritten to the new formation (57696a7)
All Iter fixtures across 9 test files rewritten to `NewIter(origin, bdv, parent?)` with DimVar/BoundedDimVar objects built first; new simplify nest-wiring pins (TestLowerWiresNestChain, TestLowerSharedPrefixNotFlat, as-order respect) fail against the old IR.

### Task 2 — GREEN: the IR restructure (b917d46)
- `internal/ir/node_loop.go`: NewIter(origin, bdv, parent) — 1-op nest root / 2-op nested header; bounded/unbounded split deleted.
- `internal/ir/verify.go`: iter shape check now enforces operand 0 = bounded dimvar, operand 1 = parental iter, operand count ∈ {1,2}, with named violations.
- `internal/ir/catalog.go` + `print.go`: row doc updated; renderer goes through the BIV.
- Production caller migration: lower.go:169 (the only one).

### Task 3 — GREEN: wired nests in simplify-lower (7b801c0)
Per as-order chains materialized as value objects, interned per (dim, parent-chain): `as [a,b,c]` lowers to a depth-0/1/2 chain; `as [a,b]/[a,c]` shares ONE a-root with separate b/c children (not flat); `as [b,a]` yields root b child a; MemLoad axes resolve cluster-locally.

### Task 4 — Pins: prefix fusion + forest consequence (90d66bb)
- TestGVNIterFusionPrefix: nest2's common prefix erased in one gvn run; D2 re-parented onto C1 (operand-slot assertion); gvn-verify green.
- TestForestAfterGVNPrefixFusion: one root, chain depths 0/1/2/3.
- Both passed on the first run after Tasks 2-3 (pure pins of verified behavior).

## Zero-change validations (plan's mechanical analysis held)
- gvn.go / value.go / sideeffect.go logic: untouched — identity keying is formation-agnostic; root [biv] vs child [biv, parent] never merge (slot-count); DimVar→TensorDimOf→BIV→Iter cascade is generic one-run CSE.
- typeinfer rules: untouched — Iter/DimVar/BIV/IV/TensorDimOf already type u32.
- End-node/Env/SeaRoots model (K+1 roots, returned environment): untouched — already correct.
- doc/optana/: read-only.

## Verification Results
- `go test ./...`: 15 packages ok, zero failures (verified post-merge on master)
- `go vet ./...` clean; touched files gofmt-clean
- Key pins green: TestLowerWiresNestChain, TestLowerSharedPrefixNotFlat, TestGVNIterFusion{Bounded,NestedAwareness,Prefix}, TestForestAfterGVN{Fusion,PrefixFusion}, TestNestedChain, TestTwoNestsTwoRoots

## Commits
| Commit | Type | Content |
|--------|------|---------|
| 57696a7 | test (RED) | Rewrite Iter formation pins to (biv, parent?) + simplify nest-wiring pins |
| b917d46 | feat (GREEN) | Restructure KindIter to (biv, parent?) per doc/optana/{loop_fusion,induction_variable}.md |
| 7b801c0 | feat (GREEN) | Build wired nests in simplify-lower — value-object chains interned per (dim, parent) |
| 90d66bb | test | Pin GVN prefix fusion + post-fusion forest chain on the (biv, parent?) formation |

## Deviations from Plan
None. Environment notes: doc/optana/ was read from the main checkout (untracked, absent in the worktree); pre-existing gofmt drift in internal/analysis/doc.go, internal/ir/ssa.go, and effect_test.go's memoryNodes map predates this task (verified at base) — out of scope.

## Process Note
The executor's SUMMARY.md draft was written untracked inside its worktree and lost to the worktree cleanup; this SUMMARY was reconstructed by the orchestrator from the executor's verified completion report (commits, outcomes, and verification results cross-checked against the merged master tree).

## Auth Gates
None.

## Known Stubs
None.

## Threat Flags
None.
