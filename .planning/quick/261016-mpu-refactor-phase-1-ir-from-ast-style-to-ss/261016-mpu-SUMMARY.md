# Quick Task 261016-mpu Summary

**Task:** Refactor phase 1 IR from AST-style to SSA/CFG optimizing-compiler IR per doc/review/phase1_1.md
**Date:** 2026-10-16
**Status:** Complete — 3/3 tasks, final gate green

## Final Gate

```
go build ./... && go test ./... -count=1 && go test ./pkg/ -run TestDeterminism -count=2 && go vet ./...
```

All green. Determinism byte-identical across double-run. Goldens regenerated; the regen
matched the Task-1 hand-edit of `matmul_scale.ir` exactly (zero diff).

## Commits

- `da60a24` — refactor(261016-mpu-01): reshape IR to SSA/CFG optimizer shape — InputParam nodes,
  SSAUse/SSADef handles, intrusive Statement list, payload uses (P1R-01/02/03/06)
- `b51724b` — refactor(261016-mpu-02): resolve builder symbols on the fly; building-phase DCE —
  mark/fixpoint/wipe with per-statement ownership, legality windows, pkg wiring (P1R-05/07/08)
- `c470396` — feat(261016-mpu-03): add lowered node suite (dimvar/bdimvar, tiles, for-loop) with
  IV/BIV relations — KindVersion=2, fail-closed shape checks, doc/tca.md §5.13 + #8/#9 goldens (P1R-04/05)

## What Changed

- **Symbol removal / SSA resolution (P1R-01, P1R-03):** name-keyed `Param` record deleted; kernel
  params are `SSAUse` handles pointing at `KindInputParam` nodes (name retained as debug-only
  payload). Builder performs single-pass, on-the-fly symbol-table resolution; the final IR graph
  contains no unresolved symbols. No phis (single BB per kernel in v0), per review.
- **CFG/BB/statement shape (P1R-02, P1R-06):** new `internal/ir/cfg.go` — CFG → BasicBlock →
  statements stored in an intrusive doubly-linked list, giving O(1) removal
  (`MutContext.Erase` no longer searches an array).
- **Mutation support (P1R-07):** use-list/RAUW machinery preserved; handles now hold resolved
  `ir.Node` (see deviations).
- **Building-phase DCE (P1R-08):** new `internal/ir/dce.go` — mark dead unreferenced
  side-effect-free SSADefs, dataflow worklist fixpoint propagates death marks, wipe via the
  intrusive list.
- **Specialized lowered nodes (P1R-04, P1R-05):** per-concern submodule files —
  `dimvar.go` (DimVar/BoundedDimVar + SameIV/SameBIV/Includes/BelongTo per
  doc/optana/induction_variable.md), `tile.go` (loadtile/storetile → FamilyTileOp),
  `loop.go` (lowered for-loop → FamilyLoop). Single-tier IR retained; kinds are append-only;
  KindVersion bumped to 2.

## Deviations (4, all intentional)

1. **Handles hold resolved `ir.Node`, not bare SSAUse handles** — a handle-held record would be
   slot-unowned and fail the verifier.
2. **DCE statement ownership uses full operand/payload/lambda DFS** instead of `StmtNodes` —
   bare binop roots are not stage nodes; `StmtNodes` would have failed conservative-live for
   everything.
3. **Walk push-loop swap** to restore the documented params-first root order.
4. **§5.3's schematic `// statement #1/#2` comment blocks** left unchanged — hypothetical
   pseudo-statements, not the pinned matmul oracle; representation did not genuinely change there.

## Constraints Honored

- Code-only commits with attribution trailer; SUMMARY.md / PLAN.md / STATE.md / ROADMAP.md
  untouched by task commits.
- ROADMAP.md not updated (quick task is separate from planned phases).
- Pure Go stdlib-only — no new external imports.
