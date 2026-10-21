---
phase: quick-261021-i1o
plan: 01
subsystem: opt-tiling
tags: [tiling, portion-grammar, psched-search, refactor]
requires:
  - doc/optana/loop_tiling.md final model (261021-gca patch state)
provides:
  - Decide pass (internal/opt/tiling) matching the final loop_tiling.md constraint/objective model
affects:
  - future loop-tile-lower pass (record consumers: CTA LoC == MaxCTA, warp LoC portion multiples)
tech-stack:
  added: []
  patterns:
    - portion-grammar LoC completion (quantum = warps/PortionWarp) replacing greedy preset loops
    - v0 policy gate as an enumeration skip (odometer still runs all 3^K)
key-files:
  created: []
  modified:
    - internal/opt/tiling/record.go
    - internal/opt/tiling/search.go
    - internal/opt/tiling/verify.go
    - internal/opt/tiling/tiling_test.go
    - internal/opt/tiling/verify_test.go
    - internal/opt/tiling/overlap_test.go
decisions:
  - Commit boundaries reshaped for per-commit green gates (tasks 1+2 compile-coupled; task 3 split in two)
  - AutoWarpPortion hoisted to a named constant (plan specified the literal 4 in a comment)
metrics:
  duration: 13m05s
  completed: 2026-10-21T05:15:47Z
  tasks: 3/3
  commits: 3
---

# Quick 261021-i1o: Refactor internal/opt/tiling to the final loop_tiling.md model Summary

One-liner: The tiling decide pass now runs the final portion-grammar constraint set (PortionCTA=10/PortionWarp=8, CTA dim exhausting MaxCTA=100, warp dims as portion multiples of the 2-warp quantum) with a single-term traffic-avoided objective, a v0 policy gate (<=1 CTA + <=1 independent WARP dim per LNT), a CTA-carrying fallback, and tiling-verify invariants matching — overlap analysis, window mechanics, record side table, and the matmul worked-example expectations untouched.

## What Was Built

### Params surface (record.go)

- Package-level named constants: `PortionCTA = 10`, `PortionWarp = 8` (doc: PORTION_CTA / PORTION_WARP — the latter is a portion count, not a warp count), plus `AutoWarpPortion = 4` for the auto warp-portion rule.
- `Params`: deleted `ParallelWeight`, `ReuseWeight`, `CTALoCs`, `WarpLoCs`; added `WarpPortion int` (0 = auto: AutoWarpPortion when the vector carries a CTA dim, else PortionWarp full).
- `DefaultParams`: MaxCTA 100 (Ampere 108 rounded DOWN to the PortionCTA multiple), MaxThreadsPerCTA 512 (= 16 warps), WarpPortion 0, everything else unchanged.
- `Validate` now mirrors the runtime ASSERT-AND-TRAP pre-conditions: MaxCTA a positive multiple of PortionCTA; MaxThreadsPerCTA a multiple of 32 with warp count a multiple of PortionWarp (256-thread minimum); WarpPortion in [0, PortionWarp]; remaining positive bounds and the TileSizes list check kept; weight/preset checks deleted.
- Package doc comment updated to cite the final model (portion grammar, single-term objective, v0 policy).

### Search (search.go)

- Portion completion replaces the greedy preset loops: the (at most one) CTA dim gets LoC = MaxCTA (portion -1/full); the (at most one) independent warp dim gets LoC = warpPortion x warpsPerPortion where warpsPerPortion = (MaxThreadsPerCTA/32)/PortionWarp — the CTA dim's coupled tile-inside walk implicitly consumes the remaining warp portions, its stride derived by the lower pass, not recorded. Sequential stays LoC 1. The prod/warpSum accounting is deleted.
- v0 POLICY FILTER in searchSchedule: SoC vectors carrying >1 SoCTA or >1 SoCWARP are skipped (cited to loop_tiling.md "Composition Semantics, CTA x WARP": same-category multi-dim is FUTURE pending combined-slicing arithmetic); the odometer still runs over all 3^K. The >=1-CTA gate (`hasCTALevel`) is kept.
- Single-term objective: `res.score = float64(traffic)`; the locSum computation and weighted sum are deleted.
- The defensive fallback now carries a CTA dim (outermost window level = SoCTA, rest sequential, t=1) — all-sequential is INVALID under the >=1-CTA ruling; always feasible since t=1 promotes nothing (SMEM sum 0).

### tiling-verify (verify.go)

- FEASIBILITY invariants reworked to the final model: CTA LoC must EQUAL MaxCTA (portion -1/full); warp LoC must be a warp-portion multiple (LoC/quantum in [1, PortionWarp]); v0 policy enforced on the record (at most one CTA and one independent WARP entry per LNT); >=1 CTA dim per LNT kept; sequential LoC 1 and positive LoC/TileSize kept. Legacy CTA-product / warp-sum bounds removed as subsumed (Params.Validate re-checks the assert-and-trap divisibility).
- New corruption classes pinned: a well-formed second CTA dim (v0 policy) and a non-portion warp LoC.

### Tests

- Ported: TestParamsValidate (portion divisibility error classes replace weight/preset classes), TestFeasibilityBounds (minimal MaxCTA=10 exhausted exactly; PortionWarp-warp/256-thread budget), TestTrafficArithmeticAndOptimality + enumerateBest (mirror the policy filter, portion completion, single-term score).
- Added: TestDefaultParams (100/512/auto, presets and weights absent), TestPortionCompletion (evalSchedule unit pin: CTA exhausts MaxCTA, warp LoC = portion x quantum under auto and explicit WarpPortion), TestV0PolicyFilter (7-deep window and matmul records carry exactly one CTA dim and at most one independent WARP dim).
- UNCHANGED by design: TestMatmulOverlap and all overlap-analysis expectations (SET {lhs,rhs}@iterK, lhs tie iterN covering [K], rhs tie nil covering [N][K]) — zero diff lines touch them.

## Verification

- `go build ./...` clean.
- `go vet ./internal/opt/...` clean.
- `go test ./internal/opt/tiling/... -count=1` green (39 tests incl. new pins).
- `go test ./internal/opt/... -count=1` green; full-repo `go test ./... -count=1` green.
- `grep -rn "ParallelWeight|ReuseWeight|CTALoCs|WarpLoCs" internal/` → no matches.
- Registry anchors 114 (tiling) / 115 (tiling-verify) untouched; no files outside internal/opt/tiling/ modified.

## Commits

| Commit | Content |
|--------|---------|
| 8a48207 | refactor(261021-i1o): params + search to the final portion-grammar model (tasks 1-2) |
| 111b960 | refactor(261021-i1o): tiling-verify invariants to the final model (task 3/2) |
| 0a91779 | test(261021-i1o): pin the final-model behaviors (task 3/3) |

## Deviations from Plan

### Commit-boundary reshaping (compile coupling)

- **Found during:** Task 1 commit preparation.
- **Issue:** The plan's per-task file partitions cannot each yield a green commit: deleting `ParallelWeight`/`CTALoCs`/... from record.go breaks compilation of search.go (the greedy preset loops) and both test files that construct Params — and task 1's own verify step demands a green build with "no ParallelWeight/CTALoCs remain". A record.go-only commit would be red (bisect-hostile), violating the task-context gate "all green BEFORE each commit".
- **Fix:** Commit 1 carries plan tasks 1+2 (record.go + search.go) plus the mechanical test port; plan task 3 is split into commit 2 (verify.go + verify_test.go invariants) and commit 3 (added tests). Every commit passes build/vet/test.
- **Files modified:** unchanged set — only the commit grouping differs from the plan's task grouping.

### Minor placements

- **[Rule 3 - Blocking] Portion-divisibility validation-error test cases** live in commit 1's TestParamsValidate port (the function had to compile against the new surface) rather than in task 3's commit as the plan listed. Content identical to the plan's intent.
- **[Rule 2 - Convention] AutoWarpPortion named constant**: the plan specified the auto rule's literal 4 in a field comment; hoisted to a named package constant per the repo's no-magic-numbers standard. search.go and the tests reference it.
- **Fallback direct test not added**: the CTA-carrying fallback is defensive-only (unreachable for validated Params — the t=1 point of any policy-clean CTA-carrying vector is always feasible, SMEM sum 0). It is pinned indirectly by the record-level "zero-cta never selected" test (R4's observable surface).

## Auth Gates

None — no external services or credentials involved.

## Known Stubs

None — no placeholder data paths were introduced; the record's coupled-inside warp stride is intentionally absent by design (derived by the future loop-tile-lower pass, per the plan and doc).

## Self-Check: PASSED

- Files: internal/opt/tiling/{record,search,verify,tiling_test,verify_test,overlap_test}.go all modified as listed (git diff --stat vs 9f7c2eb confirms).
- Commits 8a48207, 111b960, 0a91779 present on worktree-agent-a77189773eb3a5d76 (git log verified).
- Gates re-run after the final commit: build, vet, and both test scopes green.
