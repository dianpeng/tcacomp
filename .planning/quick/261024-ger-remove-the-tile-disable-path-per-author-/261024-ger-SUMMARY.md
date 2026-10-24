# Quick Task 261024-ger Summary: Remove the Tile-Disable Path per the 2026-10-24 Ruling

**One-liner:** the T==1 single-form path is deleted from the compiler; every dim is now tiled at tile >= 2, enforced fail-closed at four gates, with the test suite re-targeted to the minimal-legal-tile endpoint {CTA,100,2}/{WARP,8,2} and the full gate ladder (hermetic, goldens, CPU runtime, A6000 e2e + sanitizers) green.

## The ruling (normative)

The compiler NEVER generates a non-tiling schedule. Tile size 1 is not a legal
schedule. The author's 2026-10-24 ruling replaces the whole degenerate-case
family (the SEQ-single identity-indexExpr wall, HI-02's scoping bug, the dead
`betweenBlocks` branch, the missing T==1 WARP mirror gate) with one
enforcement rule at every boundary. The plan's design rulings R1–R6 were
implemented as settled.

## What was deleted (Task 1, commit 0079a6a)

- **`internal/opt/tiling/lower_rewrite.go`**: the entire `case d.tile == 1:`
  branch of `generateDimExprs` — the SEQ/CTA/WARP single-form constructions
  and the T==1 WARP single-form mirror gate; the `dimRewrite` fields
  `single`, `singleIV`, `singleIVNode`, `singleBound`, `singleAttrs`.
- **`internal/opt/tiling/lower.go`**: the `checkFillPlacement` single-IV loop
  entry.
- **`internal/opt/tiling/record.go`**: the `ReasonTileOne` judge constant.
- **`internal/opt/tiling/search.go`**: the judge's `== 1` promotion gate;
  the `t=1` CTA-carrying fallback in `searchSchedule` (the unreachable
  defensive path — replaced by a hard error).
- **`internal/opt/finallower/reducer.go`**: `refuseThreadPartitionedConsumers`
  and `warpSingleIVs` (keyed on `prov.Tile == 1`, unconstructible now) plus
  gate (b) of `moveCombineConsumers`; the `home := x` fallback (a missing
  provenance `Inside` is record corruption, every dim is tiled).
- **Docs**: `doc/optana/loop_tiling.md`, `doc/optana/loop_tile_lower.md`,
  `doc/optana/final_lower.md`, `doc/tca.md` §5.10, `doc/runtime_objects.md`
  amended to the never-tiling model (F4 floor-2 drop, the zero-between-dims
  degenerate, Inside always populated). `doc/review/*.md` left historical
  per R5. `doc/rt.md` §4.6 point 3 (the WarpReducer trailing-barrier rider)
  VERIFIED present — no edit needed.

## The four enforcement gates (with their rejection messages)

1. **Params.Validate** (`record.go:445`):
   `tiling: TileSizes entry %d is below 2 — tile size 1 is not a legal schedule: the compiler never generates a non-tiling schedule (2026-10-24 ruling); every entry must be >= 2`
2. **tiling-verify** (`verify.go:192`):
   `tiling-verify: iter node %d carries LoC %d / TileSize %d (LoC must be >= 1 and TileSize >= 2 — the compiler never generates a non-tiling schedule, the 2026-10-24 ruling)`
3. **tile-lower rewrite backstop** (`lower_rewrite.go:397`) — kills
   hand-stamped Verified records and bad budget overrides that bypassed
   gates 1–2:
   `tile-lower: iter node %d carries effective tile size %d (record decision or budget override) — tile size 1 is not a legal schedule: the compiler never generates a non-tiling schedule (2026-10-24 ruling; gate 3 backstop — Params.Validate and tiling-verify are gates 1-2)`
4. **the searcher** (`search.go:155-157`) — the deleted t=1 fallback's
   corner now errors:
   `tiling: LNT rooted at iter node %d has no feasible schedule at any tile >= 2 under the kernel-wide budgets (SMEM floor %d + candidate / FRAG floor %d + candidate vs MaxCache %d / MaxFragPerThread %d) — tiling is never disabled (2026-10-24 ruling); widen the budgets or the TileSizes set`

Plus the F4 halving floor amendment (`lower_cache.go`): `resolveBudget` never
writes an override below 2 — a picked site whose halved covered dim would
fall below 2 is DROPPED to GMEM (never demoted to a non-tiling schedule).
The `anyTileOne` sweep stays as belt-and-braces (can no longer fire). R1:
non-window sequential iters take the swept uniform tile (`evalSchedule`).
R4: the judge rejects `TileSize < 2` input with `ReasonUnknownIter`.

## Test retarget (Task 2, commit ab6bb9c)

- Rejection pins at all four gates: `TestParamsValidate` (extended + ruling
  message pin), `TestVerifyFlagsCorruption/tile_one_decision`,
  `TestRewriteTileOneRejected` (gate-3 on a Verified hand-stamped record),
  `TestSearcherNeverDisabledError` (gate 4; the no-feasible corner is
  synthesized with floors beyond the budgets — same reachability the deleted
  fallback had, documented in the test).
- Deleted with the single form: `TestRewriteTileOne`,
  `TestRewriteWarpSingleMirrorGate`, `TestReducerWarpSingleConsumerRefusal`.
- New shape pins: `TestRewriteMinimalTile` (floor-tile nest: four loops at
  {CTA,100,2}/{SEQ,1,2}), `TestReducerBracketMinTile` (one-between R-A law
  at {CTA,100,2}/{WARP,8,2}).
- Re-pinned for the F4 floor-2 amendment: `TestLowerDefaultShapeDecline`,
  `TestBudgetFallback` (the shared-K drag-to-1 drop of the lhs FRAG site is
  gone — it survives at the floor tile 2, 8 B), `TestProvenancePublished`,
  `TestSearchWindow`, `TestPromotionJudge`, `enumerateBest`,
  `TestLowerFailClosed`, all finallower/emit/e2e fixtures.
- R6: fixture identifiers keep their names (`redt1`, `ReduceT1`,
  `reducet1_e2e`, `reduceT1Graph`, `TestReduceT1E2E`); comments rewritten to
  the minimal-legal-tile endpoint.

## Measured gate results (Task 3)

| Gate | Result |
| --- | --- |
| `go build ./...` + `go vet ./...` | green |
| `go test ./...` (fresh, `go clean -testcache`) | all packages ok |
| Goldens | `internal/cg/cxx/testdata/` git-clean (byte-identical); `TestGoldenMatmulStore`, `TestGoldenReduceWarp`, `TestReduceShapesEmit` pass; both goldens also compile as nvcc conformance objects in the runtime build |
| CPU runtime unit layer (`cmake -DTCACOMP_RUNTIME_TESTS=ON`, `tcacomp_cpu_tests`) | 2029 checks, 0 failures |
| `TCACOMP_E2E=1 go test ./internal/e2e/ -v` on the A6000 | all 7 tests pass: matmul (5.0s), reducewarp, matmul sanitizer (11.9s), reducewarp sanitizer, reduce3, **reducet1 retargeted to {CTA,100,2}/{WARP,8,2} (3.3s, canary 0)**, reduce3+reducet1 sanitizers (8.8s) |
| memcheck + racecheck | clean on reduce3 AND reducet1 fixtures (both tools), plus matmul/reducewarp |
| Final straggler grep (`TileSize: 1`, `tile == 1`, `ReasonTileOne`, `singleIV`, `warpSingleIVs`, `refuseThreadPartitionedConsumers`, T==1/single-form language over `.go` + normative `.md`) | clean — the only remaining hits are the intentional rejection pins and historical records (doc/review/*.md per R5, .planning artifacts) |

## Deviations from plan

1. **[Rule 2 - Missing critical functionality, test placement]**: the plan
   asked for a new standalone Validate pin test; an existing
   `TestParamsValidate` (in `overlap_test.go`) already covered the rejection
   table, so the tile-1 case + ruling-message pin were folded into it
   instead of duplicating a second test (the project's "only necessary
   tests" constraint). Same coverage, one test.
2. **[Rule 1 - Bug (stale expectations), behavioral]**:
   `TestLowerDefaultShapeDecline` and two `TestBudgetFallback` cases pinned
   "both sites drop to GMEM" under tiny MaxCache, driven by the old
   shared-K drag to tile 1. Under the ruling's F4 floor-2 amendment the K
   iter floors at 2, so the lhs FRAG site (8 B, inside MaxFragPerThread)
   legitimately STAYS PROMOTED while the rhs SMEM site drops. The tests
   were re-pinned to the amended behavior (rhs dropped, lhs promoted at
   the floor tile) — a direct consequence of the ruling R2, not an
   improvisation.
3. **[Rule 1 - Bug (stale shape knowledge)]**: `TestRewriteMinimalTile`
   initially pinned between-IV step = tile for the CTA dim; the canonical
   CTA between walk steps by LoC·T (200) and the coupled inside walk steps
   by coupledThreads (256) — assertions corrected to the canonical forms.

No architectural deviations. No auth gates.

## Known Stubs

None. The plan's goal is achieved end-to-end: no tile-1 path exists, four
gates reject it with named errors, the compiler's flat endpoint
{CTA,100,2}/{WARP,8,2} runs correctly on the A6000 with sanitizers clean.

## Commits

| Hash | Task | Content |
| --- | --- | --- |
| `0079a6a` | T1 | feat(quick-261024-ger): remove the tile-disable path (13 files, +253/-332) |
| `ab6bb9c` | T2 | test(quick-261024-ger): retarget the test suite (7 files, +323/-287) |

## Self-Check: PASSED

- `git log --oneline f2f4976..HEAD` shows both commits; `git status` clean
  (SUMMARY.md intentionally uncommitted per constraints).
- All seven measured gates above re-verified after the final commit.
