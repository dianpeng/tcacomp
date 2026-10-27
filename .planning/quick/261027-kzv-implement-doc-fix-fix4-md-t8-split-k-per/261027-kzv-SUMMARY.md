---
phase: quick-261027-kzv
plan: 01
subsystem: compiler
tags: [split-k, warp-reducer, c1b, tiling, codegen, cuda, e2e, golden]

# Dependency graph
requires:
  - phase: quick-261027-iq6
    provides: "fix4 T1–T7 factorization refactor (Wave 1, ReducedDimWarp=false), the C1b chunk-local ACCESS fix, and the A6000 e2e harness"
provides:
  - "ReducedDimWarp=true as the Wave-2 default — fix4 §3.5 split-k live end-to-end (decide → tile-lower → finallower bracket → WarpReducer emission)"
  - "Dedicated split-k matmul e2e proven numerically on the A6000 (both syntax variants, canary 0, maxAbs 4.196e-05 ≤ 2e-3; memcheck + racecheck zero errors)"
  - "Committed split-k emission golden testdata/reduce_splitk.cu + the §3.5 schedule pin in fePinScheduleSignature"
  - "The Wave-1 defense re-targeted as the explicit-false opt-out, fail-closed and pinned"
  - "Mechanical fix: the C1b FRAG fill window base now carries the lane chunk origin (roundBase + lane·ceil(T_d/32)) — GPU-proven defect, RED→GREEN pinned"
affects: [fix4, split-k, iq6-design-gap, warp-reducer, cache-lowering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "C1b lane-chunk arithmetic has TWO sides: the chunk-LOCAL access index (insideIV − lane·c) and the per-thread WINDOW base ((rank·T + round) + lane·c) — either side alone silently mis-numerics; only a GPU run exercises both"
    - "ReducedDimWarp=false is now the explicit Wave-1 policy opt-out (prune + fail-closed backstop), not the default"

key-files:
  created:
    - internal/cg/cxx/testdata/reduce_splitk.cu
  modified:
    - internal/opt/tiling/record.go
    - internal/opt/tiling/tiling.go
    - internal/opt/tiling/search.go
    - internal/opt/tiling/lower.go
    - internal/opt/tiling/lower_rewrite.go
    - internal/opt/tiling/lower_cache.go
    - internal/opt/tiling/overlap_test.go
    - internal/opt/tiling/lower_test.go
    - internal/opt/tiling/tiling_test.go
    - internal/cg/cxx/emit_test.go
    - internal/e2e/frontend_test.go
    - internal/cg/cxx/testdata/matmul_store.cu
    - internal/opt/tiling/testdata/tile_lowered_matmul.ir
    - doc/fix/fix4.md

key-decisions:
  - "Flipped DefaultParams.ReducedDimWarp to true (Wave 2, fix4 §3.5) with exactly the planner-probe blast radius: 4 tests re-pinned, nothing else"
  - "The T8b GPU failure (maxAbs 28.99) was fixed mechanically at the C1b fill-window layer (chunkBase = roundBase + lane·c) — not a redesign; the lowers-translate-verbs-only ruling held"
  - "No iq6 candidate rulings 1/2/3 adopted; both matmulGoldenDesignGapSkip skip sites remain; rejectWarpReducedDims and reducedIters code byte-identical (comment-only elsewhere)"

patterns-established:
  - "Split-k policy pins: TestReducedDimWarpPruning pins the prune under explicit false AND the deterministic split-k winner (cta/4/256, seq/1/256, warp/2/32; lhs FRAG 4 B, rhs SMEM 65536 B, reuse 256) under the default"
  - "fePinScheduleSignature pins the §3.5 emission class: __launch_bounds__(64), dim3(64)/dim3(64)/65544u/512u, WarpReducer + staging row at tca_smem+65536 extent 2u, ThrIterator N rounds stride K_d·T_d = 64"

requirements-completed: [fix4-T8]

# Metrics
duration: 45min
completed: 2026-10-27
---

# Phase quick-261027-kzv: fix4 §3.5 split-k enablement (Wave 2 T8) Summary

**fix4 §3.5 split-k is live under the Wave-2 default: the frontend matmul selects M=CTA × N(reduced)=WARP/2 × Q=SEQ, lowers through the WarpReducer bracket, and proves numerically correct on the A6000 (canary 0, maxAbs 4.196e-05, sanitizers clean) after a mechanical C1b fill-window fix the GPU battery exposed.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-10-27T07:15:00Z
- **Completed:** 2026-10-27T07:58:00Z
- **Tasks:** 3 (T8a hermetic flip + re-pins + golden; T8b A6000 GPU battery + defect protocol; T8c doc + summary)
- **Files modified:** 14 (5 source/comment, 5 tests, 3 goldens, 1 doc)

## Accomplishments

- **The flip:** `DefaultParams.ReducedDimWarp` false→true. Measured blast radius exactly matched the planner probe: 4 failing tests (`TestLowerWarpReducedDimRejected`, `TestDefaultParams`, `TestReducedDimWarpPruning`, `TestFrontendMatmulScalePipeline/written`), nothing else in `go test ./...`.
- **Split-k live end-to-end:** the real frontend matmul_scale (feM=64, feN=128, feQ=64, runtime extents) selects M=CTA(T=256,LoC=64) × N(reduced)=WARP(K=2,T=32) × Q=SEQ(T=256) and emits the complete §3.5 kernel: `__launch_bounds__(64)`, grid/block dim3(64), dynamic SMEM pool 65544 B = 65536 B rhs window at the MaxCache cap + 8 B reducer staging at `tca_smem + 65536u` (extent 2u = #warps), `WarpReducer<float> reducer0(...)` with per-element `[Init → N tile-between ThrIterator rounds (stride K_d·T_d = 64, lane-chunk inside walks) → Reduce(value) → Combine()]`, uniform M/Q element walks, single-owner store gated on `is_first_in_cta()`.
- **GPU battery green (A6000, TCACOMP_E2E=1):** frontend matmul BOTH syntax variants canary 0, max abs diff **4.1961669921875e-05** (tolerance 2e-3); reduce regression fixtures (warp/3/t1) canary 0, maxAbs 0; memcheck + racecheck **zero errors** on every runnable fixture (the split-k kernel's cross-warp staging fold and warp-varying LoadTileCta window bases are exactly what racecheck arbitrates).
- **The Wave-1 defense survives as the explicit-false opt-out:** `TestLowerWarpReducedDimRejected` pins fail-closed under explicit `ReducedDimWarp=false` AND that `DefaultParams()` no longer trips it.
- **New committed golden:** `testdata/reduce_splitk.cu` — the crafted-record {CTA,100,32}/{WARP,2,32} split-k emission, byte-regenerated, ScanBracketLaw-clean.
- **GPU-proven defect fixed mechanically (the C1b fill window):** see Deviations.

## Task Commits

Each task was committed atomically:

1. **Task 1 T8a: flip + re-pins** — `f25d53b` (feat: the default flip, comment-only elsewhere, the four policy re-pins)
2. **Task 1 T8a: split-k golden** — `c860b68` (test: reduceSplitKGraph + TestGoldenReduceSplitK + testdata/reduce_splitk.cu)
3. **Task 2 T8b defect protocol: RED pin** — `feb290e` (test: TestLowerC1bFragChunkOriginWindow, failing)
4. **Task 2 T8b defect protocol: GREEN fix** — `cd70fa2` (fix: chunkBase in lower_rewrite.go + class-aware coveredBaseSubs in lower_cache.go + 2 regen goldens)
5. **Task 2 T8b: dedication + numeric log** — `cbee07d` (test: split-k dedication comment, canary/maxAbs success log)
6. **Task 3 T8c: fix4.md** — (docs: §staging + §2.1 row to the landed Wave-2 default; orchestrator commits with the docs set)

## Files Created/Modified

- `internal/opt/tiling/record.go` — the flip (`ReducedDimWarp: true`) + truthful field/DefaultParams comments
- `internal/opt/tiling/search.go`, `tiling.go`, `lower.go` — comment-only truth updates (code byte-identical; fence)
- `internal/opt/tiling/lower_rewrite.go` — **the fix:** `dimRewrite.chunkBase` = roundBase + lane·ceil(T_d/32), materialized in the WARP case of generateDimExprs
- `internal/opt/tiling/lower_cache.go` — **the fix:** `coveredBaseSubs(covered, class)` substitutes chunkBase for a FRAG-class WARP-covered dim; call site + comment truth
- `internal/opt/tiling/overlap_test.go` — TestDefaultParams inverted to default-true
- `internal/opt/tiling/lower_test.go` — TestLowerWarpReducedDimRejected re-targeted (explicit-false negative + default-admits positive control); NEW TestLowerC1bFragChunkOriginWindow (the RED→GREEN pin)
- `internal/opt/tiling/tiling_test.go` — TestReducedDimWarpPruning: prune under explicit false + the deterministic split-k winner pin (iters cta/4/256, seq/1/256, warp/2/32; lhs FRAG 4 B reuse 256; rhs SMEM 65536 B reuse 256)
- `internal/e2e/frontend_test.go` — fePinScheduleSignature re-pinned to the §3.5 split-k class (6 pins); split-k dedication comment; canary/maxAbs success log
- `internal/cg/cxx/emit_test.go` — reduceSplitKGraph + TestGoldenReduceSplitK
- `internal/cg/cxx/testdata/reduce_splitk.cu` — NEW split-k emission golden
- `internal/cg/cxx/testdata/matmul_store.cu`, `internal/opt/tiling/testdata/tile_lowered_matmul.ir` — regen: the inserted lane-term statement (the fix's only golden blast radius)
- `doc/fix/fix4.md` — §staging bullet + §2.1 `ReducedDimWarp` row to the landed Wave-2 default (§3.6 alias corrected to §3.5)

## Decisions Made

- **Trusted the run over the pin wherever they could differ** (the plan's re-derive-first rule): the probe's split-k winner values were re-measured locally and matched; every re-pinned literal was derived from freshly emitted bytes.
- **The T8b numerical failure was a mechanical C1b window bug, not the §3.1 design-gap class** — the flipped selection's only thread-partitioned walk is the REDUCED dim's lane chunks; surviving M/Q walks are uniform per-thread element walks. Fixed at the smallest layer per the iq6 defect protocol; no ruling sought, no fence touched.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The C1b FRAG fill window base omitted the lane chunk origin — GPU-proven numerical defect**
- **Found during:** Task 2 (T8b A6000 numerical battery): `TestFrontendMatmulScale` failed BOTH syntax variants, max abs diff = 28.992233276367188 > tolerance 0.002 at flat 1546 (got 89.014, want 60.022).
- **Issue:** the per-thread `LoadTileThread` fill of a C1b forced-FRAG site over a WARP-covered dim used the bare warp window origin `rank·T_d + round` as the GMEM window base. Every lane of a warp loaded the warp's FIRST element while its lane-chunk inside walk consumed a DISTINCT element — 32 lanes read the same GMEM cell per round; the products summed the wrong lhs column. Latent since the C1b ruling: the iq6 fix pinned only the chunk-LOCAL ACCESS index (`insideIV − lane·c` — the fragment side); the WINDOW side of the same chunk arithmetic was structurally pinned but never numerically exercised until the Wave-2 flip (all prior WARP-bearing fixtures were zero-warp or skipped by the design gap).
- **Fix:** mechanical, verb-translation only: `generateDimExprs` now also materializes `d.chunkBase = roundBase + lane·ceil(T_d/32)` (the lane's chunk origin), and `coveredBaseSubs` substitutes it for a FRAG-class WARP-covered dim (SMEM cooperative windows unchanged — CTA-uniform by R1). The existing chunk-local access then indexes exactly the elements the window filled.
- **Files modified:** `internal/opt/tiling/lower_rewrite.go`, `internal/opt/tiling/lower_cache.go` (+ regen `matmul_store.cu`, `tile_lowered_matmul.ir`)
- **Verification:** RED pin `TestLowerC1bFragChunkOriginWindow` (feb290e) failed pre-fix, GREEN post-fix; full hermetic suite green; A6000 battery re-run green (canary 0, maxAbs 4.196e-05 both variants; memcheck + racecheck zero errors).
- **Committed in:** `feb290e` (RED), `cd70fa2` (GREEN + regen)

**2. [Rule 2 - Missing Critical] Success-path numeric log for the dedicated split-k e2e**
- **Found during:** Task 2 (battery record): the test asserted canary/maxAbs but printed values only on failure, leaving no machine-parseable success record (threat T-kzv-02 wants present-and-zero/present-and-within lines).
- **Fix:** one `t.Logf` with canary, maxAbs, tolerance per variant.
- **Files modified:** `internal/e2e/frontend_test.go`
- **Committed in:** `cbee07d`

---

**Total deviations:** 2 (1 bug, 1 observability)
**Impact on plan:** both within the plan's own defect protocol (Task 2 step 6 explicitly anticipates a mechanical split-k defect with a RED→GREEN pin); no scope creep, no fence touched.

## The Interaction Record (for the author)

1. **Split-k is live end-to-end.** With the flag open, the decide side selects reduced-WARP vectors and the pipeline lowers them per §3.5 — decide → tile-lower → finallower bracket → WarpReducer emission, proven numerically through the real frontend pipeline on the A6000.
2. **The iq6 case-b reach broadens.** The coarse `reducedIters` set (ALL dims referenced by a reduce's piped value, `search.go:357`) previously shielded reduce-carrying kernels from ANY WARP selection — post-flip, SURVIVING-WARP vectors are also enumerable for reduce kernels, so the unruled iq6 case-b silent-wrong class (CTA×WARP(surviving) uncovered cross product) is unchanged in nature but **broader in reach**. Candidates 1/2/3 remain open; the golden-matmul skips stay.
3. **Fences kept:** `rejectWarpReducedDims` (`lower.go:270-303`) and `reducedIters` (`search.go:357-372`) code byte-identical (comment-only elsewhere); no surviving-WARP gate added; `grep -c 't.Skip(matmulGoldenDesignGapSkip)' internal/e2e/e2e_test.go` == 2.
4. **Bonus finding for the record:** the T8b battery exposed that the C1b lane-chunk arithmetic had been half-wired (access side only) since iq6 — the WINDOW side is now pinned too (`TestLowerC1bFragChunkOriginWindow`). Any future WARP-dim work should treat both sides of the chunk arithmetic as one invariant.

## Issues Encountered

- The sanitizer battery is slow on the split-k kernel (racecheck ~72 s under compute-sanitizer) — expected, within the 60 m timeout.
- First cold e2e build takes ~15 s of CMake/nvcc (content-addressed fixture cache); warm runs are sub-second.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- fix4 Wave 2 (T8) is the last staged item of the refactor: the §8 task list is now fully landed.
- Open for the author (NOT touched here): the iq6 surviving-WARP design gap ruling (candidates 1/2/3) — now broader in reach per the interaction record; the golden-matmul e2e stays skipped until then.

## Self-Check: PASSED

- `internal/cg/cxx/testdata/reduce_splitk.cu` — FOUND (committed c860b68, byte-regenerating)
- Commits f25d53b, c860b68, feb290e, cd70fa2, cbee07d — FOUND (git log)
- Fence audit — PASSED (2 skip sites; search.go/lower.go diffs comment-only)
- Hermetic gate — PASSED (`go build ./... && go vet ./... && go test ./...` green; `go list -m all | wc -l` == 1)
- GPU battery — PASSED (A6000 present; numerical + sanitizer logs in /tmp/t8b-numerical2.log, /tmp/t8b-sanitizer.log, /tmp/t8b-e2e.log)

---
*Phase: quick-261027-kzv*
*Completed: 2026-10-27*
