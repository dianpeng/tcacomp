---
phase: quick-261029-qwo
plan: 01
subsystem: opt/tiling + opt/finallower
tags: [tiling, scheduling, adaptive-loc, reducer-collapse, tier-efficiency, gpu-validation]

requires:
  - phase: quick-261029-n63
    provides: the occupancy-aware objective (score' = traffic × occupancy factor) and the 14-kernel testing suite the before/after benchmark rides
provides:
  - adaptiveLoC in internal/opt/tiling/constraints.go — the single shared source of the R1 extent-derived adaptive top digit, consumed by the search's digitListOf AND tiling-verify's locPresetOK (tile + effective-extent signature)
  - the R1 menu widening — one adaptive digit per (dim, SoC, tile), deduped (presets win), so a full-grid candidate exists at every tile size
  - reducedTierEff in internal/opt/tiling/judge.go — the R4 tier-efficiency guard, folded into the float64 score' currency only (the int traffic pin survives raw)
  - the R2 one-tier (SEQ) ladder verification — RuntimeSequentialReducer 2-operand face, zero WarpReducer/ThreadMapping/staging ceremony, runtime/ untouched
  - GPU-verified before/after benchmark — family-B grid 64→128 re-choice via the adaptive digit, zero kernels >5% slower
affects: [tiling-search, tiling-verify, codegen-schedule-selection, fix5-t2-r3, fix6-work-shape-model]

tech-stack:
  added: []
  patterns:
    - "adaptiveLoC in constraints.go is the single source of the R1 formula; digitListOf enumerates it and locPresetOK re-checks it — neither re-derives, and both resolve the SAME effective extent (constExtentOf with the NominalExtent stand-in)"
    - "R4 is selection-only exactly like the n63 occupancy factor: reducedTierEff scales the float64 score' accumulation ONLY; res.traffic (int), the judge, the eviction, and tiling-verify stay R4-blind"
    - "Re-derived-pin discipline (n63 precedent): TestStructuralInfeasibility's nil-presets premise became impossible under R1 (the adaptive digit always exists) — the pin now asserts the R1 menu completeness plus the surviving digit-less degenerate case; fePinScheduleSignature re-derived from the fresh .cu (grid 64→128)"

key-files:
  created:
    - internal/opt/tiling/adaptive_test.go
    - .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md
    - .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/before-bench.txt
  modified:
    - internal/opt/tiling/constraints.go
    - internal/opt/tiling/search.go
    - internal/opt/tiling/judge.go
    - internal/opt/tiling/verify.go
    - internal/opt/tiling/record.go
    - internal/opt/tiling/tiling_test.go
    - internal/opt/finallower/finallower_test.go
    - internal/e2e/frontend_test.go

key-decisions:
  - "Effective-extent reading of R1 (plan-sanctioned fallback): every R1/R4 formula's extent is the dim's const extent when known, else Params.NominalExtent — REQUIRED for search/verify lockstep on const-extent dims (const 96/tile 32 yields adaptive 3, which the bare-NominalExtent formula would reject at verify time); recorded in the plan's Task-2 action 2 and pinned by TestLocPresetOKAdaptive"
  - "R4 folds into the score' product only, bit-identical elsewhere: promoted sites covering no reduced warp-sharded dim carry tierEff exactly 1 (float64(traffic) == scaledTraffic for those vectors)"
  - "The plan's group-5 pin component 'CTA LoC beyond the old preset top of 64 in the winner' is unattainable on ANY scale_copy-shaped LNT: adaptive digits > 64 exist only at T_r <= 64, where the reuse term ceil(T_r/blockDim) with blockDim >= 64 collapses to 1 (no traffic), and the blockDim-32 full-collapse declines its FRAG sites (4096 B > 256 B MaxFragPerThread) — re-derived the pin to the achievable collapse property (no WARP digit on the reduced dim) with the exact winner vector in the comment"
  - "R2 needs no compiler change: the existing RuntimeSequentialReducer 2-operand face (init, lambda) IS the spec's plain per-thread accumulator; the one-tier ladder verified by the new TestReducerOneTierSeqLadder pin; runtime/ untouched (zero diff)"
  - "Gate-3 failure handling per the plan's stop-rule: no pin was weakened and no out-of-scope fix (R3/fix6/MaxCTA/runtime) was attempted; the failure is recorded here with the .cu-diff attribution and the mechanism diagnosis"

requirements-completed: [QUICK-261029-QWO]

duration: 41min
completed: 2026-10-29
status: complete-with-gate-3-failure
---

# Quick 261029-qwo: fix5 T1 — R1 Adaptive LoC Ceiling + R2 Degenerate-Reducer Collapse + R4 Tier-Efficiency Guard

**R1+R4+R2 landed exactly as spec'd (adaptiveLoC single source, digitListOf adaptive digits, reducedTierEff in score' only, one-tier SeqReducer verified) with the GPU suite re-validated 14/14 and memcheck clean on every schedule-changed kernel — and the elementwise ≥10× benchmark gate FAILED: the four elementwise kernels' emitted `.cu` files are bit-identical before/after, because R4 is a no-op on their dynamic extents and the fix4 FRAG cost model declines the full-collapse vector. No kernel regressed beyond +1.3% (gemm, inside its own before-run min/max band).**

## What Was Built

- **`adaptiveLoC`** (`internal/opt/tiling/constraints.go`): the single shared
  source of the R1 formula `clamp(ceil(extent/T), 2, MaxCTA)` (WARP dims:
  `MaxWarpsPerCTA` cap). Every dim's digit list gains ONE extent-derived
  adaptive top digit per SoC per tile — a full-grid candidate now exists at
  every tile size (fix5 §0 defect 2), so the n63 occupancy `fill` term is
  reachable.
- **`digitListOf`** (`search.go`): appends the adaptive digit per tile after
  each SoC section's preset digits (tiles ascending), deduped — a value
  already present (preset, or an earlier tile's adaptive value) appears
  exactly once; presets win. Respects R-LOCCAP (the adaptive value derives
  from the effective extent, so it needs no cap of its own) and R-REDUCED
  (no WARP section at all when `ReducedDimWarp=false`).
- **`locPresetOK`** (constraints.go): signature grown to `(soc, loc, tile,
  extent, Params)`; acceptance is preset membership OR equality with the
  shared `adaptiveLoC`. Both call sites (tiling-verify + the tiling tests)
  resolve the effective extent via `constExtentOf` with the `NominalExtent`
  stand-in — the same discipline the search enumerates with, so a
  search-chosen adaptive digit on a const-extent dim can never fail the
  verify re-check.
- **`reducedTierEff`** (`judge.go`): the R4 guard — per covered dim that is
  reduced AND WARP-assigned, `min(1, extent_k/(32·K_k))` multiplies the
  product; everything else contributes exactly 1. `evalSchedule` folds it
  into a float64 `scaledTraffic` alongside the **unchanged raw int
  `res.traffic` pin**; `score' = scaledTraffic × occupancyOf(...).factor`.
  Selection-only — eviction, judge, enumeration, and tiling-verify are
  R4-blind (the n63 occupancy precedent).
- **`record.go`**: `NominalExtent`'s doc now names its R1 + R4 roles (no new
  Params fields; MaxCTA stays 512 per the §6 ruling).
- **R2 verification** (`finallower_test.go`, new
  `TestReducerOneTierSeqLadder`): the one-tier (SEQ) ladder lowers to the
  existing `RuntimeSequentialReducer` 2-operand face (init, lambda) —
  exactly one SEQ construction, zero WarpReducer/GlobalReducer, no
  ThreadMapping operand, no SMEM staging row, the single-owner store
  preserved. **runtime/ has zero diff** (fix5 §3).
- **`adaptive_test.go`** (new, ~380 lines): the five RED→GREEN pin groups —
  R1 formula values (floor/cap/const paths), digitListOf emission + dedup +
  placement, locPresetOK acceptance/rejection (incl. the const-96/tile-32
  lockstep case), reducedTierEff values (0.0625/1, non-reduced cover,
  stand-in, product), and the search-level re-choice with the
  raw-traffic-never-scaled pin.

## GPU Re-Validation (A6000, testing/orchestrator.py)

**Correctness (gate 2):** full suite on the rebuilt `bin/tcacomp` — **14
passed, 0 failed** (fresh builds, canary 0 everywhere, maxdiff within every
suite.toml tolerance). Memcheck (`--sanitizer memcheck`) clean on ALL SIX
schedule-changed kernels — `mm_small` (verbose run confirms the
`compute-sanitizer --tool memcheck` argv), `mm_mid`, `mm_splitk`, `gemm`,
`qk_score`, `mlp`. The eight unchanged kernels ride bit-identical `.cu`
(same content-addressed fixture hash), covered by the prior runs.

**Benchmark (gate 3, --bench-iters 20, avg µs/iter):**

| kernel | BEFORE (179dc8f) | AFTER (6e77455) | Δ | schedule change |
|--------|------------------|-----------------|---|-----------------|
| mm_small | 2882.2 | 2882.6 | +0.01% (noise) | B — grid 64→128 |
| mm_mid | 6494.4 | 6494.7 | +0.005% (noise) | B — grid 64→128 |
| mm_splitk | 6099.6 | 6100.8 | +0.02% (noise) | B — grid 64→128 |
| gemm | 9819.2 | 9945.9 | +1.29% (in-band) | B — grid 64→128 |
| qk_score | 17409.5 | 17415.9 | +0.04% (noise) | B — grid 64→128 |
| mlp | 6295.4 | 6275.7 | −0.31% (noise) | B — grid 64→128 |
| scale_copy | 2345760 | 2350350 | +0.20% (noise) | none — bit-identical .cu |
| axpy3 | 2376770 | 2370900 | −0.25% (noise) | none — bit-identical .cu |
| dup_expr | 1168350 | 1173330 | +0.43% (noise) | none — bit-identical .cu |
| dup_const | 1183560 | 1187640 | +0.34% (noise) | none — bit-identical .cu |
| softmax | 59596.2 | 59605.8 | +0.02% (noise) | none |
| transpose | 330317 | 330313 | ~0% (noise) | none |
| row_sum | 464614 | 464597 | ~0% (noise) | none |
| layernorm | 487413 | 487675 | +0.05% (noise) | none |

gemm's +1.29% sits inside its own BEFORE run's min/max band (9761.9–9946.1
µs) and mm_small carried the identical grid change with +0.01% — the
n63 protocol's ±1.2% noise band applies; **no kernel is >5% slower (the
regression gate PASSES)**.

**GATE-3 HARD GATE — FAILED:** dup_expr, dup_const, axpy3, scale_copy each
required ≥10× faster; all four are within ±0.5% of the before state. Per
the plan's stop-rule no pin was weakened and no out-of-scope change was
attempted; the failure is recorded here with the attribution evidence.

## Gate-3 Failure Diagnosis (the .cu-diff attribution)

The four elementwise `.cu` files are **bit-identical** before/after
(same content-addressed fixture hashes) — the search re-selected the fix5
§0 ceremony vector `R=CTA(T=256,LoC=16) C=SEQ(T=32) K=WARP(T=32,LoC=2)`,
16 CTAs × 64 threads, WarpReducer bracket. Two mechanisms, both verified:

1. **R4 is a no-op on dynamic extents.** The suite kernels are compiled
   without shape constants (`uint32_t K` is a runtime parameter — the
   compiler never sees K=4), so `reducedTierEff` falls back to the
   `NominalExtent` stand-in: `min(1, 4096/(32·2)) = 1`. The ceremony
   vector's traffic credit is uncut. (Pinned in `TestTierEffReduced` —
   the stand-in fallback is the spec'd behavior, group 4 of the plan.)
2. **The fix4 FRAG model declines every full-collapse vector.** With
   `K=SEQ` AND `C=SEQ` (blockDim 32 — the fix5 §4 target shape), the loads'
   covered set `[C,K]` prices a per-thread FRAG window of
   `4 B × T_c × T_k = 4096 B > MaxFragPerThread 256 B` → both sites
   declined → traffic 0 → score' 0 → cannot beat the ceremony vector's
   ~7705 under any tierEff ≥ 0. And the R1 adaptive CTA digits (> 64 at
   `T_r ≤ 64`) all zero the reuse term `ceil(T_r/blockDim)` at
   blockDim ≥ 64. Hence no legal vector carrying an adaptive >64 digit can
   win on this LNT — the plan's group-5 "LoC beyond the old preset top of
   64" component is unattainable on any scale_copy-shaped fixture (re-
   derived pin, derivation in the test comment).

**Proof the R1+R4 machinery itself works** (unit level, const-extent
fixture): the `TestSearchRechoiceCollapse` LNT (R/C dynamic + K const 4 —
R4 bites) re-chooses exactly as designed: the collapse vector
`R=CTA(16,T256) C=WARP(2,T32) K=SEQ(T32)` scores **7704.9** vs the
ceremony vector's **481.5** (`49152 × 0.0625 × 0.1508`) — a 16× swing from
R4 alone, with both loads still promoted FRAG 128 B reuse 4 (traffic
unscaled). The one-tier ladder then lowers through the verified
SeqReducer path. What is missing for the REAL kernels is a compile-time
K extent — i.e. either frontend const-shape specialization or the fix6
work-shape model (FRAG windows > 1×1 that price the collapse honestly),
**both explicitly out of scope for T1**.

**Family-B re-choice (the R1 win that did land):** all six split-K kernels
re-derived the frontend-fixture pattern — `M=CTA` grid 64→**128** at T=32
(the adaptive digit `ceil(4096/32)`), fill saturating to 1.0 (capacity 84
at one 1024-thread CTA/SM) vs 0.7976 at equal traffic — with the split-K
structure (WarpReducer, staging row, Frag windows, tail-clamped
CtaIterator) byte-identical otherwise. Timing-neutral (these are
bracket-bound, not grid-bound), correctness re-verified by the suite and
memcheck.

## Gate Verdicts (fix5 §5)

1. **go build/vet/test exit 0** — PASS (25/25 packages, run twice).
2. **GPU suite 14/14, memcheck clean on every schedule-changed kernel** —
   PASS (6 kernels memchecked; verbose run confirms the sanitizer argv).
3. **Benchmark gate** — **FAILED** the elementwise ≥10× hard requirement
   (all four bit-identical .cu, ±0.5% noise); the no-kernel->5%-slower
   regression requirement PASSES (worst +1.29%, in-band). Recorded
   prominently per the plan's stop-rule with the mechanism diagnosis above.
4. **tiling-verify green** — PASS, implied by the suite: tile-lower refuses
   an unstamped record, and every suite kernel compiled and ran; the
   search-chosen adaptive records re-check through the grown locPresetOK
   (same formula, same effective extent). GlobalReducer/D1/D2 untouched;
   runtime/ has zero diff.

## Deviations from Plan

1. **[Plan/Task-1 group 5 — pin component re-derived, not weakened]**
   "a CTA LoC beyond the old preset top of 64" is unattainable on any
   scale_copy-shaped LNT (mechanism 2 above — adaptive digits > 64 exist
   only where the reuse arithmetic zeroes their vector's traffic, and the
   blockDim-32 collapse declines its sites). The pin re-derives to the
   achievable collapse property — NO WARP digit on the reduced dim, both
   loads promoted, exact winner vector in the comment — per the plan's own
   "re-derive from the actual search output, never weaken" rule. The
   re-derived pin still fails on a pre-fix5 binary (the winner keeps
   K=WARP), so the RED→GREEN discipline is intact.
2. **[Rule — spec-level gap recorded, not fixed] Gate 3 fails** on the
   elementwise family for the two mechanisms above; the fix requires
   const-shape frontend specialization or the fix6 work-shape model — both
   locked out of T1's scope. Per the plan's stop-rule: no pin weakened, no
   out-of-scope change, failure recorded with .cu-diff attribution.
3. **[TestStructuralInfeasibility re-derived]** its nil-CtaLoCPresets
   premise became impossible under R1 (the adaptive digit always exists —
   menu completeness is R1's purpose). The pin now asserts that R1 behavior
   (search succeeds, CTA dims carry adaptive LoCs accepted by the shared
   locPresetOK) plus the surviving hard error in the fully digit-less
   degenerate case (nil presets AND nil tiles). The fix4 §2.6 error path
   itself is unchanged.
4. **[fePinScheduleSignature re-derived (n63 precedent)]** the frontend
   fixture re-chose grid 64→128 (the R1 adaptive digit at T=32; fill
   saturation factor 1.0 vs 0.7976 at equal traffic 8355840 — derivation in
   the pin comment); the split-k class (WarpReducer, staging 128u,
   ThrIterator 8192, Frag lhs window) is preserved.

## Commits

| Hash | Message |
|------|---------|
| e88f049 | test(quick-261029-qwo): add failing fix5-T1 pins (RED) |
| 6e77455 | feat(quick-261029-qwo): implement fix5 T1 R1+R2+R4 (GREEN) |

## Self-Check: PASSED

- internal/opt/tiling/adaptive_test.go — FOUND (created)
- internal/opt/tiling/constraints.go — modified (adaptiveLoC present, verified by tests)
- before-bench.txt — FOUND (14/14 PASS, 14-kernel BEFORE table)
- SUMMARY frontmatter status: complete-with-gate-3-failure
- go build/vet/test exit 0 (25/25 packages, run twice on the final tree)
- orchestrator full suite 14/14 on rebuilt bin/tcacomp; memcheck clean on all 6 schedule-changed kernels
- elementwise .cu bit-identical before/after (content-hash equality) — the gate-3 failure evidence
- BEFORE worktree removed after the benchmark (Task 3 cleanup)
