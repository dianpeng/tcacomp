---
phase: quick-261026-tq8
plan: 01
subsystem: compiler-opt
tags: [tiling, judge, one-currency-reuse, fix4, cost-model]

requires:
  - phase: quick-261026-ky1
    provides: fix4 T1 decide pass + T2 tiling-verify (judge.go/search.go/record.go baseline this patch amends)
provides:
  - fix4 §2.4 one-currency reuse amendment in the promotion judge (collective SMEM trips, holders-scaled FRAG traffic, one-currency promotion gate)
  - fix4 §2.5 objective in one CTA-wide currency (bytes × (reuse − 1) × holders)
  - NEW unit-consistency regression guard TestFix33OneCurrencyEquality (fix4 §9(2) acceptance clause)
affects: [quick-261026-ky1 goldens (none moved), fix4 T4 tile-lower (untouched — record fields unscaled), fix4 Wave-2 split-k]

tech-stack:
  added: []
  patterns: [class-branched siteReuseCount (SMEM collective vs FRAG per-thread trips), holders as the currency scale carried on siteJudgment]

key-files:
  created: []
  modified:
    - internal/opt/tiling/judge.go
    - internal/opt/tiling/search.go
    - internal/opt/tiling/record.go
    - internal/opt/tiling/tiling_test.go
    - internal/opt/tiling/lower_test.go
    - internal/opt/tiling/verify_test.go
    - internal/opt/schedule/pipeline_test.go

key-decisions:
  - "Reuse counted at the window's OWN granularity per class (fix4 §2.4 amendment): SMEM collective per window instance (CTA/WARP -> T_d, SEQ -> T_d × readers), FRAG per-thread trips unchanged; holders (= LNT blockDim for FRAG, 1 for SMEM) lives only in the objective — record Reuse stays unscaled per §2.7"
  - "Promotion gate is now the one-currency traffic test bytes × (reuse − 1) × holders > 0 — subsumes the old reuse < 2 gate and fails closed a zero-byte site; ReasonFragBytes unchanged and still before reuse"
  - "TestVerifyKernelWideFragGate pin STAYS at 32 B: the plan's round-2 flip derivation (32 -> 4 via a K=WARP winner) missed the §2.6 eviction collateral — see Deviations"

patterns-established:
  - "Never silent re-pin: a plan-pinned expectation that contradicts the full-spec hand derivation gets re-derived, pinned with the complete arithmetic in the test comment, and escalated in the SUMMARY (261026-ky1 precedent)"

requirements-completed: [FIX4-T1.1]

duration: 16min
completed: 2026-10-26
---

# Quick Task 261026-tq8: fix4 §8 T1.1 — one-currency objective patch Summary

**judge.go site reuse now branches on class (collective SMEM trips / holders-scaled FRAG trips) so every site scores in ONE CTA-wide GMEM-byte currency; §3.3 re-pin and the new equality guard hold exactly, with one plan-derivation error (eviction collateral) caught and pinned loudly.**

## Performance

- **Duration:** ~16 min
- **Started:** 2026-10-26T15:51:57Z
- **Completed:** 2026-10-26T16:08:00Z
- **Tasks:** 2/2
- **Files modified:** 7

## Commits

| Commit | Type | Content |
|--------|------|---------|
| 4c1ce27 | test (RED) | All six test updates failing against the unamended judge: TestFix33WorkedExample re-pin (rhs Reuse 4 → 256) + fix33Graph() extraction + resolved-state rationale; NEW TestFix33OneCurrencyEquality; TestPromotionJudge low-reuse re-target; one-currency mirrors in TestTrafficArithmeticAndOptimality + enumerateBest; TestDecideLowerComposition rhs pin; TestVerifyKernelWideFragGate re-pin per the plan's derivation |
| 78f3060 | feat (GREEN) | judge.go: siteJudgment.holders + class-branched siteReuseCount + one-currency gate; search.go: score × holders; record.go: comment sync + ReasonLowReuse text generalized (name stable); verify_test.go: corrected TestVerifyKernelWideFragGate pin (see Deviations 1) with full §2.4+§2.6 arithmetic |
| b7f02b6 | docs | internal/opt/schedule/pipeline_test.go comment sync: the 1024/1024/kExtent fixtures' searched record now promotes rhs as a 65536 B SMEM window (probed: M=cta/4/256, N=seq/1/256, K=warp/2/32 on kExt ∈ {1024, 32768}) — the stale "lhs SMEM / rhs declined / exactly one fill" text replaced; structural pins unchanged and green |

## Numeric audit trail (all hand-derived, all pinned in tests)

- §3.3 winner (M=CTA/16/256, N=SEQ/1/256, K=WARP/2/32, blockDim 64): rhs SMEM 65536 B, collective reuse 256 = T_M (old per-thread 4 × 64 readers = the same collective 256), sharing 1 (K ∈ rhs subscripts) → traffic 65536×255 = **16711680**; lhs FRAG 4 B, per-thread reuse 256, holders 64 → 4×255×64 = **65280**.
- Pre-amendment vector (M=CTA/16/256, N=WARP/4/64, K=SEQ/1/64, blockDim 128): rhs **EXACTLY equal** (65536 B, reuse 256, 16711680 — the unit-consistency invariant, pinned by TestFix33OneCurrencyEquality); lhs 256 B, reuse 2 (ceil(64/32)), holders 128 → 256×1×128 = **32768**; winner wins on the honest lhs fill-cost difference 65280 > 32768.
- matmul worked example (matmulUsed, M=1024): schedule total **16776960** = 16711680 + 65280; closest challengers (N=SEQ/1/128 + K=WARP/2/64 or K=WARP/4/32) at **16776704** (their lhs = 8×127×64 = 4×127×128 = 65024), margin 256 — the schedule pin does NOT move.
- TestVerifyKernelWideFragGate (TileSizes={8}, MaxCache=100): all-SEQ winner 32×7×32 = **7168**; K=WARP challenger 4×7×(K·32) = 896·K_d BEFORE eviction but **0 after** (see Deviations 1).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - plan-arithmetic error] TestVerifyKernelWideFragGate: the 32 B pin does NOT move — the plan's flip derivation missed the §2.6 eviction collateral**
- **Found during:** Task 1 GREEN (the RED pin per the plan's derivation failed: the fixture promoted 32 B, not 4)
- **Issue:** The plan's checker round-2 derivation (interfaces block + Test 6) expected the winner to flip to M=CTA/…/8, N=SEQ/1/8, K=WARP/Kmax/8 with lhs a 4 B lane-chunk FRAG (promotedFrag 32 → 4, tight budget 31 → 3). That derivation judged the K=WARP vector's lhs traffic (4×7×(K·32) = 896·K_d = 28672 at K_d=32) but treated the over-budget rhs (an 8×(K_d·8)×4 = 256·K_d B SMEM window > MaxCache 100) as harmlessly evicted. Eviction is part of the SCORE: density = reuse − 1 TIES at 7 (lhs per-thread 8, rhs collective T_M = 8), and the UNCHANGED §2.6 tie-break (smaller bytes first — explicitly out of scope for T1.1) drops the 4 B lhs BEFORE the 8192 B rhs — the K=WARP vector evacuates BOTH sites and scores 0. The all-SEQ-covered vector (M=CTA/64/8, N=SEQ/1/8, K=SEQ/1/8, blockDim 32) keeps its 32 B lhs FRAG at traffic 32×7×32 = 7168 > 0; traffic decides and the partitionCount tiebreak never fires (the plan's "ties at 7168 and wins on partitionCount at search.go:110" scenario is unreachable — a 0-traffic candidate is never a challenger).
- **Fix:** Re-derived the full outcome by hand (probe-verified: winner M=cta/64/8, N=seq/1/8, K=seq/1/8; lhs 32 B FRAG reuse 8 promoted; rhs 256 B FRAG declines ReasonLowReuse at per-thread M-trip ceil(8/32) = 1) and pinned promotedFrag = 32 / tight.MaxFragPerThread = 31 — the OLD values — with the complete §2.4+§2.6 arithmetic in the test comment. NOT a silent re-pin: the pin's rationale is written into the test and escalated here for author visibility (the 261026-ky1 precedent). Note the amendment DID move this fixture's arithmetic (rhs now declines for low reuse rather than for FRAG bytes; K=WARP vectors now die in eviction) — just not the pinned sum.
- **Files modified:** internal/opt/tiling/verify_test.go
- **Commit:** 78f3060

**2. [Rule 1 - stale comment] schedule pipeline-fixture comments re-derived (Task 2 sanctioned path)**
- **Found during:** Task 2 — internal/opt/schedule/pipeline_test.go's matmulGraphK doc promised "promote lhs (SMEM window) and decline rhs (ReasonLowReuse) — exactly one fill", which an SMEM-class rhs can never do under the amendment (collective trips ≥ T ≥ 2).
- **Fix:** Probed the actual records (kExtent ∈ {1024, 32768}): both select M=cta/4/256, N=seq/1/256, K=warp/2/32 with lhs = 4 B FRAG (reuse 256) and rhs = 65536 B SMEM PROMOTED (collective reuse 256) — i.e. the fix33 pattern. The placement pins (exactly one FRAG fill at SN 4 in a ThrLoop block, six-loop chain, no GMEM temporary, twin-run determinism) all hold unchanged — the SMEM fill is SMEM-class storage the fragment walk never sees — so only the two comments were updated (fill placement stays derived-not-mandated, unpinned). No schedule-production or expectation change.
- **Files modified:** internal/opt/schedule/pipeline_test.go
- **Commit:** b7f02b6

### Scope audit (success-criteria check)

- search.go diff = the score accumulation line + its comment, NOTHING else (eviction loop, digit lists, enumeration order, partState pruning, better(), zero-cache warning untouched — git-diff verified).
- judge.go diff = siteJudgment.holders field, judgeSite holders/gate, siteReuseCount body, comments (siteWindowBytes and tier logic untouched).
- record.go diff = comments + the ReasonLowReuse STRING TEXT (name stable — the plan's action item sanctioned exactly this; no test pins the string, verify_test greps the verify error for "judge").
- Grep gates: `ceilDiv(d.TileSize, ctx.blockDim)` appears exactly once, inside the FRAG branch (judge.go:263); `holders` ×3 in search.go; `siteReuseCount(sf, iters, ctx, class)` call shape unchanged (verify cross-check untouched, zero verify.go edits).
- Fix33/matmulWorked pins matched the plan EXACTLY (decisions, bytes, margins) — no author-ratification loop needed there; the only divergence was Deviation 1.

## Downstream selection movements (Task 2 report)

| Fixture | Old winner | New winner | Arithmetic |
|---------|-----------|------------|------------|
| fix33 (M=4096) + matmulUsed (M=1024) + matmul | same schedule; rhs record Reuse 4 | same schedule; rhs Reuse 256 | collective T_M replaces per-thread trips; margin 16776960 vs 16776704 = 256 (schedule pin immovable) |
| schedule 1024/1024/{1024, 32768} fixtures | (stale comment claimed lhs-SMEM/rhs-declined) | M=cta/4/256, N=seq/1/256, K=warp/2/32; rhs 65536 B SMEM promoted | rhs SMEM 65536 ≤ MaxCache, reuse 256 → 16711680 dominates; structural pins held, comments synced |
| internal/cg/cxx goldens (matmul_store.cu, reduce_warp.cu) | — | UNCHANGED (byte-identical, tests green) | no decide-selection movement visible to codegen — no regen |
| pkg/testdata/matmul_scale.ir (GPU-gated e2e) | — | not run hermetically (TCACOMP_E2E off, by design); input IR fixture, no regen; numerical acceptance rides the existing T6 GPU gate |

## Gates

- Hermetic `go test ./... -count=1`: ALL GREEN (25 packages, no TCACOMP_E2E).
- `go build ./...` + `go vet ./...`: green. `go list -m all | wc -l` == 1 (MODULES-OK).
- `gofmt -l internal/opt/tiling` == 0.
- TDD gates: RED commit 4c1ce27 (all six pins failing for exactly the amendment reasons) before GREEN commit 78f3060.

## Known Stubs

None — all pinned arithmetic is live judge/search output, no placeholders.

## Threat Flags

None — internal compiler pass arithmetic only; no new trust-boundary surface. T-tq8-02 (score-product overflow) disposition unchanged/accepted: products stay far below Go's 64-bit int range (max promoted-SMEM traffic ≈ bytes ≤ 65536 × reuse ≤ ~2^31).

## Self-Check: PASSED

All 7 modified files + SUMMARY.md exist on disk; all 3 commits verified in `git log` (4c1ce27 test/RED, 78f3060 feat/GREEN, b7f02b6 docs); working tree clean except the uncommitted docs artifacts (SUMMARY.md, STATE.md) left for the orchestrator's Step 8 docs commit per the task constraints.
