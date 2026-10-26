---
phase: quick-261026-ky1
plan: 01
verified: 2026-10-26T17:50:00Z
status: gaps_found
score: 6/7 must-haves verified
overrides_applied: 0
gaps:
  - truth: "The decide pass on the matmul fixture C[4096,2048] += A[4096,512]·B[2048,512] fp32 selects exactly M=CTA LoC=16 T=256, N=WARP K=4 T=64, K=SEQ T=64, with rhs promoted ClassSMEM Bytes=65536 and lhs promoted ClassFRAG Bytes=256 (fix4 §3.3 pinned expectation)"
    status: failed
    reason: "The actual search winner on the §3.3 fixture (M=4096,N=2048,K=512, DefaultParams) is M=CTA LoC=16 T=256, N=SEQ T=256, K=WARP K=2 T=32 — N and K have swapped SoC assignments vs the §3.3 pin. lhs FRAG Bytes=4 (not 256). rhs SMEM Bytes=65536 (at cap, matches). The test itself (TestFix33WorkedExample) documents this as '§3.3 PIN DIVERGENCE (flagged for author ratification)'. The implementer's R-TRIPS ruling (CTA trips = ceil(T/blockDim) × R-SHARING) makes the K=WARP vector higher-traffic than the N=WARP vector, so the searcher correctly picks the higher-traffic schedule per the §2.5 objective formula. The §3.3 pin's SoC assignment is not the traffic maximum under the implementer's cost model."
    artifacts:
      - path: "internal/opt/tiling/tiling_test.go"
        issue: "TestFix33WorkedExample pins M=CTA/16/256, N=SEQ/256, K=WARP/2/32 (lhs FRAG 4B) — does NOT match §3.3's M=CTA/16/256, N=WARP/4/64, K=SEQ/64 (lhs FRAG 256B)"
      - path: "internal/opt/tiling/judge.go"
        issue: "R-TRIPS reuse model (CTA trips = ceil(T/blockDim), WARP trips = ceil(T/32), × cross-warp sharing factor) makes K=WARP the traffic-maximizing choice on the 4096×2048×512 matmul, not N=WARP as §3.3 assumes"
    missing:
      - "The §3.3 worked example's signature (N=WARP K=4 T=64, K=SEQ T=64, lhs FRAG 256B) is not produced by the implemented cost model"
      - "Either the cost model (R-TRIPS/R-SHARING) needs adjustment to reproduce the pin, or the §3.3 worked example needs to be updated to match the actual model's traffic maximum"
---

# Phase quick-261026-ky1 Plan 01: fix4 decide pass + tiling-verify Verification Report

**Phase Goal:** Implement fix4.md §8 Wave-1 tasks T1 (decide pass, §2) and T2 (tiling-verify, §2.7) — replace the portion-grammar searcher with the factorization model and rewrite tiling-verify's re-checks.

**Verified:** 2026-10-26T17:50:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DefaultParams.Validate passes on new field surface; portion grammar gone from internal/opt/tiling | ✓ VERIFIED | TestParamsSurface passes; grep for AutoWarpPortion/PortionCTA/PortionWarp/WarpPortion across internal/ returns zero hits; Params struct has WindowK, TileSizes, CtaLoCPresets, WarpLoCPresets, MaxPartitionCta, MaxPartitionThr, MaxCTA, MaxWarpsPerCTA, MaxCache, MaxFragPerThread, CacheDims, MaxCS, ReducedDimWarp; no MaxThreadsPerCTA or WarpPortion on Params |
| 2 | §3.3 matmul fixture C[4096,2048] selects M=CTA LoC=16 T=256, N=WARP K=4 T=64, K=SEQ T=64, rhs SMEM 65536, lhs FRAG 256 | ✗ FAILED | TestFix33WorkedExample pins M=CTA/16/256, N=SEQ/256, K=WARP/2/32. rhs SMEM=65536 (matches cap), but lhs FRAG=4B (not 256B). N and K have swapped SoC assignments vs §3.3 pin. The test itself labels this "§3.3 PIN DIVERGENCE (flagged for author ratification)". R-TRIPS (CTA trips = ceil(T/blockDim)) makes K=WARP the traffic-maximizing choice under the implemented cost model. |
| 3 | Search is deterministic: two runs produce deep-equal Records | ✓ VERIFIED | TestDeterminismIdempotent passes; fixed enumeration order (outermost window dim = most significant digit, digits ordered SoC then LoC DESCENDING then tile ASCENDING); strictly-greater lexicographic (traffic, partitionCount) selection with first-wins ties |
| 4 | Partial candidates rejected incrementally on MAPM/MALoC violation; WARP K=1 never enumerated; WARP never on reduced dims when ReducedDimWarp=false | ✓ VERIFIED | TestIncrementalPruning passes (pruned > 0, completed < raw product); WarpLoCPresets start at 2 (validated by Params.Validate); TestReducedDimWarpPruning passes (WARP digits pruned on reducer-consumed dims); search.go digit placement rejects partial vectors at placement time (stats.pruned counter proves incremental rejection) |
| 5 | Budget overflow triggers greedy value-density spill (not hard error); zero-cache emits warning; structural infeasibility is hard error | ✓ VERIFIED | TestEviction passes (both sub-tests: floors_beyond_budget_still_schedule and density_order_via_judge_entries); zero-cache logs WARN-level message "tiling: zero-cache schedule (all candidate sites evicted or declined)"; TestStructuralInfeasibility passes (no CTA dim → hard error); eviction order: min density (reuse-1), ties → smaller bytes first |
| 6 | tiling-verify rejects corrupted record for EACH new check (7 checks) | ✓ VERIFIED | All 7 checks present with negative tests: (1) CTA LoC outside presets, (2) WARP LoC outside presets, (3) SEQ LoC != 1, (4) CTA product > MaxCTA, (5) WARP product > MaxWarpsPerCTA, (6) zero CTA dims, (7) TileSize < 2, (8) SMEM budget overflow, (9) judge disagreement. Plus TestVerifyMultiCTAGreen proves v0-policy deletion (2 CTA dims verify green). All tests pass. |
| 7 | Hermetic repo gate green: go build/vet/test ./... with TCACOMP_E2E unset, module count 1 | ✓ VERIFIED | `go build ./...` — clean; `go vet ./...` — clean; `go test ./... -count=1` — all 23 packages pass; `go list -m all | wc -l` = 1; gofmt clean over touched trees |

**Score:** 6/7 truths verified (1 FAILED — §3.3 pin divergence)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `internal/opt/tiling/record.go` | New Params surface (§2.1 fields), ReasonEvicted, absolute-LoC IterDecision, SiteDecision with ReasonEvicted | ✓ VERIFIED | Params has all §2.1 fields; ReasonEvicted constant defined (line 161); IterDecision.LoC is absolute; SiteDecision.Reason supports eviction; Validate fail-fasts on new fields; Params.MaxThreadsPerCTA and Params.WarpPortion absent (verified by TestParamsSurface); Provenance.MaxThreadsPerCTA correctly retained per R-SEAM |
| `internal/opt/tiling/search.go` | Factorization feasible-area iterator with incremental pruning, deterministic enumeration, eviction, objective + tiebreaks | ✓ VERIFIED | Mixed-radix odometer over window dims; incremental pruning at digit placement (MAPM/MALoC/R-LOCCAP/R-REDUCED); deterministic order (LoC desc, tile asc); (traffic, partitionCount) lexicographic selection; greedy density eviction; searchStats with pruned counter |
| `internal/opt/tiling/judge.go` | Tier rules, factor-model bytes (K_d×T_d, cross-warp sharing), lane-aware reuse, promotion gates | ✓ VERIFIED | Tier order: CTA-covered refusal → C1b forced FRAG → WARP-covered SMEM → all-SEQ FRAG; siteWindowBytes implements K_d×T_d factor model (R-C1B); cross-warp sharing = Π K_d over WARP dims not in INS (R-SHARING); reuse uses R-TRIPS (SEQ→T, CTA→ceil(T/blockDim), WARP→ceil(T/32)); promotion gates: traffic > 0 + FRAG bytes ≤ MaxFragPerThread |
| `internal/opt/tiling/constraints.go` | MAPM/MALoC/CuMAP/MASRAM constraint helpers shared by search and verify | ✓ VERIFIED | 112 lines, 8 functions; MAPM check (ctaDims ≤ MaxPartitionCta, thrDims ≤ MaxPartitionThr); MALoC product checks; CuMAP (≥1 CTA dim); used by both search (incremental pruning) and verify (re-checks) |
| `internal/opt/tiling/verify.go` | §2.7 re-check set with deleted checks removed, CtaLoCPresets reference | ✓ VERIFIED | New checks: LoC ∈ presets (CTA/WARP/SEQ), CTA product ≤ MaxCTA, WARP product ≤ MaxWarpsPerCTA, ≥1 CTA dim, TileSize ≥ 2, kernel-wide SMEM/FRAG budget, judge cross-check; deleted checks grep-clean (portion/Portion/F2/v0-policy all absent); judgeSite cross-check re-derives over record's schedule |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| search.go / judge.go | siteWindowBytes | ONE shared factor-model function consumed by search, verify cross-check, and lower_cache.resolveBudget | ✓ WIRED | siteWindowBytes defined in judge.go; called from search (via judgeSite → siteWindowBytes), from verify (judgeSite cross-check), and from lower_cache.resolveBudget (line 130). Single source of truth — fail-closed agreement. |
| lower_rewrite.go coupledThreads | record Iters (warp K product) | blockDim = (Π warp K_d) × 32 derived from lr.decisions — no portion arithmetic | ✓ WIRED | `blockDimOf()` computes product of WARP dim LoCs × 32; used in rewriteLNT for coupled walk step; replaces old portion formula |
| lower_rewrite.go recordProvenance | Provenance.MappingPlan + finallower | GridPortions = per-CTA-dim absolute LoCs, ThreadPortions = per-WARP-dim Ks (DoSoC order), MaxThreadsPerCTA = max over LNTs of (Π K × 32) | ✓ WIRED | recordProvenance() builds grid/thread shape vectors from record decisions; Provenance.MaxThreadsPerCTA field name retained per R-SEAM; filled with derived blockDim |
| verify.go | judgeSite | Tier-class agreement cross-check re-derives the judge over the record's own schedule | ✓ WIRED | verify.go line 251: `j := judgeSite(sf, dec, ctxSchedule, v.Params)` — re-derives per-LNT blockDim context from record, calls judgeSite, compares Promoted/Class/Reuse/Bytes |
| tiling.go Run | searchSchedule | Kernel-wide SMEM/FRAG running floors threaded through LNTs in Roots() order | ✓ WIRED | tiling.go lines 113-120: smemFloor, fragFloor initialized at 0; passed to searchSchedule per LNT; accumulated after each LNT (smemFloor += res.smem, fragFloor += res.frag); Roots() order preserved |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| search.go (decisions) | digits → IterDecision map | Mixed-radix odometer over window dims, filtered by constraints | Yes — real factorization search over BIV extents, tile presets, SoC choices | ✓ FLOWING |
| search.go (site judgments) | SiteDecision map | judgeSite with siteWindowBytes + siteReuseCount + eviction | Yes — real per-site byte/reuse computation from factor model, then greedy density eviction against running floors | ✓ FLOWING |
| verify.go (cross-check) | judge agreement | Re-derived judgeSite over record decisions + blockDimOf | Yes — independent re-derivation of same factor model, compared against record values | ✓ FLOWING |
| lower_cache.go (budget) | window bytes | siteWindowBytes (shared) | Yes — same factor model used for cache sizing, fail-closed agreement with record | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full hermetic build | `go build ./...` | exit 0, no output | ✓ PASS |
| Full hermetic vet | `go vet ./...` | exit 0, no output | ✓ PASS |
| Full test suite (TCACOMP_E2E unset) | `go test ./... -count=1` | all 23 packages pass | ✓ PASS |
| Module count = 1 | `go list -m all \| wc -l` | 1 | ✓ PASS |
| §3.3 pin (4096 matmul) | `go test -run TestFix33WorkedExample` | PASS (but pins DIFFERENT values than §3.3 spec) | ✗ FAIL (divergence — see Gap #1) |
| Determinism | `go test -run TestDeterminismIdempotent` | PASS | ✓ PASS |
| Incremental pruning | `go test -run TestIncrementalPruning` | PASS | ✓ PASS |
| Eviction + zero-cache warning | `go test -run TestEviction` | PASS (WARN log emitted) | ✓ PASS |
| Structural infeasibility | `go test -run TestStructuralInfeasibility` | PASS (hard error) | ✓ PASS |
| Reduced-dim WARP pruning | `go test -run TestReducedDimWarpPruning` | PASS | ✓ PASS |
| Verify negative tests (all 9) | `go test -run TestVerifyFlagsCorruption` | 9 sub-tests all PASS | ✓ PASS |
| Verify multi-CTA green (v0-policy deletion) | `go test -run TestVerifyMultiCTAGreen` | PASS | ✓ PASS |
| Params surface retirement | `go test -run TestParamsSurface` | PASS | ✓ PASS |
| Portion grep clean (internal/) | `grep -rn 'AutoWarpPortion\|PortionCTA\|PortionWarp\|WarpPortion' internal/` | zero hits | ✓ PASS |
| verify.go deletion grep | `grep -n 'portion\|Portion\|F2\|v0.policy\|v0-policy' internal/opt/tiling/verify.go` | zero hits | ✓ PASS |
| gofmt clean | `gofmt -l internal/opt/tiling internal/opt/finallower internal/cg/cxx internal/cg` | zero files | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIX4-W1-T1 | plan 01 | fix4 §2: decide pass — Params, feasible area, constraints, cost model, objective, eviction, record | PARTIAL | All components implemented and functional; §3.3 pin diverges (see Gap #1) |
| FIX4-W1-T2 | plan 01 | fix4 §2.7: tiling-verify re-checks + deletions | ✓ SATISFIED | All 7 new checks with negative tests; 5 deleted checks grep-clean; verify F7 stamp intact |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None | — | No TBD/FIXME/XXX markers; no TODO/HACK/PLACEHOLDER in production code; no console.log-only implementations; no hardcoded empty data in production paths |

### Human Verification Required

None required for programmatic verification — automated tests cover all verifiable claims. The §3.3 divergence is a design/modeling question for the author (whether to adjust the cost model or update the spec), not a visual/behavioral check needing human UAT.

### Gaps Summary

**One critical gap:** The §3.3 worked example pin does not match the implemented cost model's output. The plan's must-have truth #2 specifies M=CTA/16/256 + N=WARP/4/64 + K=SEQ/64 + lhs FRAG 256B for the 4096×2048×512 matmul. The actual implementation produces M=CTA/16/256 + N=SEQ/256 + K=WARP/2/32 + lhs FRAG 4B. The implementer's R-TRIPS ruling (CTA trips = ceil(T/blockDim) combined with cross-warp sharing) makes K=WARP the traffic-maximizing choice, not N=WARP as §3.3 assumes. The test itself documents this as "§3.3 PIN DIVERGENCE (flagged for author ratification)."

This is a BLOCKER because:
1. The plan explicitly lists §3.3 exact pin as a must-have truth
2. The SUMMARY.md claims "PASS — exact match with fix4 §3.3" which is contradicted by the actual test code
3. Either the cost model needs adjustment or the spec needs updating — the two are inconsistent

The fix requires author decision: either the implementer rulings (R-TRIPS and/or R-SHARING) need revision to reproduce the §3.3 pin, or the §3.3 worked example needs to be updated to reflect the actual traffic maximum under the implemented model.

---

_Verified: 2026-10-26T17:50:00Z_
_Verifier: Claude (gsd-verifier)_
