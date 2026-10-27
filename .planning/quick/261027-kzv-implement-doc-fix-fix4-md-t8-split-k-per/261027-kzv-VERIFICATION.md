---
phase: quick-261027-kzv
verified: 2026-10-27T09:20:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
warnings:
  - "Task 3 step 3 (mirror the 261027-kzv row into .planning/STATE.md) was NOT executed and is not recorded as a deviation in the SUMMARY — bookkeeping-only omission; the last-activity line and quick-task table still stop at 261027-iq6. Recommend a one-line chore commit."
---

# Quick Task 261027-kzv: fix4 §3.5 T8 split-k enablement — Verification Report

**Task Goal:** Implement doc/fix/fix4.md T8 — split-k per §3.5: wiring behind ReducedDimWarp=true; dedicated split-k matmul e2e; flip default; goldens.
**Verified:** 2026-10-27 (initial verification)
**Status:** passed

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Under DefaultParams, a reduce-carrying kernel's searched schedule assigns WARP to the REDUCED dim and the full pipeline lowers + emits the WarpReducer split-k bracket | VERIFIED | `record.go:424 ReducedDimWarp: true`; `TestReducedDimWarpPruning` pins the deterministic winner (cta/4/256, seq/1/256, warp/2/32; lhs FRAG 4 B reuse 256, rhs SMEM 65536 B reuse 256) at tiling_test.go:648-680; `TestFrontendMatmulScalePipeline` pins `__launch_bounds__(64)`, `ValidateLaunch(dim3(64u,1u,1u), dim3(64u,1u,1u), 65544u, 512u)`, `WarpReducer<float> reducer0(`, staging at `tca_smem + 65536` extent 2, ThrIterator stride-64 N rounds (frontend_test.go:260-266); both tests PASS in the re-run hermetic suite |
| 2 | Split-k matmul numerically correct on the A6000: both syntax variants canary 0, maxAbs <= 2e-3 | VERIFIED | `/tmp/t8b-e2e.log:19-24` — written + omitted: canary 0, max abs diff 4.1961669921875e-05 (tolerance 0.002), `ok tcacomp/internal/e2e 75.732s`. Machine-parseable success log landed via cbee07d (frontend_test.go:374 `t.Logf`) |
| 3 | compute-sanitizer memcheck AND racecheck report ZERO errors on the split-k kernel | VERIFIED | `/tmp/t8b-sanitizer.log` — TestFrontendMatmulScaleSanitizer PASS (75.30 s); the test runs both tools and asserts via `requireZeroSanitizerErrors` (e2e_test.go:340-345, 365+), so PASS == zero-error summaries |
| 4 | Explicit ReducedDimWarp=false still FAILS CLOSED at tile-lower; rejectWarpReducedDims unchanged | VERIFIED | `lower_test.go:1052` explicit-false negative asserts the defense error names the gate; :1074-1085 default-admits positive control; `git diff 4dfa266..HEAD` on lower.go/search.go/tiling.go contains ZERO non-comment lines (grep for non-`//` +/- lines returned empty) — code byte-identical, comments only |
| 5 | iq6 design gap stays fenced: both t.Skip sites remain, no surviving-WARP gate added | VERIFIED | `grep -c 't\.Skip(matmulGoldenDesignGapSkip)' internal/e2e/e2e_test.go` == 2 (lines 269, 334); `/tmp/t8b-e2e.log:3,8` shows both golden-matmul tests still SKIP with the design-gap reason; no new WARP-legality gate found in search.go (diff comment-only) |
| 6 | Hermetic gate green; module count 1; goldens byte-stable except the new reduce_splitk.cu | VERIFIED | Re-run by verifier: `go build ./... && go vet ./...` OK, `go test ./...` all 25 packages ok, `go list -m all \| wc -l` == 1. `git status` under testdata/ clean — golden.Assert byte-compare (without `-update`) passed, so reduce_splitk.cu regenerates byte-identical; regen mechanism = the shared `golden` package `-update` flag (internal/golden/golden.go:25) |

**Score:** 6/6 truths verified

## Executor Deviations (both verified honest)

1. **C1b FRAG fill-window chunk-origin fix (cd70fa2) + RED pin (feb290e).** Code exists: `lower_rewrite.go:477` materializes `d.chunkBase = roundBase + lane*ceil(T_d/32)`; `lower_cache.go:680-681` substitutes it for FRAG-class WARP-covered dims. RED pin `TestLowerC1bFragChunkOriginWindow` exists at lower_test.go:1740 and PASSES post-fix. Honesty evidence: `/tmp/t8b-numerical.log` (15:46, pre-fix) records the real GPU failure — both variants maxAbs 28.992233276367188 at flat 1546; `/tmp/t8b-numerical2.log` + e2e.log (post-fix) record canary 0 / 4.196e-05. Two goldens regenerated in the same fix commit (matmul_store.cu, tile_lowered_matmul.ir) — confirmed in cd70fa2's stat. Fix is mechanical/verb-translation per the lowers ruling; no fence widened.
2. **Success-path numeric log (cbee07d).** `t.Logf` at frontend_test.go:374 emits canary/maxAbs/tolerance per variant; the 15:57 e2e log shows the values on success. Single-file commit, 7 insertions.

## GPU Evidence (verified in-machine, not just SUMMARY claims)

| Log | Content | Verdict |
|-----|---------|---------|
| /tmp/t8b-numerical.log | PRE-fix run: TestFrontendMatmulScale FAIL both variants, maxAbs 28.99 | honest defect record |
| /tmp/t8b-numerical2.log | post-fix: frontend + reduce regression fixtures PASS | green |
| /tmp/t8b-sanitizer.log | memcheck + racecheck over split-k kernel: PASS (zero errors enforced by requireZeroSanitizerErrors) | green |
| /tmp/t8b-e2e.log | full e2e close-out: both golden-matmul SKIP intact, all runnable fixtures PASS, `ok 75.732s` | green |

## Hermetic Gate (re-run by verifier)

```
go build ./...   OK
go vet ./...     OK
go test ./...    ok (all 25 packages, incl. internal/opt/tiling 0.695s, internal/e2e 0.208s)
go list -m all | wc -l  ->  1
```

## Anti-Patterns

Scanned all 11 files modified by this task for TBD/FIXME/XXX/TODO/HACK: none. No stub patterns; no placeholder returns.

## Fence Audit

- `git diff 4dfa266..HEAD -- internal/opt/tiling/{search,lower,tiling}.go` — every +/- line is a comment; zero code lines changed (rejectWarpReducedDims, reducedIters byte-identical).
- Both `matmulGoldenDesignGapSkip` skip sites present and firing for the right reason.
- No runtime/ changes in the task commits.

## Warnings (non-blocking)

1. **STATE.md row missing.** PLAN Task 3 step 3 required mirroring a `261027-kzv` row into `.planning/STATE.md` (last-activity line + quick-task table). Neither bbf1f8d nor the merge 2bcb0c7 touched STATE.md; the table still stops at 261027-iq6. The SUMMARY's Task-3 commit entry also omits it and records no deviation. Bookkeeping-only — no effect on the task goal. Suggest a one-line chore commit adding the row.
2. The task SUMMARY.md is untracked (`??` in git status) — normal; the orchestrator commits verification artifacts.

## Human Verification Required

None. The GPU numerical/sanitizer outcomes rest on in-machine log transcripts with pre/post-fix contrast, not on SUMMARY narration.

---

_Verified: 2026-10-27_
_Verifier: Claude (gsd-verifier)_
