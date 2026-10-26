---
phase: quick-261026-tq8
verified: 2026-10-27T00:00:00Z
status: human_needed
score: 6/7 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Ratify Deviation 1 (adjudicated CORRECT by this verifier): the plan's truth #3 round-2 prediction (TestVerifyKernelWideFragGate flips 32 -> 4 B via a K=WARP winner) is factually wrong; the 32 B pin stands. Independent confirmation below (live probe + full family sweep). Accept by replying / signing off, or add an overrides entry per the suggestion in this report."
    expected: "Author confirms the 32 B / tight-31 pins and the §2.4+§2.6 derivation in verify_test.go:344-368 are the intended outcome (plan-arithmetic error in the checker round-2 draft, executor Rule-1 escalation upheld)."
    why_human: "The deviation contradicts a checker-ratified plan must-have; only the author can ratify the superseding derivation (261026-ky1 author-ratification precedent)."
---

# Quick Task 261026-tq8: fix4 §8 T1.1 — one-currency objective patch Verification Report

**Task Goal:** Implement doc/fix/fix4.md T1.1 (§8, 2026-10-26 one-currency objective amendment): judge.go `siteReuseCount` branches on class per §2.4, `judgeSite` scales FRAG traffic by holders, one-currency promotion gate; search scores in one currency; affected tests re-pinned incl. a NEW unit-consistency guard; search/enumeration/eviction-order/verify-cross-check/record-shape UNCHANGED; hermetic suite green.
**Verified:** 2026-10-27
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | §3.3 winner rhs records ClassSMEM/65536/Reuse 256 (was 4), decisions + bytes unchanged | ✓ VERIFIED | `TestFix33WorkedExample` (tiling_test.go:237-266) pins M=cta/16/256, N=seq/1/256, K=warp/2/32, blockDim 64, rhs `(promoted, SMEM, 65536, 256)`; PASS live. Judge SMEM branch (judge.go:242-253): ReuseDims {M}, M CTA → `reuse *= d.TileSize` = 256. Old per-thread 4 visible as removed in the git diff. |
| 2 | TestDecideLowerComposition keeps schedule + lhs pins, moves ONLY rhs Reuse 4 → 256; lowered products/golden untouched | ✓ VERIFIED | lower_test.go diff (81cb912..b7f02b6) confined to the fixture comment + rhs pin `sr.Reuse != 256` + failure message; schedule pin (dM/dN/dK block) untouched. `tile_lowered_matmul.ir` and all testdata absent from the commit range. Test PASS live. Margin hand-checked: 16776960 = 16711680 (rhs) + 65280 (lhs); challengers 16776704 = 16711680 + 65024; margin 256. |
| 3 | TestVerifyKernelWideFragGate flips to 4 B K=WARP winner (promotedFrag 32 → 4) | ⚠️ ADJUDICATED — plan prediction WRONG, executor deviation CORRECT | The pin STAYS at 32 B. Independent re-derivation from actual code: (a) LIVE PROBE (temp test, since deleted) dumped the searched record: winner M=cta/64/8, N=seq/1/8, K=seq/1/8 (blockDim 32), lhs = promoted FRAG 32 B reuse 8, rhs = declined `ReasonLowReuse` (per-thread reuse 1) → promotedFrag = 32. (b) Judge probe on the plan's K=WARP challenger: lhs 4 B FRAG r8 holders 256 (7168 pre-eviction) BUT rhs SMEM 2048/8192 B > MaxCache 100 → eviction density ties 7=7 → smaller-bytes-first (search.go:226-235) drops the 4 B lhs, then rhs still overflows → both evacuated → score 0. (c) Full family sweep by hand: M=CTA family max 7168; N=CTA family max 448 (64 B lhs-SMEM at M=seq/N=cta/K=warp-2: reuse {N} CTA → T_N = 8, traffic 64×7); K=CTA both sites tier-1 refused → 0; multi-CTA → 0. Global max 7168 = the all-SEQ winner; traffic decides, the plan's partitionCount-tiebreak scenario (7168 vs 7168) is unreachable. Falls within the plan's done criteria ("holds as-is ... with hand-derived §2.4 arithmetic in a comment" — verify_test.go:344-368). Escalated per Rule 1 → human ratification item. |
| 4 | lhs FRAG 4 B Reuse 256, objective contribution 4×255×64 = 65280 | ✓ VERIFIED | judge.go:140-143 `holders = ctx.blockDim` for FRAG; pinned in TestFix33OneCurrencyEquality (tw == 65280, line 352) and TestFix33WorkedExample lhs pin; PASS live. |
| 5 | Pre-amendment vector rhs EXACTLY equal to winner's: 65536 B, reuse 256, 16711680 both | ✓ VERIFIED | NEW `TestFix33OneCurrencyEquality` (tiling_test.go:283-358) judges rhs under both literal decision maps (winner bdm 64 / pre-amend bdm 128), asserts both `(promoted, SMEM, 65536, 256)` AND `bytes×(reuse−1) == 16711680` for BOTH. Hand-checked: winner rhs 2·32·256·4 = 65536, reuse T_M = 256, sharing 1 (K ∈ INS); pre-amend rhs 4·64·64·4 = 65536, reuse T_M = 256, sharing 1 (N ∈ INS). Also pins the deciding lhs margin 65280 > 32768. PASS live. |
| 6 | Eviction order, enumeration, digit lists, constraints, verify cross-check, record field set UNCHANGED | ✓ VERIFIED | git diff 81cb912..b7f02b6: search.go = score line + comment ONLY (eviction loop 216-248, digitListOf, enumerate, better() untouched); judge.go = holders field + gate + siteReuseCount body + comments (siteWindowBytes, tier logic untouched); record.go = comments + ReasonLowReuse TEXT (name stable; SiteDecision field set unchanged — holders lives on siteJudgment, not the record, per §2.7); verify.go/overlap.go/constraints.go/lower*.go production: ZERO diff. Cross-check `c.sd.Reuse != c.j.reuse` intact at verify.go:291 with no verify.go edit. Grep gates: `ceilDiv(d.TileSize, ctx.blockDim)` exactly once (judge.go:263, FRAG branch); `holders` ×3 in search.go. |
| 7 | Hermetic `go test ./...` green | ✓ VERIFIED | Live run: all packages `ok` (no TCACOMP_E2E); `go build ./...` + `go vet ./...` green; `go list -m all \| wc -l` = 1; `gofmt -l` clean over tiling + schedule. |

**Score:** 6/7 truths verified outright; truth #3 adjudicated (plan-arithmetic error, intent achieved — see human verification item)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `internal/opt/tiling/judge.go` | class-branched siteReuseCount + holders + one-currency gate (`contains: holders`) | ✓ VERIFIED | holders field (line 85), holders set at class determination (140-143), gate `bytes*(reuse-1)*holders <= 0` (154), SMEM/FRAG branches (226-269), sharing tail unchanged (270-283) |
| `internal/opt/tiling/search.go` | score in one currency (`contains: holders`) | ✓ VERIFIED | `res.score += s.j.bytes * (s.j.reuse - 1) * s.j.holders` (line 258); rest of file untouched by diff |
| `internal/opt/tiling/tiling_test.go` | re-pins + NEW TestFix33OneCurrencyEquality | ✓ VERIFIED | fix33Graph() extracted (190-220), TestFix33WorkedExample re-pinned (237), TestFix33OneCurrencyEquality NEW (283), TestPromotionJudge re-targeted (830-857), mirrors updated (923-964, 1132-1186) |
| `internal/opt/tiling/lower_test.go` | rhs record pin (`contains: sr.Reuse != 256`) | ✓ VERIFIED | line 2010; comment + failure message synced; nothing else in the file changed |
| `internal/opt/tiling/verify_test.go` | TestVerifyKernelWideFragGate §2.4 sweep (`contains: promotedFrag`) | ✓ VERIFIED (with adjudicated deviation) | pin 32 B with full §2.4+§2.6 derivation in comment (344-368); tight budget 31 still fires (32 > 31); PASS live |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| search.go evalSchedule | judge.go judgeSite | siteJudgment.holders flows into score | ✓ WIRED | `score += s.j.bytes * (s.j.reuse - 1) * s.j.holders` (search.go:258) matches pattern `score \+= .*holders` |
| judge.go judgeSite | judge.go siteReuseCount | class parameter selects trips | ✓ WIRED | `siteReuseCount(sf, iters, ctx, class)` at judge.go:148, signature unchanged |
| verify.go judge cross-check | judge.go judgeSite | re-derives same judgment, NO verify.go edit | ✓ WIRED | verify.go:291 `c.sd.Reuse != c.j.reuse \|\| c.sd.Bytes != c.j.bytes` — verify.go has zero diff in the task range |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Six pinned tiling tests | `go test ./internal/opt/tiling/ -count=1 -run 'TestFix33\|TestPromotionJudge\|TestDecideLowerComposition\|TestVerifyKernelWideFragGate\|TestTrafficArithmeticAndOptimality' -v` | all PASS | ✓ PASS |
| Disputed fixture record dump (temp probe, deleted) | `go test -run TestProbeTQ8FragGateWinner -v` | winner M=cta/64/8,N=seq,K=seq; lhs 32 B FRAG promoted; rhs ReasonLowReuse; K=WARP challenger evacuates to 0 | ✓ PASS (confirms executor) |
| Tiling package suite | `go test ./internal/opt/tiling/ -count=1` | ok | ✓ PASS |
| Full hermetic suite | `go test ./... -count=1` | all packages ok | ✓ PASS |
| build/vet/modules/fmt | `go build ./... && go vet ./... && go list -m all \| wc -l && gofmt -l` | green / 1 / clean | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIX4-T1.1 | 261026-tq8-PLAN (plan 01) | fix4 §8 T1.1 verify clause: TestFix33WorkedExample re-pinned to §3.3 (decisions unchanged, rhs Reuse 4 → 256); NEW unit-consistency pin (pre-amendment vector rhs EXACTLY equal); hermetic suite green | ✓ SATISFIED | Truths 1, 5, 7 — all verified with live test runs |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | no TBD/FIXME/XXX, no TODO/HACK/PLACEHOLDER, no stub returns in any of the 7 modified files | — | — |

Note: `ReasonLowReuse` string text changed (record.go:162-166) — sanctioned by the plan's action item ("name stable, text generalized"); no test pins the string content (verify_test greps the verify error for "judge", which matches on other text).

### Commit / Scope Audit

Commits 4c1ce27 (test RED), 78f3060 (feat GREEN), b7f02b6 (docs) touch EXACTLY the authorized files: judge.go, search.go, record.go, tiling_test.go, lower_test.go, verify_test.go, schedule/pipeline_test.go (comment-only, sanctioned by Task 2's stale-comment path). No lower*.go production code, no runtime/, no verify.go, no goldens (cg/cxx + pkg testdata absent from the range; cg/cxx tests pass → byte-identical). Working tree clean over internal/ and runtime/ vs HEAD (a8c81d6 = b7f02b6 + worktree merge, identical code tree). STATE.md Quick Tasks row present (uncommitted, for the orchestrator's docs commit).

### Human Verification Required

### 1. Ratify Deviation 1 — TestVerifyKernelWideFragGate pin stays 32 B (plan truth #3 prediction wrong)

**Test:** Review the adjudication above (truth #3): the plan's checker-round-2 flip derivation (32 → 4 B via a K=WARP winner, tight budget 31 → 3) missed that §2.6 eviction is part of the score — the K=WARP challenger's rhs SMEM window (≥ 512 B > MaxCache 100) triggers the density-7 tie and the unchanged smaller-bytes-first tie-break, evacuating BOTH sites to score 0 vs the all-SEQ winner's 7168. The executor's counter-derivation (verify_test.go:344-368 comment; pin 32 / tight 31) was independently confirmed by this verifier via a live record-dump probe and a full feasible-area family sweep.
**Expected:** Author ratifies the 32 B / tight-31 pins as the intended outcome. The executor followed Rule 1 (STOP-and-report, full arithmetic in the test comment, SUMMARY escalation) — no silent re-pin occurred.
**Why human:** The deviation contradicts a checker-ratified plan must-have; only the author can ratify the superseding derivation (261026-ky1 precedent).

**Override suggestion** (to formalize acceptance in a future VERIFICATION.md frontmatter):

```yaml
overrides:
  - must_have: "verify_test.go TestVerifyKernelWideFragGate is re-derived under §2.4: the winner flips DETERMINISTICALLY to a K=WARP vector ... promotedFrag moves 32 → 4"
    reason: "Plan-arithmetic error: the round-2 derivation ignored §2.6 eviction collateral (density ties at 7, smaller-bytes-first drops the 4 B lhs before the over-budget rhs; the K=WARP vector scores 0). Verifier-confirmed by live probe + family sweep; the 32 B pin with hand-derived §2.4+§2.6 arithmetic satisfies the plan's done criteria."
    accepted_by: "{author}"
    accepted_at: "{ISO timestamp}"
```

### Gaps Summary

No gaps. All six re-pinned/NEW test artifacts exist, are substantive, and are wired to the amended judge/search; all gates pass live (tiling package, full hermetic suite, build/vet/module-count/gofmt); the machinery fence holds (diff-audited: eviction, enumeration, digit lists, constraints, verify cross-check, record shape untouched); the three commits are scope-clean. The single open item is governance, not implementation: author ratification of the Rule-1 deviation on TestVerifyKernelWideFragGate, where the executor's derivation was independently confirmed correct and the plan's numeric prediction was wrong.

---

_Verified: 2026-10-27_
_Verifier: Claude (gsd-verifier)_
