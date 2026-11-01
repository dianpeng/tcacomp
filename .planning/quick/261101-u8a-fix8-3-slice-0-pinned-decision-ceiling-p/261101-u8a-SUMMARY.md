---
phase: quick-261101-u8a
plan: 01
subsystem: compiler-tiling-search
tags: [fix8, slice-0, pinned-decision-probe, occupancy, measurement, stop-rule, env-hook, tdd, gpu-validation]

# Dependency graph
requires:
  - phase: quick-261101-oqg
    provides: "fix7 §5 slice 1c — the 1c sha256 manifest (the zero-diff HARD gate's reference) and the 1c min-µs numbers this probe re-measures in-session"
provides:
  - "the env-gated Params overlay (paramsFromEnv/mustParamsFromEnv, TCACOMP_TILING_CTA_LOC/_TILES/_MAX_CTA/_NOMINAL_EXTENT/_WARP_LOC) at the three tiling-family factories — OFF by default, sha256-proven zero-default-path; the reproducible instrument for any future decision-pinning measurement"
  - "the MEASURED fix8 §3 record: the STOP verdict at 1.00x (slice 1 BLOCKED), the 22-point bench table with launched AND active threads, the two-series curve, and the measured-concrete contradiction mechanisms slice 1's re-derivation must answer"
  - "the corrected 3-knob pin recipe: single-dim presets [N] + MaxCTA=N + NominalExtent=N x maxTile (the plan's 2-dim presets are provably unrealizable — adaptive-digit tile collapse + tier-1 both-CTA refusal)"
affects: [fix8-slice-1-objective-amendment, fix5-MaxCTAPolicy-revisit, nx-cross-product-lowering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "env-gated Params overlay at registry factories: search and verify/lower read the SAME env in lockstep by construction — no override semantics, decisions stay search-chosen; panic-carried named errors are the fail-fast choke where the Factory signature has no error return"
    - "intended-vs-measured decision reconciliation: every sweep point's grid/blockDim is READ BACK from the emitted .cu (launch header + LNT schedule comment) and asserted; re-labeled points stay in the table and plot against MEASURED threads"
    - "launched AND active threads per point: active = Prod_d min(LoC_d, ceil(extent_d/T_d)) x blockDim against runner-baked extents — the number the occupancy model's NominalExtent stand-in mis-credits"

key-files:
  created:
    - "internal/opt/tiling/params_env.go"
    - "internal/opt/tiling/params_env_test.go"
    - ".planning/quick/261101-u8a-fix8-3-slice-0-pinned-decision-ceiling-p/{probe.py, 1c-manifest-copy.txt, zero-diff-manifest.txt, probe-manifest.txt, probe-readback.txt, probe-bench.txt, probe-curve.txt, memcheck.txt, suite-run.txt, stop-rule-verdict.txt}"
  modified:
    - "internal/opt/tiling/{tiling.go,verify.go,lower.go} (exactly the 3 factory lines -> mustParamsFromEnv)"
    - "design/fix/fix8.md (§3 STATUS + measured record + STOP verdict; top Status line; §8 row 0)"

key-decisions:
  - "STOP verdict honored with NO further interpretation: best pinned row_sum 5,240.4 µs vs in-session unpinned 5,239.6 µs = 1.00x << 10x — recorded in stop-rule-verdict.txt with the arithmetic shown; fix8.md says slice 1 does not proceed on this model"
  - "The pin recipe was CORRECTED before the sweep froze (isolation runs E1-F9b): grid N must ride the PRESET list (the R1 adaptive digit collapses to its lowest tile under the MaxCTA clamp) and must be SINGLE-dim (2-dim CTA factorizations cover every load subscript -> R2 tier-1 refusal -> zero traffic -> the 1-dim shape or partism wins)"
  - "dx-g64/dx-g512 are recorded FAILED points, not skips: their all-CTA-covered schedules select the v0-unreachable GlobalReducer (the known D1 defect class) and assert at launch under both bench and memcheck"
  - "The only performance bend in the whole sweep is dup_expr dx-g256 at 262,144 ACTIVE threads (blockDim-1024 warp-K=32 landing): 998.7 µs vs 1,740.6 unpinned = 1.74x — active threads buy time when they actually materialize, but via the WarpReducer SMEM path, not the CTA axis"

requirements-completed: [QUICK-261101-U8A]

# Metrics
task-commits:
  - "9125067 test(quick-261101-u8a): fix8 §3 slice 0 RED pins — env-gated Params overlay contract"
  - "ecd7a65 feat(quick-261101-u8a): fix8 §3 slice 0 GREEN — env-gated Params hook at the three tiling factories"
  - "f856907 docs(quick-261101-u8a): fix8 §3 — slice-0 probe landed, STOP rule FIRED at 1.00x"
  - "(probe.py + probe-*.txt evidence live in this task dir uncommitted-by-executor per the orchestrator's instruction; the orchestrator commits them with this SUMMARY)"

gates:
  hermetic: "go build/vet green; go test ./... 25 packages ALL GREEN; go list -m all | wc -l == 1; scope gate: search.go/judge.go/record.go/constraints.go/lower*.go/internal/cg/runtime/ ZERO diff (git-diff verified); production diff = params_env.go + params_env_test.go (new) + exactly 3 factory lines"
  hard: "zero-diff gate PASS: all 14 unpinned suite emissions sha256 BIT-IDENTICAL to the 1c manifest (machine-checked empty diff; zero-diff-manifest.txt)"
  sweep: "22 points emitted (>= 18), 0 compile failures, 17 asserted-clean + 5 honest RE-LABELED (all at intended grids, different blockDim), every row carries launched AND active threads"
  gpu: "bench --jobs 1 --bench-iters 20, 4 repeats/point, unpinned baselines FIRST in-session, min-µs statistic, orchestrator-enforced canary 0 + maxdiff in tolerance on every benched point; memcheck 16 distinct pinned schedules CLEAN (0 errors), 2 FAILED (GlobalReducer D1 class, recorded); unpinned full suite 14/14"
  stop-rule: "explicit arithmetic in stop-rule-verdict.txt; exactly one verdict: STOP at 1.00x"

deviations:
  - "[Rule 3] Pin-recipe correction: the plan's 2-dim presets ([16,4]@G64, [16,8]@G128, [16,16]@G256, [16,32]/[32,16]@G512) do NOT realize — found by isolation experiments (E1-E9, EB-EE, F1-F9b) BEFORE the sweep; two independent mechanisms recorded in probe.py's POINTS comment and fix8.md §3. Corrected to single-dim [N] presets for rs-g64..512 / ax-g64..256 / dx-g128; the plan's small-tile points kept as-is (their warp-ceremony landings are findings); two over-provisioned points ADDED (ax-g512t256, dx-g256t256) per the research's sweep-density discretion -> 22 points"
  - "[Rule 1] probe.py driver bugs found and fixed across runs (all re-run, measurements never fabricated): BENCH parsing grepped raw BENCH_US lines but the orchestrator prints a summary table; memcheck cleanliness re-keyed on rc==0 (the orchestrator enforces the 0-error SUMMARY contract internally and does not echo it); side-car format spec applied to a string; verdict mins double-joined"
  - "[Rule 1] The zero-diff gate's first run reported FAIL on a comparator artifact (sha256sum './' prefix) while all 14 hashes matched — the comparator was fixed and re-run so the empty diff is genuinely machine-checked, not eyeballed"

duration: ~27 min wall (14:07:52Z -> 14:35:19Z session window; GPU bench/memcheck/suite inside it)
completed: 2026-11-01
---

# Quick 261101-u8a: fix8 §3 Slice 0 — Pinned-Decision Ceiling Probe Summary

**One-liner:** The env-gated Params hook landed (sha256-proven zero-default-path, 3 factory lines) and the 22-point pinned-decision sweep measured the ceiling: **STOP at 1.00×** — no pinned grid buys ANY time on row_sum/axpy3 because every default-tile schedule runs 32–64 ACTIVE threads regardless of launched count; the only bend in the sweep is dup_expr's 262,144-active-thread warp landing at 1.74×.

## What was measured (the honest record)

- **row_sum**: unpinned 5,239.6 µs (in-session, 4 repeats 5,239.6–5,243.3); pinned grids 16→512 all 5,240.4–5,516.2 µs — the 16,384-launched-threads point (rs-g512) carries **64 active threads** and is 5% WORSE than unpinned. Best pinned = 5,240.4 µs → ratio **1.00×**.
- **axpy3**: 3,523.3 unpinned → 3,524.1–3,579.2 across every pinned point (all 64 active threads); the warp-monster ax-g512 (524,288 active) 3,731.0.
- **dup_expr**: 1,740.6 unpinned (32 active); **dx-g256 = 998.7 µs (1.74×)** at 262,144 active threads — the ONLY improvement in the sweep; dx-g64/dx-g512 FAILED at launch (GlobalReducer assert, D1 class).
- **Curve** (`probe-curve.txt`): the launched axis does not bend for row_sum/axpy3; the active axis is a vertical wall at 64/32 threads; the only bend is the warp-K=32 materialization of active threads on dup_expr.
- **STOP-rule candidates, measured-concrete**: (1) active-vs-launched (NominalExtent stand-in credits 16 active on row_sum where reality is 2); (2) fully-active big CTA grids at blockDim 32 are inexpansible (T<32 zeroes FRAG reuse: ceil(T/32)=1 trip → zero traffic → the search refuses or degenerates); (3) per-thread MLP bounded by the 256 B FRAG budget over the T-strided coupled walk.

## Task record

| Task | Result | Commit |
|---|---|---|
| 1 — env hook + unit pins + sha256 HARD gate | RED→GREEN; all 14 unpinned emissions bit-identical to 1c (HARD gate PASS) | 9125067, ecd7a65 |
| 2 — sweep driver + manifest + readback | 22 points, 17 clean / 5 re-labeled / 0 failed emissions; manifest 22 entries | (task-dir artifacts, orchestrator commits) |
| 3 — GPU bench + curve + STOP verdict + fix8.md §3 | STOP at 1.00× recorded with arithmetic; memcheck 16 clean / 2 D1-FAILED; suite 14/14; fix8.md §3 updated | f856907 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - blocking] Pin recipe corrected from 2-dim to single-dim presets**
- **Found during:** Task 2 (first emission pass: 13/20 points landed on the DEFAULT binary)
- **Issue:** Two independent mechanisms void the plan's 2-dim pins: (a) the R1 adaptive LoC digit collapses to its FIRST (lowest) tile when MaxCTA clamps the ladder, so grid N is only enumerable at T=256 if N is a preset; (b) a 2-dim CTA factorization covers every load-subscript dim of these kernels → R2 tier-1 refuses the sites → zero traffic → the single-dim shape (or the partitionCount tiebreak) wins.
- **Fix:** Single-dim presets [N] + MaxCTA=N + NE=N×maxTile, verified point-by-point by isolation runs before freezing the table; small-tile points kept (their warp-ceremony landings are findings); 2 over-provisioned points added. 22 points, 17 asserted-clean.
- **Files:** probe.py (POINTS table + recipe comment)
- **Commit:** n/a (task-dir artifact)

**2. [Rule 1 - bug] probe.py driver parser defects**
- **Found during:** Task 3 (first bench pass: 20/22 rows INCOMPLETE)
- **Issue:** parse_bench grepped raw `BENCH_US` lines, but the orchestrator captures those and prints a summary table; memcheck cleanliness keyed on an unechoed SUMMARY line; two smaller format bugs.
- **Fix:** Parse the orchestrator's bench table row; key cleanliness on rc==0 (the orchestrator enforces the 0-error sanitizer contract + canary/maxdiff itself); fixed the side-car and verdict formatters. All stages re-run — fixtures were content-hash cached, so the re-runs measured the same binaries.
- **Files:** probe.py
- **Commit:** n/a (task-dir artifact)

**3. [Rule 1 - bug] zero-diff gate comparator artifact**
- **Found during:** Task 1 HARD gate (first run: FAIL with all 14 hashes matching)
- **Issue:** `sha256sum ./*.cu` prefixes names with `./`, so the textual diff was non-empty despite bit-identical hashes.
- **Fix:** Fixed the comparator and re-ran — the empty diff is machine-checked (threat T-u8a-02 honored: never eyeballed).
- **Files:** /tmp gate script; transcript `zero-diff-manifest.txt`
- **Commit:** n/a

### Recorded findings (not deviations)

- **dx-g64 / dx-g512 FAIL at launch**: all-CTA-covered schedules select the v0-unreachable GlobalReducer (D1 defect class from the t4u suite record) — kept as FAILED rows in probe-bench.txt and memcheck.txt, never skipped.
- **rs-warpk2-g16 did not take the warp digit** (assumption A1 miss): WarpLoCPresets=[2] does not exclude SEQ on the reduced dim; the point is binary-identical to unpinned and recorded as such.
- **Small-tile points land warp-K=32 monsters** (blockDim 1024): T<32 zeroes FRAG reuse, SMEM warp windows fit the 64 KB budget at small tiles, tierEff credits them fully under the NE stand-in — one of them (dx-g256) is the sweep's only speedup.

## Auth gates

None — no authentication-boundary operations in this plan.

## Known Stubs

None — `paramsFromEnv` is fully wired into the three production registry factories; probe.py is fully functional driver code (no placeholder logic).

## Threat Flags

None beyond the plan's threat model — the only new input surface (env → Params) is mitigated exactly as registered (T-u8a-01: strict parse, named errors, Validate floors retained; T-u8a-02: machine-checked gate; T-u8a-03: every point recorded with reconciliation, no cherry-picking surface).

## Self-Check: PASSED

All 14 claimed artifacts exist on disk (params_env.go, params_env_test.go, fix8.md, probe.py + 9 evidence transcripts + this SUMMARY); all 3 task commits present in git log (9125067, ecd7a65, f856907); hermetic gates green at HEAD (build/vet/test 25 packages, module count 1); scope gate zero-diff on search/judge/record/constraints/cg/runtime.

