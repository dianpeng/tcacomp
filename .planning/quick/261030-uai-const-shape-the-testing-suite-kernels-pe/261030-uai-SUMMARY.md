---
phase: quick-261030-uai
plan: "01"
subsystem: testing
tags: [tca, const-shape, shape-specialization, fix6, cuda, testing-suite]

# Dependency graph
requires:
  - phase: quick-261029-v9k
    provides: the §3.1 surviving-WARP prune + the 2026-10-30 const-K probe verdict (fix6.md §4) that proved schedules bit-identical under const-K and fixed the fully-static D-fix6-01/03 blockers' scope
provides:
  - six const-shaped suite kernels (mixed dynamic/static forms) as fixtures for fix6 per-shape specialization
  - before/after sha256 manifests + preserved emitted .cu for the full 14-kernel suite
  - a hunk-local normalized-diff gate proving schedule identity modulo the Launch wrapper signature
  - runner Launch-signature coupling (D-fix6-02) exercised by the suite
affects: [fix6, shape-specialization, testing-suite, scheduling]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "const-extent flip discipline: manifest-first (before sha256 of all 14 emitted .cu from a fresh scripts/build.sh rebuild), flip, schedule-identity diff gate, then GPU suite + memcheck"
    - "runner Launch faces must be read from the actual emitted wrapper, never assumed (D-fix6-02)"

key-files:
  created:
    - .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-sha256-manifest.txt
    - .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/after-sha256-manifest.txt
    - .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-cu/ (14 emitted .cu, pre-flip)
    - .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/after-cu/ (14 emitted .cu, post-flip)
  modified:
    - testing/kernels/dedup/dup_expr.tca
    - testing/kernels/dedup/dup_const.tca
    - testing/kernels/fusion/axpy3.tca
    - testing/kernels/fusion/softmax.tca
    - testing/kernels/runtime/row_sum.tca
    - testing/kernels/runtime/scale_copy.tca
    - testing/runners/dup_expr.cu
    - testing/runners/dup_const.cu
    - testing/runners/axpy3.cu
    - testing/runners/softmax.cu
    - testing/runners/row_sum.cu
    - testing/runners/scale_copy.cu
    - design/fix/fix6.md

key-decisions:
  - "row_sum/softmax use the const-MIDDLE mixed form [R, Ku, C] — fully-static is blocked by D-fix6-01 (repeated extents) and D-fix6-03 (cg fill coordinate reconstruction) per fix6.md §4"
  - "runners pass the baked dK constant (not a fresh literal) for the e<extent> param, keeping one source of truth for the extent value"
  - "suite.toml left untouched: its dims fields are reporting mirrors of the runtime dims, which the const declaration does not change (runner sources hold the load-bearing values)"

patterns-established:
  - "const-shape flip gate: unflipped kernels must stay sha256-bit-identical; flipped kernels must pass a hunk-local normalized diff (dim identity K↔eN + extent literal renames normalized away) with residual diffs confined to the Launch wrapper signature region"

requirements-completed: [QUICK-261030-UAI]

# Metrics
duration: 14min
completed: 2026-10-30
---

# Quick 261030-uai: Const-shape the testing suite kernels Summary

**Six non-matmul suite kernels now declare their reduced extent as a compile-time constant in the mixed dynamic/static form (const-middle for row_sum/softmax), with runner Launch signatures matched to the actual emitted `uint32_t e<extent>` wrapper; suite stays 14/14 with schedules proven bit-identical modulo the wrapper signature.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-10-30T13:55:41Z
- **Completed:** 2026-10-30T14:09:54Z
- **Tasks:** 3
- **Files modified:** 13 (6 kernels + 6 runners + design/fix/fix6.md)

## Accomplishments

- All six in-scope kernels flipped to const-K mixed forms: dup_expr/dup_const `[R, C, 8u]`, axpy3/scale_copy `[R, C, 4u]` (trailing reduced), row_sum `[R, 4u, C]`, softmax `[R, 8u, C]` (const-MIDDLE, per the D-fix6-01/03 fully-static blockers).
- bin/tcacomp rebuilt from current source via scripts/build.sh (the gitignored stale prebuilt shadowed it); BEFORE manifest of all 14 emitted .cu recorded from unmodified sources.
- The eight unflipped kernels' emitted .cu are sha256 BIT-IDENTICAL before/after (automated gate).
- The six flipped kernels differ ONLY in the Launch wrapper signature + const-extent propagation: a hunk-local normalized diff gate (normalize `eN`→`K` prefix-included and the standalone extent literal→`K` within each hunk; residual diffs must be wrapper-region and whitelist-matched) passes for all six — kernel bodies, LNT schedule vectors (`R=CTA(T=256,LoC=16) C=SEQ K=WARP(T=32,LoC=2)`), grid/block/smem literals, and reducer setup are byte-unchanged, confirming the 2026-10-30 probe verdict end-to-end on the suite.
- GPU suite 14/14 at --jobs 1 (A6000, canary 0, maxdiff ≤ 5.3e-05, all within tolerances); memcheck clean on all six flipped kernels.
- fix6.md §2 carries the appended const-shape-suite evidence; §4 notes D-fix6-02's runner coupling is now suite-exercised.

## Task Commits

1. **Task 1: Rebuild bin/tcacomp and record the BEFORE sha256 manifest** — no commit (bin/tcacomp is gitignored; the manifest + before-cu/ are docs artifacts the orchestrator commits).
2. **Task 2: Flip the six kernels + match runner Launch signatures** — `bc9d2cc` (feat)
3. **Task 3: AFTER validation + fix6.md update** — `3fd962b` (docs)

**Plan metadata:** not committed by the executor (quick-task constraint: orchestrator owns the docs commit).

## Files Created/Modified

- `testing/kernels/dedup/dup_expr.tca`, `testing/kernels/dedup/dup_const.tca` — `[R, C, K]` → `[R, C, 8u]`
- `testing/kernels/fusion/axpy3.tca`, `testing/kernels/runtime/scale_copy.tca` — `[R, C, K]` → `[R, C, 4u]`
- `testing/kernels/runtime/row_sum.tca` — `[R, K, C]` → `[R, 4u, C]` (const-middle)
- `testing/kernels/fusion/softmax.tca` — `[R, K, C]` → `[R, 8u, C]` (const-middle)
- `testing/runners/{dup_expr,dup_const,axpy3,softmax,row_sum,scale_copy}.cu` — Launch prototypes + all call sites gain the prepended `uint32_t e<extent>` param, passing the runner-baked `dK`
- `design/fix/fix6.md` — §2 appended const-shape-suite evidence; §4 D-fix6-02 suite-exercised note
- `.planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/{before,after}-sha256-manifest.txt` + `{before,after}-cu/` — manifests and preserved emitted .cu (docs artifacts)

## Per-flipped-kernel diff attribution

All six: 28 changed lines (14 replacement pairs). Exactly four difference classes, all expected per D-fix6-02 and the probe:

1. **Kernel signature + TensorView construction**: `uint32_t K` → `uint32_t eN`; `x_v(x, R, C, K)` → `x_v(x, R, C, <ext>u)` (middle form for row_sum/softmax: `x_v(x, R, <ext>u, C)`).
2. **Dim-identity rename**: LNT comment `K=thread/warp…` → `eN=thread/warp…`; iterators `K_it4/K_it5` → `eN_it4/eN_it5` + their uses; `auto vN = K` → `auto vN = <ext>u`.
3. **Wrapper signature**: `void <Name>Launch(uint32_t eN, …)` — the prepended const-dim param.
4. **Wrapper const-extent propagation**: `const uint32_t K = x.Dim(n);` and `TCA_REQUIRE(z.Dim(n) == K, "…(dim K)")` for the const dim removed; the kernel launch passes `eN` in the dim position.

Nothing else changed: grid `dim3(16u,…)`, block `dim3(64u,…)`, smem `8u`, `__launch_bounds__(64)`, ValidateLaunch, tile constants, and reducer setup are byte-identical.

## Decisions Made

- **suite.toml no-op (plan-mandated record)**: left untouched. Its dims fields are reporting mirrors of the runtime dims; the const declaration changes only the compile-time spelling, not the runtime sizes the runners allocate and pass. The runner sources hold the load-bearing values (dR/dC/dK unchanged).
- **Call sites pass `dK`, not a fresh literal**: keeps the extent value single-sourced in each runner; the emitted param name (`e4`/`e8`) is position- and extent-derived, matching the actual wrapper read in Task 2 step 2.
- **dimvar declarations unchanged**: `(i, j, k)` binds iteration variables, not shape dims; only the tensor shape lists changed, per the plan.

## Deviations from Plan

None - plan executed exactly as written.

(Two execution notes, not deviations: Task 1 produced no commit — bin/tcacomp is gitignored and the manifest is a docs artifact; the emit helper script lived in /tmp and never entered the tree.)

## Issues Encountered

- None blocking. One observation for the fix6 author: the const-dim Launch wrapper no longer derives K from `x.Dim(n)` and drops the `z.Dim(n) == K` cross-validation for the const dim (the `e<extent>` param is host-trusted). This is the compiler's existing const-dim emission (probed green 2026-10-30), consistent with D-fix6-02's "matching runner/launcher declarations" framing — but the §1 dispatching launcher should own this validation when it lands.

## Known Stubs

None — no placeholder/hardcoded-empty data introduced; all six kernels are fully wired to real inputs.

## Threat Flags

None — no new network endpoints, auth paths, file-access patterns, or schema changes. The plan's threat register (T-uai-01 tampering → mitigated by the manifest + schedule-identity gate; T-uai-02 DoS → accepted, --jobs 1) was executed as written.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The suite now carries six const-shaped fixtures (mixed dynamic/static, incl. const-middle) for fix6 per-shape specialization work.
- fix6 §1 prerequisites still open by design: D-fix6-01 (repeated extents) and D-fix6-03 (cg fill coordinate reconstruction) remain the blockers for fully-static shapes; the dispatching launcher must absorb the D-fix6-02 signature coupling and re-own the dropped z-dim cross-validation.
- Downstream fix6 work can diff against `before-cu/`/`after-cu/` in this task dir as the pinned const-shape baseline.

---
*Phase: quick-261030-uai*
*Completed: 2026-10-30*

## Self-Check: PASSED

- before-sha256-manifest.txt / after-sha256-manifest.txt: FOUND, 14 lines each, unflipped eight bit-identical (automated re-run after Task 3)
- before-cu/ (14 .cu) / after-cu/ (14 .cu): FOUND
- 261030-uai-SUMMARY.md: FOUND
- design/fix/fix6.md: FOUND, §2/§4 updated
- Commit bc9d2cc (Task 2 flip) and 3fd962b (Task 3 fix6.md): verified on branch worktree-agent-a6d3be24c149bf740 via git log
- GPU suite 14/14 and memcheck 6/6: logs captured during execution (orchestrator PASS lines)
