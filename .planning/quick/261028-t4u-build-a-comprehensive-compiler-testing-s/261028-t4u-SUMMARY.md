---
phase: quick-261028-t4u
plan: 01
subsystem: testing
tags: [tcacomp, testing-suite, cuda, python, tomllib, cmake, compute-sanitizer, a6000]

# Dependency graph
requires:
  - phase: 01.1 (ir-architecture-refinement)
    provides: the compiler CLI + pipeline the suite drives end-to-end
provides:
  - the repo-root testing/ suite: 14-kernel TCA corpus, 15 runner TUs, TOML
    registry, fixture cmake template, stdlib-only Python 3.13 orchestrator
    (compile -> content-hash cmake build -> GPU run -> numerical verify ->
    colorful reporting, plus benchmark and memcheck/racecheck modes)
affects: [verification, roadmap VER-01/VER-03/VER-05/VER-06/NUM-01, compiler-schedule, compiler-frontend]

# Tech tracking
tech-stack:
  added: [Python 3.13 stdlib only (tomllib, argparse, concurrent.futures), CUDA C++20 runners, CMake 4.4 fixtures]
  patterns:
    - "content-addressed fixture cache with .done sentinels under system temp"
    - "fixed argv lists only, never shell=True, 300s timeouts everywhere"
    - "machine-parseable runner stdout protocol (CANARY/CHECKSUM/REF_CHECKSUM/MAXDIFF/SAMPLES/BENCH_US)"
    - "pinned feRunnerTail launcher contract derived from emitted .cu bytes"

key-files:
  created:
    - testing/kernels/{dedup,fusion,tiling,runtime,llm}/*.tca (14)
    - testing/runners/common.cuh + testing/runners/*.cu (14)
    - testing/suite.toml
    - testing/cmake_template.txt
    - testing/orchestrator.py
  modified:
    - .gitignore (testing/.build/, testing/__pycache__/)

key-decisions:
  - "No tolerance was ever broadened to make a failure pass; every reshape is documented per kernel with the wall that forced it"
  - "Single-input and rank-1-output reductions select the v0-unreachable GlobalReducer (runtime assert) — recorded as a compiler defect; the corpus carries two-input weighted forms instead"
  - "The transpose as-permutation miscompiles (both as-clause and swapped-binding spellings) — recorded as a compiler defect; the shipped transpose covers the explicit-written-as path"
  - "gemm sized [128,128]x[128,96] so the racecheck-instrumented run fits the 300s subprocess timeout"

patterns-established:
  - "Suite entry point: python3.13 testing/orchestrator.py (|--list|--filter|--category|--bench-iters|--sanitizer|--jobs|--keep-builds|--clean|--verbose)"
  - "Every kernel file carries a header comment recording the spike wall(s) that shaped its final form"

requirements-completed: [SUITE-01]

# Metrics
duration: ~3h
completed: 2026-10-28
---

# Quick 261028-t4u: Comprehensive Compiler Testing Suite Summary

**Repo-root `testing/` suite: 14 TCA kernels across 5 categories, all passing 14/14 on the A6000 (canary 0, maxdiff within per-test tolerance vs the in-binary CPU reference), with benchmark and zero-error memcheck/racecheck modes, driven by a stdlib-only Python 3.13 orchestrator with a content-hash cmake build cache.**

## Performance

- **Duration:** ~3 h (GPU battery + compiler-defect probing and reshapes)
- **Started:** 2026-10-28T13:20:00Z (approx)
- **Completed:** 2026-10-28
- **Tasks:** 3/3
- **Files modified:** 34 (31 created + .gitignore + 2 registry/runner revisions)

## Accomplishments

- The full suite is the single user-facing validation entry point:
  `python3.13 testing/orchestrator.py` — 14 passed, 0 failed, exit 0.
- Benchmark mode (`--bench-iters 20`) prints the per-kernel microseconds
  table with nominal GB/s; sanitizer modes run every binary under
  compute-sanitizer and enforce zero-error SUMMARY lines plus the
  numerical checks (memcheck 14/14, racecheck 14/14).
- The suite found and documented TWO genuine compiler defects (below) —
  exactly the validation surface the plan called for.

## Task Commits

1. **Task 1: kernel corpus + registry + runners + cmake template** — `3086e59` (test)
2. **Task 2: orchestrator.py** — `f20738d` (feat)
3. **Task 3: battery + hermetic gate** — `06090c8` (chore, .gitignore), plus the
   defect-driven corpus reshape `ea6d13a` (fix) and the gemm resize `a4652e8` (fix)

## Files Created/Modified

- `testing/orchestrator.py` — stdlib-only suite driver (probe/compile/cache-build/run/verify/report, bench + sanitizer modes)
- `testing/suite.toml` — 14-test registry (dims, tolerance, scalars mirror, bench_bytes)
- `testing/cmake_template.txt` — fixture CMakeLists (mirrors internal/e2e fixtureCMake)
- `testing/runners/common.cuh` — shared prologue (elem formula, CUDA_OK, protocol printers, bench timer)
- `testing/runners/*.cu` (14) — per-kernel GPU launch + host fp32 reference + protocol
- `testing/kernels/**/*.tca` (14) — the corpus (each header records its wall(s))
- `.gitignore` — `testing/.build/`, `testing/__pycache__/`

## Spike walls and fallbacks applied (every one recorded)

**W1 — v0 pkg rungs require at least one reduction before any store**
(`internal/syntax/build/build.go:389`). Pure map-only chains (the plan's
dup_expr, dup_const, transpose, scale_copy elementwise forms) are not
expressible. Fallback: the elementwise maps moved into a root map over a
trailing reduced axis (map|reduce), preserving each kernel's coverage
intent (GVN subexprs, literal reuse, written-as, affine map + tiled store).

**W2 — a piped map requires a preceding reduction**
(`build.go:355`). The plan's axpy3 chained map|map form is not v0.
Fallback: map|reduce|map over a trailing axis (both maps + the chain
composition remain exercised).

**W3 — the mlp piped map with a tensor binding on(bias[j])** passes the
frontend but final-lower rejects it (P-12: moved consumer references a
dimvar whose loop does not enclose the bracket block). The plan's
documented fallback shipped: bias-free GELU MLP, piped map on() empty.

**W4 — rank-2 dynamic intermediate lets are not constructible**
(`"a multiplicative GMEM size is not constructed"`). Walled the
five-let layernorm (d/sq intermediates) and the softmax let e.
Fallbacks: layernorm → RMS-flavored map|reduce|map; softmax restructured.

**W5 — reading a rank-1 dynamic intermediate from a downstream chain**
trips a typeinfer invariant (`"memload operand is not a tensor
parameter"`), and an unused max-reduce let trips final-lower (`"cannot
erase node: 1 use(s) still point at it"`). Walled the softmax
max-subtract + exp/sum/div chain and the documented plain-exp fallback
(its final div is a map-only chain anyway — see W6). Fallback: the
shipped softmax is the exp+sum reduction core with a weight operand.

**W6 — a reduce must drop exactly one axis** (`relocate`, build.go:424),
so no identity-reduce exists; a rank-preserving final map (the softmax
divide) is unexpressible. Folded into W5's fallback.

**W7 — GlobalReducer selection (COMPILER DEFECT — see below).**
Every single-input reduction probed — any axis position, any dims — and
every rank-1-output reduction (single- OR two-input) selects the
v0-unreachable `KindRuntimeGlobalReducer` ("v0-unreachable forward
scaffolding", `internal/ir/catalog.go:70`); the runtime asserts
`"GlobalReducer: not enabled in v0"` at launch. Only two-input
rank-2-output reductions select the WARP class. Fallback: dup_expr,
dup_const, scale_copy, transpose, layernorm, softmax and row_sum carry a
second weight operand z (the attention-mask / score-pooling pattern);
softmax and row_sum moved to rank-2 batched outputs ([R,K,C] -> [R,C],
middle-axis reduce). SeqReducer is consequently never selected anywhere
in the corpus — noted as a coverage gap caused by the defect.

## Compiler defects found (per the defect protocol)

**D1 — schedule selects the v0-unreachable GlobalReducer.**
Legal frontend programs (e.g. the plain single-input row sum
`map on(x[i][j]) | reduce as (i)`) compile cleanly and emit a kernel that
asserts at launch. Repro: any single-input reduce kernel in this corpus,
pre-reshape versions preserved in commit `3086e59`. The IR catalog marks
the kind "v0-unreachable"; `internal/opt/tiling/search.go` admits CTA
digits on reduced dims with no pruning (only WARP has the ReducedDimWarp
gate). The scheduling policy comment in
`internal/opt/finallower/reducer.go:848` ("no CTA-parallel reduced
dims") does not match the implemented feasible area.

**D2 — transpose permutation lowering miscomputes.**
Both spellings of a legal transpose — written `as (j, i, k)` on
`on(x[i][j][k], ...)` and the swapped-binding `on(x[j][i][k]) as (i, j,
k)` — compile but produce identical wrong output. Tiny-case dump
(R=4, C=3, K=2, x == z): expected out[i][j] = Σ_k x[j][i][k]²; observed
`out[j][i] = Σ_k x[0][j+i][k]²` (axis coordinates fold along
anti-diagonals, zeros elsewhere). Probes: `/tmp/t4u-tdbg.sh`,
`/tmp/t4u-tdbg2.sh`. Both forms are legal per doc/syntax-design.md
§2.6/§4.4 rule 11 ("a transpose must be written explicitly").

## Final tolerance per test (with justification)

| test | tolerance | justification |
|---|---|---|
| dup_expr, dup_const, axpy3, scale_copy, transpose, layernorm | 1e-5 | tiny-K (<= 8 term) fp32 tree-vs-sequential reduce + elementwise vocabulary; measured maxdiffs 2.4e-7..9.5e-6 |
| mm_small, mm_mid, mm_splitk, gemm, mlp, qk_score, row_sum, softmax | 2e-3 | reduce-carrying — the proven frontend-matmul bound (internal/e2e reduceTol); measured maxdiffs 4.8e-7..5.3e-5 |

No kernel uses tolerance 0.0 (no bitwise-exact kernel survives the W1
reshape; transpose, the one candidate, is defect-blocked per D2).

## Measured benchmark numbers (--bench-iters 20, A6000)

Note: v0 codegen is heavily latency-bound (per-thread SEQ loops), so
per-kernel times are large and the derived GB/s are NOMINAL
(bench_bytes / elapsed), exceeding device bandwidth for the compute-bound
kernels — the table is for regression tracking, not bandwidth analysis.

| name | avg us | min us | max us | nominal GB/s |
|---|---|---|---|---|
| dup_expr | 1,256,240 | 1,214,650 | 1,280,040 | 3,547,450 |
| dup_const | 1,234,680 | 1,206,470 | 1,346,460 | 3,609,395 |
| axpy3 | 2,447,850 | 2,392,270 | 2,622,790 | 3,855,295 |
| mlp | 50,987 | 48,514 | 55,070 | 1,928,040 |
| softmax | 60,777 | 59,871 | 67,481 | 36,662,230 |
| mm_small | 8,027 | 7,954 | 8,574 | 2,041,063 |
| mm_mid | 193,653 | 191,643 | 199,635 | 1,353,679 |
| mm_splitk | 48,797 | 48,282 | 55,111 | 2,014,567 |
| transpose | 339,513 | 334,733 | 345,115 | 3,691,688 |
| row_sum | 471,440 | 466,740 | 475,525 | 40,035,568 |
| scale_copy | 2,378,580 | 2,373,280 | 2,386,220 | 3,967,571 |
| gemm (resized) | 307,671 | 305,834 | 312,368 | 692,272 |
| qk_score | 122,647 | 121,539 | 128,828 | 2,270,973 |
| layernorm | 492,265 | 488,341 | 496,186 | 34,347,939 |

## Sanitizer results

- **memcheck**: 14/14 PASS, zero errors, numerical checks enforced (152.8 s battery).
- **racecheck**: 14/14 PASS, zero hazards, numerical checks enforced
  (gemm resized after the original [256,128]x[128,192] form exceeded the
  plan-pinned 300 s subprocess timeout at 300.1 s; the resized kernel
  completes in 211 s — a timeout, never a hazard).

## Repo hermetic gate

- `go build ./...` green, `go vet ./...` green, `go list -m all | wc -l` == 1.
- `testing/` holds zero `.go` files; no compiler/runtime/Go source changed.

## Suite mechanics verified

- `--list` prints all 14 tests, exit 0, no tool probing.
- Missing tools → yellow ALL-SKIPPED banner, exit 0 (unit-checked by
  emptying PATH + fallbacks).
- scale_copy smoke: end-to-end PASS; immediate second run reports
  `(cache)` and skips the rebuild; a perturbed reference exits 1 with a
  red FAIL line and the reason.
- `--filter`/`--category`/`--jobs 2`/`--keep-builds`/`--clean`/`--verbose`
  all behave per spec; exit codes 0/1.

## Self-Check

- All 34 files exist (committed across 5 commits: 3086e59, ea6d13a,
  f20738d, 06090c8, a4652e8); no post-commit deletions in any commit.
- Final full-suite line: `14 passed, 0 failed, 0 skipped, total 22.8s`, RC=0.

## Self-Check: PASSED

## Next Phase Readiness

- The suite is the validation harness for roadmap VER-01/03/05/06 and
  NUM-01; it complements the TCACOMP_E2E-gated Go tests.
- Follow-ups for the compiler owners: D1 (GlobalReducer selection) and D2
  (permutation lowering) — both with repros in this summary; once fixed,
  the corpus can be restored toward its plan-original forms (pure
  elementwise maps if the rung wall lifts, true softmax/layernorm chains,
  the SeqReducer row sum, and the real transpose).
