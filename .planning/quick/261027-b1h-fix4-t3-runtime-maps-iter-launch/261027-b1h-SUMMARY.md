---
phase: quick-261027-b1h
plan: 01
subsystem: runtime
tags: [cuda, runtime, mapping, factorization, launch-validation, conformance]

# Dependency graph
requires:
  - phase: quick-261026-ky1 (fix4 T1+T2)
    provides: the decide/verify-side factorization model (absolute LoC records, MaxCTA/MaxWarpsPerCTA params) the runtime contract now matches
provides:
  - runtime/tca factorization maps — (kinfo, ndim, extents...) ctors, reverse stride-shape decomposition, WARP-unit ThreadMapping coords, kinfo-derived warp_count()/total()
  - 4-arg ValidateLaunch(grid, block, smemBytes, maxCTA) per fix4 §4.3 (compiler-emitted MaxCTA, whole-warp blockDim, ≤ 32 warps, queried SMEM ceiling)
  - §4.2 seam comments on the unchanged iterator faces (loc = absolute LoC / K_d; coupled walk = flat init + blockDim step from the caller)
  - tca_conformance_new compile-only nvcc OBJECT at sm_86 pinning the exact T4/T5 call shapes
  - §4.5 CPU test layer — 5934 checks / 0 failures
affects: [fix4 T4 (tile-lower map binding), fix4 T5 (codegen launch derivation + golden regen/restore), fix4 T6 (e2e), fix4 T7 (doc pointer stubs)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Derived-launch-value rule: map totals come from KernelInfo (blockDim/gridDim), never the shape product — surplus multi-LNT ranks wrap in-shape, guards skip bodies"
    - "Compile-only conformance OBJECT as the API-freeze gate between runtime and the emitter tasks that target it"

key-files:
  created:
    - runtime/tests/conformance_new_api.cu
  modified:
    - runtime/tca/mapping.h
    - runtime/tca/launch.h
    - runtime/tca/iterator.h
    - runtime/tests/cpu_tests.cpp
    - runtime/tests/CMakeLists.txt

key-decisions:
  - "warp_count()/total() are kinfo-DERIVED launch values, not shape products (fix4 §4.1/§3.2) — no assert ties an LNT's shape product to the launch total; loader striping and reducer staging see the whole CTA even when an LNT factors fewer warps"
  - "ThreadMapping::index(d) decomposes the WARP rank (index()/32) over the K shape — the portion-era x32 thread exposure of dimension() is superseded; no live consumer read it"
  - "ValidateLaunch maxCTA has NO default argument — the compiler must emit its policy constant (T-T3-01); the SMEM ceiling stays a device-attribute query, never hardcoded (T-T3-04)"
  - "Golden conformance targets STOOD DOWN in the T3 commit (committed goldens speak the pre-fix4 transitional ctor); T4/T5 regenerate the goldens and restore the targets — recorded in runtime/tests/CMakeLists.txt"

patterns-established:
  - "Absolute-extent factorization ctor: (const KernelInfo&, unsigned ndim, E... extents) with static_assert(pack >= 1), assert(ndim == sizeof...), extents >= 2 defense-in-depth"
  - "Conformance fixture mirrors the generated launcher: combiner template + anonymous-ns kernel + inline host launcher (ValidateLaunch -> launch -> TCA_CUDA_CHECK)"

requirements-completed: [FIX4-T3]

# Metrics
duration: 35min
completed: 2026-10-27
---

# Quick Task 261027-b1h: fix4 T3 — Runtime maps/iterators/launch flip to the factorization model

**Absolute-extent factorization maps ((kinfo, ndim, extents...) ctors, reverse stride-shape decomposition, WARP-unit thread coords, kinfo-derived warp_count/total), 4-arg ValidateLaunch with the compiler-emitted MaxCTA, §4.2 seam comments on the unchanged iterator faces, a sm_86 nvcc conformance object pinning the T4/T5 call shapes, and the §4.5 CPU decomposition test layer — 5934 checks / 0 failures, runtime/ only.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-10-27 (worktree dispatch at base 02ff7cf)
- **Completed:** 2026-10-27T00:16:34Z
- **Tasks:** 2 / 2
- **Files modified:** 6 (5 modified, 1 created)

## Accomplishments
- fix4 §4.1 in `runtime/tca/mapping.h`: both map ctors take explicit ndim + ABSOLUTE extents (CTA LoC / warp K_d, all >= 2 asserted); Decomposition copies extents verbatim (portion constants, -1 remainder, quantum division, unit multiplier, exposed[]/unit_sum() all deleted); ThreadMapping::index(d) decomposes the warp rank in WARP units; warp_count()/total() answer the kinfo-derived launch values with the §3.2 surplus-rank rationale commented; KernelInfo copy stored on both maps; word "portion" eradicated from runtime/ (grep == 0).
- fix4 §4.3 in `runtime/tca/launch.h`: ValidateLaunch(grid, block, smemBytes, maxCTA) — grid.x <= compiler-emitted maxCTA (no default arg), whole-warp block.x >= 32 with <= kMaxWarpsPerCTA = 32, kMinThreadsPerCTA and the mapping.h include deleted; SMEM ceiling block kept VERBATIM (device-attribute query, never-hardcode rule, off-by-1KB bug note).
- fix4 §4.2 in `runtime/tca/iterator.h`: signatures and walk arithmetic byte-identical; only the file-top and loc/dosoc comments moved to the absolute-LoC/K_d semantics (coupled walk = ThrIterator with init = thread_map.index() flat, step = blockDim from the caller).
- §4.4 untouched, verified: loader.h / reducer.h / view.h / tensor.h / check.h / runtime.h / runtime/CMakeLists.txt show ZERO diff vs base — loader's mapping.index()/mapping.total() striping and reducer's warp queries keep their semantics against the new maps.
- `runtime/tests/conformance_new_api.cu` (new) + CMake swap: tca_conformance_new compile-only nvcc OBJECT at sm_86 pins the exact T4/T5 emission shapes — map ctors, all queries, CTA/WARP round walks + the coupled inside walk, the WarpReducer<float> bracket with a file-scope combiner template, and the §3.3 worked-example launcher (ValidateLaunch(dim3(16,1,1), dim3(64,1,1), 65536, 512) + launch); the two stale-golden OBJECT targets stood down with the T5-restore debt recorded in a CMake comment.
- §4.5 CPU test layer: five new test families (grid factorization bijection, warp-unit thread factorization, flat-rank identities incl. the §3.3 canonical pair, surplus-rank totality on the kinfo-derived counts, absolute-loc iterator walks incl. the coupled-walk single-ownership partition); five portion-era tests deleted; all surviving call sites ported with every existing CHECK kept.

## Task Commits

Each task was committed atomically:

1. **Task 1: Flip maps + ValidateLaunch to the factorization API; port surviving tests; swap conformance target** - `1fa6c28` (feat)
2. **Task 2: §4.5 factorization decomposition tests** - `3a93a4e` (test)

## Files Created/Modified
- `runtime/tca/mapping.h` — factorization maps: absolute-extent ctors, verbatim Decomposition, warp-unit coords, kinfo-derived warp_count()/total()
- `runtime/tca/launch.h` — 4-arg ValidateLaunch (maxCTA, whole warps, ≤ 32 warps); smem query block verbatim
- `runtime/tca/iterator.h` — unchanged faces; §4.2 seam comments only
- `runtime/tests/cpu_tests.cpp` — ported surviving sites + five new §4.5 test families
- `runtime/tests/conformance_new_api.cu` — NEW: compile-only nvcc conformance of the T4/T5 call shapes at sm_86
- `runtime/tests/CMakeLists.txt` — tca_conformance_new OBJECT replaces the two golden OBJECTs; T5-restore note

## Decisions Made
- Followed the plan's contract exactly on ctor shape, decomposition, derived totals, and ValidateLaunch conditions (see key-decisions above); no architectural choices were open beyond the plan.
- Conformance fixture: kept the kernel in an anonymous namespace (generated-code protocol, 261023-ten) and had the host launcher-shape function actually LAUNCH it (validate -> launch -> TCA_CUDA_CHECK, mirroring the golden launcher emission) — pins a strictly more faithful T5 shape than the plan's minimum and keeps the build warning-free.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan's verify command used `-G Ninja`, but ninja is not installed on this host**
- **Found during:** Task 1 (verification)
- **Issue:** `command -v ninja` finds nothing (no dpkg/snap/local install); the plan's automated verify `cmake -S runtime -B /tmp/tca-rt-t3 -G Ninja ...` cannot run as written.
- **Fix:** Configured with CMake's default generator (Unix Makefiles) — identical targets, identical gates; `cmake --build ... -j8` unchanged.
- **Files modified:** none (invocation-only)
- **Verification:** Configure + build + test all green.
- **Committed in:** n/a

**2. [Rule 1 - Bug, plan-arithmetic] Coupled-walk trip-count prose in Task 2 item 5 is imprecise**
- **Found during:** Task 2 (writing test_iterator_absolute_loc_walks)
- **Issue:** The plan says "tile_max 600, blockDim 512: tid 0..511 take one trip (flat < 600)" — but under init=tid/step=512/bound=600, every tid < 88 takes a SECOND trip (tid + 512 < 600). The prose undercounts trips for the first 88 threads.
- **Fix:** Pinned the exact per-tid formula `trips = (tid<600) + (tid+512<600)` over all 1024 injectable tids AND the single-ownership partition of [0,600) by the real 512-thread block (element e -> thread e%512, trip e/512, each covered exactly once) — a strictly stronger pin that subsumes the plan's intent; the discrepancy is documented in a comment at the test and in the Task 2 commit message.
- **Files modified:** runtime/tests/cpu_tests.cpp
- **Verification:** CPU suite 5934 checks / 0 failures.
- **Committed in:** 3a93a4e

**3. [Rule 3 - Blocking] nvcc warning #177-D (kernel declared but never referenced) on the first conformance draft**
- **Found during:** Task 1 (first build)
- **Issue:** A compile-only __global__ in an anonymous namespace triggers the unused warning; done criteria requires a clean build.
- **Fix:** The host launcher-shape function now launches the kernel after ValidateLaunch (mirroring the golden launcher: validate -> launch -> TCA_CUDA_CHECK) — warning gone, emission shape more faithful.
- **Files modified:** runtime/tests/conformance_new_api.cu
- **Verification:** Clean rebuild, zero warnings for the new target.
- **Committed in:** 1fa6c28

---

**Total deviations:** 3 auto-fixed (1 bug/plan-arithmetic, 2 blocking/environment)
**Impact on plan:** No scope creep — all three were required to execute the plan's own gates as written. Code semantics match the plan contract exactly.

## Issues Encountered
- Two pre-existing warnings surfaced in UNTOUCHED cpu_tests.cpp code under -Wall -Wextra (`std::popcount` sign-compare in test_mask_derivation_arithmetic; a maybe-uninitialized note on dstbuf in test_loader_vector_eligibility). Per the scope boundary these are out of scope for T3 — logged here, not fixed.

## Scope Fence (fix4 §8)
- `git diff 02ff7cf..HEAD --stat` shows ONLY the six planned runtime/ files; loader.h/reducer.h/view.h/tensor.h/check.h/runtime.h/runtime CMakeLists are byte-identical (0 diff lines).
- `go build ./... && go vet ./... && go test ./...` hermetic green (all packages ok, e2e skipped without TCACOMP_E2E as designed) — the Go compiler side did not move.
- `go list -m all | wc -l` == 1 (stdlib-only gate).

## Verification Results
- `cmake -S runtime -B /tmp/tca-rt-t3 -DTCACOMP_RUNTIME_TESTS=ON && cmake --build /tmp/tca-rt-t3 -j8` — clean (CPU runner + tca_conformance_new at sm_86).
- `/tmp/tca-rt-t3/tests/tcacomp_cpu_tests` — **5934 checks, 0 failures**, exit 0 (Task 1 commit: 2352/0; count grows at Task 2 as required).
- `grep -rn 'ortion' runtime/ | wc -l` — **0**.
- `git status --porcelain` — only the six runtime/ files (+ .planning artifacts, uncommitted per orchestrator contract).
- Hermetic go gate — build/vet/test green, module count 1.
- Manual spot-check — conformance pins `GridMapping(kinfo, 2u, 4u, 16u)`-style ctor, `ThreadMapping(kinfo, 2u, 2u, 8u)`, the WarpReducer<float> ctor bracket with file-scope combiner, and 4-arg `ValidateLaunch(dim3(16u,1u,1u), dim3(64u,1u,1u), 65536u, 512u)` at lines 25/26/67/86.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- T4 (tile-lower §3.2 map binding): emit `GridMapping(kinfo, <m, LoC…>)` / `ThreadMapping(kinfo, <n, K…>)` against these ctors; the coupled inside walk is `ThrIterator(map, dosoc, K, thread_map.index(), blockDim, tile_max)`.
- T5 (codegen/final-lower §5.1): derive gridDim = max-over-LNT Π LoC, blockDim = (max Π K)·32, emit `__launch_bounds__`, the 4-arg ValidateLaunch, and regenerate both goldens — then RESTORE the golden conformance targets (debt recorded in runtime/tests/CMakeLists.txt).
- Known debt until T4/T5: `internal/cg/cxx/launch.go` still emits the 3-arg ValidateLaunch and the goldens still speak the old ctor — expected sequencing per fix4 Wave 1, NOT a defect of this task.

## Self-Check: PASSED
- Files exist: runtime/tca/mapping.h, runtime/tca/launch.h, runtime/tca/iterator.h, runtime/tests/cpu_tests.cpp, runtime/tests/conformance_new_api.cu, runtime/tests/CMakeLists.txt — all FOUND.
- Commits exist on branch worktree-agent-a00670c6a31964efb: 1fa6c28 (Task 1), 3a93a4e (Task 2) — FOUND.

---
*Phase: quick-261027-b1h (fix4 §8 T3)*
*Completed: 2026-10-27*
