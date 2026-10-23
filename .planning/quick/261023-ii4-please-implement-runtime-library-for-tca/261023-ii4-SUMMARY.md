---
phase: quick-261023-ii4
plan: 01
subsystem: runtime
tags: [runtime, cuda, c++20, cmake, e2e, compute-sanitizer]
requires:
  - doc/rt.md (the implementation-ready runtime design)
  - internal/cg/cxx/testdata/{matmul_store.cu, reduce_warp.cu} (the conformance goldens)
provides:
  - runtime/ header-only C++20 library behind tca/runtime.h + INTERFACE target tcacomp_runtime
  - Host-compilable CPU unit layer (runtime/tests) pinning the doc formulas
  - TCACOMP_E2E-gated Go e2e harness (internal/e2e) with numerical + sanitizer tests
affects: []
tech-stack:
  added: []
  patterns:
    - dual-compile KernelInfo (host members / device gridDim reads under __CUDA_ARCH__)
    - host-compilation shim (__device__/__host__ no-op macros) for plain-g++ CPU tests
    - function-scope static __shared__ reducer scratch per <T,Slots> instantiation
    - content-addressed CMake fixture cache keyed by sha256 of fixture contents
key-files:
  created:
    - runtime/CMakeLists.txt
    - runtime/tca/runtime.h
    - runtime/tca/check.h
    - runtime/tca/tensor.h
    - runtime/tca/launch.h
    - runtime/tca/mapping.h
    - runtime/tca/iterator.h
    - runtime/tca/view.h
    - runtime/tca/loader.h
    - runtime/tca/reducer.h
    - runtime/tests/CMakeLists.txt
    - runtime/tests/cpu_tests.cpp
    - internal/e2e/harness.go
    - internal/e2e/e2e_test.go
  modified: []
decisions:
  - cudaDeviceGetAttribute used in its 3-arg device-explicit form (cudaGetDevice first) — this toolkit's 12.2 headers declare only that signature
  - Tensor template parameter spelled `R` internally — a Rank() method would shadow the template parameter name
  - WarpReducer.Combine folds only the CTA's EXISTING warps (blockDim/32) so never-written scratch rows are never read; scratch capacity stays 16 per the ruling, guarded by assert
  - WarpReducer host branch implements the seq algebra (CPU-testable; nvcc host pass compiles it) — the warp protocol lives under __CUDA_ARCH__
  - e2e numerical tests left failing-as-written (contract-true) until the C1/C2 compiler fixes land; the hermetic gate stays green (tests skip without TCACOMP_E2E=1)
metrics:
  duration: ~50 min
  completed: 2026-10-23
  tasks: 3/3
  commits: [a4db48c, 12266b2, c1b57dd]
---

# Phase quick-261023-ii4 Plan 01: TCA C++ Runtime Library Summary

**One-liner:** Header-only tca/ C++20 runtime (maps/iterators/views/loaders/reducers + host Tensor/ValidateLaunch) with a 131-check CPU unit layer, golden compile conformance, and a TCACOMP_E2E-gated Go harness whose sanitizer runs are clean — with two root-caused compiler-side numerical findings (C1/C2) that the e2e layer exists to expose.

## What Was Built

### Task 1 — runtime library (a4db48c)

10 files per doc/rt.md §1 tree: the umbrella `tca/runtime.h` (only include
generated code needs), host surface (`check.h` TCA_REQUIRE/TCA_CUDA_CHECK/Throw;
`tensor.h` non-owning Tensor<T,R> with From/Data/Dim/Rank/layout; `launch.h`
ValidateLaunch — trap-never-clamp over the 1-D normalized launch grammar,
SMEM ceiling always queried via cudaDeviceGetAttribute, never hardcoded) and
device surface (`mapping.h` KernelInfo dual-compile + portion-decomposition
GridMapping/ThreadMapping + umin/umax; `iterator.h` the uniform round walk;
`view.h` bounds-predicated TensorView (OOB Get→T{}, Set no-op) + SmemView +
Frag over a shared row-major shape; `loader.h` cooperative LoadTileCta
(sync pair around a CTA-striped copy, trailing-axes window contract) +
per-thread LoadTileThread; `reducer.h` SlotValue + Seq/Warp/GlobalReducer
with the function-pointer combiner ctor, flat <T,Slots> addressing, shuffle
down-tree + static-`__shared__` staging + replicated-local-combine
broadcast). INTERFACE target `tcacomp_runtime`, C++20/CUDA20, arch unpinned.

Conformance: both goldens compile as sm_86 objects with
--expt-relaxed-constexpr (the only warning is the golden's own unused `c2`
line 30 — emitted-code finding F0 below, not a library warning); the four
shimmed headers pass plain g++ -std=c++20 -fsyntax-only.

### Task 2 — CPU unit layer + conformance targets (12266b2)

`runtime/tests/cpu_tests.cpp`: plain C++20 runner (no gtest — environment has
none, no network), **131 checks, 0 failures**, pinning the exact plan check
values: GridMapping (-1)@100 and (4,-1)@100 → extents (40,60), coords
(id/60)%40 / id%60; ThreadMapping (4,-1)@512 → dims (256,256), index(0)=
(tid/256)%256, index(1)=tid%256; (-1)@256 minimum config; iterator walks
(16-trip 0/64/1024, nonzero init, empty, single-trip, Cta/Thr identical);
TensorView stride math (2×3×4), OOB suppression + canary no-op + const-T;
SmemView linear (the golden's 16384 shape) + 2-D; Frag; umin/umax;
SeqReducer float,0 / float,32 algebra incl. declared-init fill (2.0f) and
per-slot independence; WarpReducer host-branch algebra. CMake gates
everything behind TCACOMP_RUNTIME_TESTS (default OFF — plain `cmake -S
runtime` stays INTERFACE-only, verified) and builds both goldens as sm_86
OBJECT libraries in the same tree.

### Task 3 — e2e harness + numerical e2e + sanitizers (c1b57dd)

`internal/e2e/harness.go` (stdlib only): TCACOMP_E2E=1 + toolchain gate with
PATH/absolute fallbacks; sha256 content-addressed fixture cache under
os.TempDir/tcacomp-e2e with .done sentinel; cmake configure/build with NO -G
(ninja absent; Unix Makefiles default); 5-min timeouts; fixed-argv runner;
hex-float parser. `internal/e2e/e2e_test.go`: four tests composing each
golden + a runner.cu (pinned launcher signatures, deterministic inputs
v(i)=float((i*2654435761u)%1000u)/1000.0f, canary -12345.0f, %a hex-float
output) + a CMakeLists mirroring matmul_store_cmake.txt.

**Results on the A6000:** both sanitizer tests PASS — memcheck and racecheck
report zero errors/hazards on both fixtures. Both numerical tests FAIL with
compiler-side findings C1/C2 (next section). Hermetic behavior verified:
without TCACOMP_E2E all four tests skip; `go build ./...` and `go test
./...` stay green repo-wide; go vet + gofmt clean.

## Compiler-Side Findings (root-caused, with evidence — NOT runtime defects)

The plan's contingency applied: Task 1/2 gates green, runtime math
CPU-pinned to the doc formulas → the runtime side is pinned. Attribution
was made conclusive by a standalone diagnostic (not committed) driving the
runtime primitives directly with hand-written index arithmetic:
LoadTileCta in BOTH window forms (the golden's 1-D trailing-axis tile of
16384 elems over a (1024,128) source, and a 2-D (256,64) window) — exact;
WarpReducer slot sums over 16 warps — exact (960.00 bitwise) and
deterministic; on-device map queries — equal to the pinned formulas;
memcheck 0 errors, racecheck 0 hazards.

**C1 — matmul golden: output coverage/arithmetic defect.**
`out` canary count = **1,040,187,392 = 2^30 − 32·2^20 exactly**: only
33,554,432 elements written = 16 CTAs × 512 threads × 4096 stores — i.e.
only 32 distinct d0 rows. Cause in the emitted arithmetic:
`v17 = v5 + d0_it3.Index() + v2` with `v2 = thread_map1.index(0)`. Per the
plan-pinned decomposition (CPU-pinned in Task 2, device-pinned by the
diagnostic), ThreadMapping(4,-1)@512 gives index(0) = (tid/256)%256 ∈ {0,1}
— the row-major coordinate of the SLOWEST axis — so each CTA writes only
rows {v5, v5+1} instead of a distinct rank per coupled thread. The coupled
inside walk needs a 0..255 element rank; no ThreadMapping formula supplies
one here, and additionally the between-round base v5 = blockIdx·64 (stride
64) is self-inconsistent with a 256-thread coupled window (adjacent CTAs'
windows overlap by 192 rows — a cross-CTA write race if coverage completed;
today's disjoint {v5, v5+1} rows are why racecheck is clean). Written values
also use lhs row v5 for out rows v5+v2 (lhs row never varies with the out
row). Checksum −1.284e13 vs expected 2.679e8 (canaries dominate).

**C2 — reduce golden: slot mapping / tile-row coverage defect.**
Out rows r mod 32 ∈ {0,1} hold sum(row 32b)+sum(row 32b+1) (≈511 ≈ 2× the
reference ≈256) — both thread halves (v2 ∈ {0,1}, reading lhs rows v5+v2)
fold into slot 0 because the slot operand `d0_it2.Index()` is always 0 (its
loop has step 256 over bound ≤ 32 — a single trip, never enumerating the
32-row tile); out rows r mod 32 ∈ [2,32) hold the never-Reduced init 0.0
(slots 1..31) that the epilogue nevertheless reads via d0_it4 ∈ [0,v8).
Max abs diff 258.67 at row 339 (0 vs 258.67); canary 0 (every row written —
consistent with memcheck/racecheck clean: the failures are coverage and
arithmetic, not memory unsafety).

**F0 — minor, emitted code:** `matmul_store.cu` line 30 declares an unused
`auto c2 = 65536u;` (nvcc warning #177-D). Harmless; noted for the codegen
dead-const cleanup.

**C3 — generated CMakeLists latent configure failure:** `CUDA::cudart`
requires `find_package(CUDAToolkit)` in CMake 4.x — `project(... LANGUAGES
CXX CUDA)` alone does not define the imported target (reproduced: configure
fails at target_link_libraries). The e2e fixture CMakeLists adds it; the
emitter (internal/cg/cxx) should too.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] cudaDeviceGetAttribute is 3-arg in this toolkit**
- **Found during:** Task 1 (launch.h)
- **Issue:** This CUDA 12.2 install's cuda_runtime_api.h:1918 declares only
  `cudaDeviceGetAttribute(int*, cudaDeviceAttr, int device)`; the 2-arg form
  in the plan does not compile.
- **Fix:** `cudaGetDevice(&device)` first, then the 3-arg call (query the
  CURRENT device); semantics unchanged.
- **Files modified:** runtime/tca/launch.h
- **Commit:** a4db48c

**2. [Rule 1 - Bug] Tensor Rank() method shadows its template parameter**
- **Found during:** Task 1 (tensor.h) — the plan flagged the collision risk
- **Issue:** A `Rank()` method makes every use of `Rank` as a value inside
  the class resolve to the member function (nvcc errors on the array bound).
- **Fix:** Template parameter spelled `R` internally (external Tensor<float,
  3u> shape unchanged); `kRank` constant + `Rank()` accessor as planned.
- **Files modified:** runtime/tca/tensor.h
- **Commit:** a4db48c

**3. [Rule 1 - Bug] Annotated defaulted ctors warn under nvcc (#20012-D)**
- **Found during:** Task 1 (mapping.h, tensor.h)
- **Issue:** `__host__ __device__ KernelInfo() = default;` triggers warning
  #20012-D (annotation ignored on a defaulted function) — verify demands
  zero library warnings.
- **Fix:** Dropped the annotations — defaulted functions are implicitly
  dual-execution under nvcc (the warning itself documents this); device
  `KernelInfo kinfo;` still compiles (golden verified).
- **Files modified:** runtime/tca/mapping.h, runtime/tca/tensor.h
- **Commit:** a4db48c

**4. [Rule 3 - Blocking] CUDA::cudart undefined without find_package(CUDAToolkit)**
- **Found during:** Task 2 (also latent in the generated CMakeLists — C3)
- **Issue:** CMake 4.x does not create the CUDA::cudart imported target from
  `LANGUAGES CUDA` alone; configure failed.
- **Fix:** `find_package(CUDAToolkit REQUIRED)` in runtime/tests/CMakeLists.txt
  and in the e2e fixture template.
- **Files modified:** runtime/tests/CMakeLists.txt, internal/e2e/e2e_test.go
- **Commits:** 12266b2, c1b57dd

**5. [Rule 1 - Bug] sanitizer summary parser missed racecheck's wording**
- **Found during:** Task 3
- **Issue:** racecheck prints "RACECHECK SUMMARY: 0 hazards displayed (0
  errors, 0 warnings)" — not "ERROR SUMMARY:".
- **Fix:** Match any "*SUMMARY:" line; require zero errors in any line that
  mentions errors.
- **Files modified:** internal/e2e/e2e_test.go
- **Commit:** c1b57dd

**6. [Rule 2 - Correctness] WarpReducer fold range narrowed to existing warps**
- **Found during:** Task 1 (reducer.h)
- **Issue:** Folding all kMaxWarpsPerCTA=16 scratch rows would read
  never-written rows when blockDim < 512 (e.g. the 256-thread minimum
  config) — undefined values.
- **Fix:** Fold exactly the CTA's existing warps (blockDim.x/32); every
  existing warp staged before the barrier. Capacity stays 16 per the ruling,
  with `assert(warp < kMaxWarpsPerCTA)`.
- **Files modified:** runtime/tca/reducer.h
- **Commit:** a4db48c

## Known Stubs

None. GlobalReducer is a documented v0-unreachable stub per doc/rt.md §4.6
(compiling, traps at runtime) — deliberate design, not an omission.

## Verification Evidence

- Task 1 verify: CONFORMANCE_OK — both goldens → sm_86 objects; four shimmed
  headers → g++ syntax-only clean.
- Task 2 verify: UNIT_TESTS_PASS — cmake configure+build green, 131 checks /
  0 failures; plain INTERFACE-only configure green.
- Task 3 verify: E2E partially per contingency — `TCACOMP_E2E=1 go test`:
  both SANITIZER tests PASS (memcheck + racecheck, both fixtures, exit 0,
  zero errors); both NUMERICAL tests FAIL with compiler-side findings C1/C2
  (root-caused above; runtime exonerated by the primitives diagnostic).
  `go vet` + `gofmt -l` clean; `go build ./...` + `go test ./...` green.

## Self-Check: PASSED

All 14 created files exist (runtime/CMakeLists.txt, 9× runtime/tca/*.h,
runtime/tests/{CMakeLists.txt,cpu_tests.cpp}, internal/e2e/{harness.go,
e2e_test.go}); all three task commits present (a4db48c, 12266b2, c1b57dd).
