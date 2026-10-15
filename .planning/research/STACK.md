# Stack Research

**Domain:** CUDA kernel compiler written in pure Go, emitting CUDA C++ for a C++20 runtime
**Researched:** 2026-10-15
**Confidence:** HIGH (most claims verified against NVIDIA/CMake/GCC official docs or multiple agreeing sources; a few flagged MEDIUM)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Go | 1.26.2 | Compiler implementation (lexer, IR, passes, codegen, CLI) | Hard project constraint: pure stdlib, no cgo, no external modules. Go's own compiler toolchain (go/parser, go/ast, go/types, cmd/compile/internal/ssa) is an in-repo, world-class reference for every compiler subsystem we need. |
| nvcc (CUDA toolkit) | 12.2 | Compiles generated `.cu` device/host code; PTX/SASS for sm_86 | Fixed by target environment. CUDA 12.2 supports `--std=c++20` on both host and device; supported g++ range is 6.x–12.2 so g++ 11.4 is inside the supported window. |
| g++ (host compiler) | 11.4 | Host-side compilation of generated `.cu` TU and the C++20 runtime | Fixed by target environment and inside CUDA 12.2's supported host-compiler range. C++20 core language is effectively complete in GCC 11; see "What NOT to Use" for library gaps. |
| CMake | 4.4.3 | Builds the C++20 runtime library; drives nvcc for e2e test fixtures | Fixed by target environment. First-class CUDA language support (`LANGUAGES CXX CUDA`, `CUDA_ARCHITECTURES`, `CUDA_SEPARABLE_COMPILATION`, `gtest_discover_tests`) since 3.18/3.24 — mature well before 4.x. |
| Ninja | latest available | CMake generator for fast rebuild cycles of the runtime | Available on the dev machine; standard pairing with CMake for local iteration. |
| CUDA runtime API + headers | 12.2 | `cooperative_groups`, `cuda_pipeline_primitives` (`__pipeline_memcpy_async`), `cudaFuncSetAttribute`, launch APIs | The sm_86 async-copy ceiling is `cp.async` (sm_80+, PTX ISA 7.0); TMA/`cp.async.bulk` is sm_90+ and out of scope. All required facilities exist in CUDA 12.2 headers. |
| Graphviz (`dot`) | any | IR visualization from the `internal/visualization` package | `internal/visualization` only needs to *emit* `.dot` text; rendering is a dev tool, not a dependency. |

### Go Standard Library Packages (the entire compiler dependency set)

| Package | Used For | Notes |
|---------|----------|-------|
| `go/scanner`, `go/token`, `go/ast` | **Reference implementations only** (not imported into our lexer) | These are how a production-grade Go lexer/scanner/AST/visitor are structured; our `internal/syntax` mirrors their architecture (scanner struct, token constants, error handler callback). Importing go/* would drag Go-language semantics; copy the pattern, not the code. |
| `strconv` | Numeric literal parsing in the lexer | `ParseInt`/`ParseFloat` with explicit base/size arguments; fail-fast with position-attributed errors. |
| `strings`, `bytes` | Buffer-based codegen emission | `strings.Builder` for the CUDA C++ emitter; `bytes.Buffer` where seeking/inspection is needed. |
| `fmt` | Diagnostics, logging, `.dot` emission | `fmt.Fprintf` into builders; do NOT use fmt for the hot emission path once profiling says so (unlikely to matter here). |
| `os`, `os/exec`, `path/filepath` | File emission and e2e test driver | `os/exec` runs nvcc/CMake from Go tests; see Build Integration below. |
| `testing` | All tests | Stdlib-only by definition. |
| `sort`, `container/heap`, `container/list` | Pass-manager ordering, worklists | Enough for topological pass ordering and DCE worklists; no graph library needed. |
| `hash`, `maps`, `slices` | Caching/canonicalization keys | Go 1.21+ generic `maps`/`slices` utilities; FNV via `hash/fnv` for node canonicalization. |
| `log/slog` | Compiler log infrastructure (`internal/log`) | Structured logging is a better starting point than hand-rolled levels; wrap behind our own `internal/log` facade so pass code never imports slog directly. |
| `errors`, `fmt.Errorf` | Error plumbing in diagnostics | Position-attributed errors defined in `internal/diagnostic`. |
| `flag` | CLI in `cmd/` | Stdlib flag parsing is austere but sufficient for a v0 compiler CLI. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `compute-sanitizer` | Memory/race checking of generated kernels in e2e tests | Part of CUDA 12.2 toolkit; run the compiled test binary under it for memcheck/racecheck on the A6000. |
| `nsys` / `ncu` | Profiling generated kernels | Available per PROJECT.md; use sparingly, per scalfold methodology. |
| `dot` (Graphviz) | Rendering IR dumps from the visualization pass | Dev-only; compiler emits text. |

## (a) Lexer/Parser Construction — Decision

**Decision: hand-written recursive-descent parser with precedence-climbing expression parsing, one-token lookahead, mirroring `go/parser`'s architecture.**

Rationale:
- **External parser generators are all disqualified.** goyacc (and any yacc/bison derivative) is an external tool; ANTLR is Java/external; tree-sitter is C with cgo bindings. The stdlib-only constraint forces hand-written — which is also what LLVM and GCC now use in production.
- The TCA grammar (`doc/syntax.md`) is small, LL-friendly, and has no left recursion after standard elimination (`A → A α | β` becomes `A → β (α)*`). Recursive descent is the lowest-complexity, best-error-message option for this grammar class.
- **Do not use `text/scanner`** for the lexer: it is Go-literal-oriented (semicolon insertion assumptions, Go identifier rules, rune literals) and its defaults fight a C++-style language. Hand-roll the lexer as a direct port of `go/scanner`'s structure: a scanner struct over the source string, `next()`/`peek()` rune reading, position tracking from the start, error-handler callback.
- **Expression parsing: precedence climbing** (each precedence level a method, tightest called first; left-assoc = loop, right-assoc = recurse). This is the Crafting Interpreters / production-Go pattern and avoids a Pratt-parser's operator-table indirection for a language with a handful of levels.
- AST nodes: Go interfaces with concrete structs per node (Go has no sum types), with a typed-`switch` visitor — exactly `go/ast.Visitor`'s shape. This lands in `internal/syntax`.
- **Confidence: HIGH** (grammar is small and LL; constraint eliminates alternatives; pattern is proven by go/parser itself).

## (b) IR/SSA Data Structures — Decision

**Decision: mutable, pointer-based SSA graph with explicit intrusive use-lists (LLVM-style), exactly as design.md mandates — SSADef/SSAUse sharing an SSAVar with use-def AND def-use chains. Mutations go through IR methods (`ReplaceAllUsesWith`, `RemoveUse`, `Erase`), never direct pointer surgery by passes.**

Rationale and corroborating evidence:
- design.md already prescribes this ("IR must be carefully designed to support optimization mutation", "Change def value thus making all ref change accordingly", "Delete"). Mutable-with-use-lists is the only design that satisfies "change a def and have all refs update" without full IR rebuilds.
- `llir/llvm` (github.com/llir/llvm) demonstrates the pure-Go version of LLVM's model: values own ordered operand slices (use-def), and each use knows its parent (def-use); `ReplaceAllUsesWith` walks the use list. It is an **external module — banned as a dependency — but the definitive pure-Go design reference.**
- `golang.org/x/tools/go/ssa` is **not** a good structural model: it is analysis-oriented, lazily built, models Go semantics (channels, defer, interfaces), and lacks the fine-grained mutation API a transformation-pass compiler needs. Also external/banned. Read it only for how a Go SSA *types* its values.
- The Go compiler's `cmd/compile/internal/ssa` is the right second reference: mutable values with integer IDs inside blocks, passes rewrite in place, and a `check` verifier pass runs between phases. Copy two of its habits: (1) **unique integer IDs per IR node** (stable across mutations, essential for pass debugging, `.dot` dumps, and verifier messages), (2) **a cheap IR verifier callable after every pass** — which design.md's "verification pass per transformation pass" requirement formalizes.
- Fil Pizlo's SSA design notes (JSC/WebKit) argue for *dropping* use lists during construction; that is an optimization for massive-scale JIT construction we don't need, and it contradicts design.md's mutation requirements. Note it, don't follow it.
- Since v0 has exactly **one basic block per kernel and no phis**, we skip phi construction entirely — no FwdRef machinery, no dominance frontiers until (if ever) control flow arrives. CFG = one BB; SSA is trivially satisfiable; the use-list machinery is what matters.
- Single-tier IR (no high/mid/low): lowering introduces node kinds into the same suite — so node kinds need a stable, versioned `Kind` enumeration from day one, and the pass manager should walk kinds it knows and reject kinds it doesn't (forward-compatibility within our own pipeline).
- **Confidence: HIGH** for the mutable pointer-graph decision (design doc + llir/llvm + go compiler agree); MEDIUM on fine-grained API shape (ours to design in `doc/tca.md` terms).

## (c) CUDA Codegen + Runtime Specifics for sm_86 / CUDA 12.2

Verified facts that drive runtime/codegen design:

1. **`cp.async` is available and is the async-copy ceiling.** PTX `cp.async` requires sm_80+; usable from CUDA 12.2 via `cuda_pipeline_primitives.h` (`__pipeline_memcpy_async`, `__pipeline_commit`, `__pipeline_wait_prior`) or `cooperative_groups::memcpy_async`. Sizes 4/8/16 bytes, gmem→smem only. (MEDIUM-HIGH; PTX ISA facts are stable, fetched via community sources — reconfirm exact intrinsic signatures against the 12.2 headers during runtime implementation.)
2. **Dynamic SMEM >48 KB requires opt-in.** Kernels needing more than 48 KB dynamic shared memory must call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` once per kernel before launch, and allocate via `extern __shared__`. **On sm_86 the per-block opt-in maximum is 101,376 bytes (99 KB), not 163 KB** — the 160 KB figure is sm_80 (A100). Per-SM total on sm_86 is 102,400 bytes (100 KB) with 1 KB reserved per block. **The design's 64 KB SMEM+FRAG budget exceeds 48 KB, so the opt-in path is mandatory, not optional.** (HIGH — Stack Overflow/NVIDIA forum sources agree; verify at runtime with `cudaDevAttrMaxSharedMemoryPerBlockOptin` instead of hardcoding.)
3. **`cudaOccupancyAvailableDynamicSMemPerBlock` had a confirmed off-by-1KB bug on sm_86** (NVBUG 5209234: failed to subtract reserved 1 KB, halving cooperative occupancy); fixed in a CUDA 13.1-bundled driver. Driver here is 580.178 (CUDA-13 capable), so likely fixed — but compute the limit defensively via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` + `cudaDevAttrReservedSharedMemoryPerBlock` rather than trusting that one API. (MEDIUM — forum-confirmed bug; treat the workaround as the default code path.)
4. **`cooperative_groups` is fully present in 12.2** (`<cooperative_groups.h>`: `this_thread_block()`, `tiled_partition<N>()` — whole-warp `tiled_partition<32>` matches the whole-warp preset rule). **Grid-level `grid_group::sync()` requires `cudaLaunchCooperativeKernel`** — not needed in v0: S6 runs full reductions inside one CTA, so `this_thread_block().sync()` (i.e., `bar.sync` / `__syncthreads()`) suffices, and cluster-per-kernel avoids cooperative-launch bookkeeping entirely. **Do not pull `cudaLaunchCooperativeKernel` into v0.**
5. **Launch APIs:** generated host stubs use `<<<grid, block, smemBytes, stream>>>` triple-chevron on `extern "C"`-style entry points (runtime validates extents at launch time per PROJECT.md). Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for launch-config validation, and standard `cudaGetLastError()`/`cudaDeviceSynchronize()` error checking in the runtime's launch wrapper.
6. **C++20 device code in nvcc 12.2: `--std=c++20` is supported for host and device, with caveats.** Coroutines are unsupported in device code; modules are unsupported anywhere; and NVIDIA notes C++20-mode regressions in nvcc 12.0–12.4 fixed in 12.4.1+ (e.g., `requires` on special member functions). **Policy: the runtime and generated device code restrict themselves to a conservative C++20 subset** — templates, `constexpr`/`if constexpr`, concepts OK in host code, `auto`, lambdas — and avoid device-side concepts/requires and anything from the C++20 library that GCC 11 lacks. Practically: write device code as C++17-plus-`if constexpr`. (HIGH for the support-level facts from NVIDIA docs; MEDIUM on exactly which device-side C++20 corners are buggy in 12.2 — the fix-in-12.4.1 note comes from a community-maintained compiler matrix.)

## (d) Build Integration — Decision

**Decision (confirmed, already logged in PROJECT.md): the Go compiler emits `.cu` + runtime files + host stubs and stops. It never invokes nvcc. The test harness drives nvcc via `os/exec`.**

Concretely:
- **Runtime library** (`runtime/`): CMake 4.4.3, `project(... LANGUAGES CXX CUDA)`, `CMAKE_CUDA_ARCHITECTURES` pinned to `86` (single-target A6000 — do NOT use `native`, which bakes the dev machine into reproducibility), `CMAKE_CUDA_STANDARD 20`, `CUDA_SEPARABLE_COMPILATION ON` for the library, installed/exported as a static library target the generated code links against.
- **E2E Go tests**: a shared test helper (one file in `internal/`, e.g. `testutil`) that (1) writes generated `.cu` + a tiny generated `CMakeLists.txt` (or a checked-in fixture harness) to `t.TempDir()`, (2) runs `cmake -G Ninja` + `ninja` via `os/exec`, (3) runs the produced binary, (4) compares against the CPU reference computed in Go. Guard with an env var (`TCACOMP_E2E=1`) and `t.Skip` when nvcc/GPU is absent, so unit-test runs stay hermetic.
- **Emit `.cu` + CMake, not raw nvcc, in the compiler's default output** — the user keeps build control (PROJECT.md decision), and CMake handles `-arch=sm_86`, host-compiler selection (`-ccbin`), and future multi-kernel builds for free.
- Optional nvcc smoke flags for the harness: `-arch=sm_86 -std=c++20 --expt-relaxed-constexpr -lineinfo` (`-lineinfo` for `compute-sanitizer`/ncu correlation; `--expt-relaxed-constexpr` only if generated constexpr math needs it — prefer avoiding the flag).
- **Confidence: HIGH** (mechanics are standard; the decision itself is already a logged project decision).

## Installation

```bash
# Go compiler module — no third-party dependencies to install.
# go.mod (Go 1.26.2) with zero external requires, enforced by code review
# and optionally by a CI check: `go list -m all | wc -l` == 1 (stdlib only
# appears implicitly; the module itself is the only entry).

# Runtime library (from runtime/)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_STANDARD=20
cmake --build build

# E2E gate (from repo root, GPU present)
TCACOMP_E2E=1 go test ./...
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Hand-written recursive-descent parser | Table-driven LR/LALR (goyacc-style) | Never under stdlib-only (external generator); would only matter for a much larger grammar with heavy ambiguity. |
| Mutable pointer SSA with intrusive use-lists | Immutable IR with copy-on-write passes | Only if passes were few and whole-graph snapshots were wanted — design.md explicitly requires mutation and RAUW-style updates, so immutability fights the requirements. |
| `cp.async` intrinsics for tile loads | `cuda::memcpy_async` (libcu++ / CCCL) | CCCL is header-only and ships with CUDA 12.2 — usable; prefer it if/when pipeline-group abstractions are wanted. For v0 reducers/tiles, the lower-level `__pipeline_*` intrinsics give explicit control matching scalfold's style. (MEDIUM confidence — validate CCCL 12.2 header availability in the runtime's CMake check.) |
| Plain `<<<>>>` launches | `cudaLaunchCooperativeKernel` | Only when multi-CTA grid sync lands (split-kernel / CTA-reduction future work). v0's single-CTA S6 reduction doesn't need it. |
| CMake-driven e2e fixtures | Go test invokes nvcc directly (no CMake) | Acceptable for a single-file smoke test; prefer CMake fixtures once a generated kernel links the runtime library, to exercise the real user build path. |
| `slog` behind `internal/log` facade | Hand-rolled log levels | If `slog`'s API surface becomes friction for compiler-style verbose-pass logging; facade makes swapping cheap. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| goyacc / any yacc-bison generator | External tool dependency violates stdlib-only; error-quality control is poor | Hand-written recursive descent (go/parser pattern) |
| tree-sitter / ANTLR / PEG libraries | tree-sitter is C + cgo; ANTLR is external/Java; all violate constraints | Hand-written recursive descent |
| `text/scanner` as our lexer | Go-semicolon and Go-literal assumptions baked in; position/error model fights C++-style syntax | Hand-rolled lexer copying go/scanner's *structure* |
| `cgo` for anything (e.g., calling nvrtc/libcuda directly) | Explicit project constraint; also destroys the single-binary compiler story | `os/exec` for toolchain driving |
| llir/llvm, tinygo's IR, x/tools/go/ssa as dependencies | External Go modules — banned; tinygo additionally wraps LLVM (cgo) | Read as design references only |
| TMA / `cp.async.bulk(.tensor)` / cluster features | sm_90+ hardware; A6000 is sm_86 | `cp.async` (Ampere ceiling) |
| `cudaLaunchCooperativeKernel` in v0 | Unneeded for single-CTA reductions; adds launch-bounds bookkeeping | `<<<>>>` + `__syncthreads()` |
| `std::format`, `std::ranges` heavy use, C++20 modules in runtime/device code | Missing/partial in GCC 11; modules unsupported by nvcc anywhere | printf-style/ostream diagnostics (host only); C++17+`if constexpr` device subset |
| Device-side coroutines, device-side concepts/`requires` | Unsupported in device code (coroutines); nvcc 12.0–12.4 C++20 regressions fixed 12.4.1 | Host-only C++20 features |
| `cudaOccupancyAvailableDynamicSMemPerBlock` as sole smem-limit source | Confirmed off-by-1KB bug on sm_86 (NVBUG 5209234) on drivers before the 13.1 fix | Query `cudaDevAttrMaxSharedMemoryPerBlockOptin` + reserved-per-block; validate with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| `CMAKE_CUDA_ARCHITECTURES native` | Bakes dev machine into builds, breaks reproducibility on any other host | Pin `86` (single-target hardware) |
| Hardcoding 163 KB as the sm_86 SMEM opt-in max | That is sm_80 (A100); sm_86 max is 101,376 B/block | Query device attributes at runtime; assert ≥ design's 64 KB budget |

## Stack Patterns by Variant

**If a future milestone adds control flow to the frontend:**
- Add phis and a real CFG to the same node suite (design's single-tier rule survives); implement dominance (Lengauer-Tarjan) in `internal/analysis`.
- Because this contradicts nothing in the current use-list design, the SSAVar machinery carries over unchanged — only construction gets a phi-insertion step.

**If generated kernels must run on non-A6000 hardware later:**
- Parameterize codegen over compute capability with `86` as the floor feature set; guard `cp.async` behind a capability check emitted into the runtime, and keep the `CUDA_ARCHITECTURES` pin at `86` until a second target actually exists.

**If CCCL/libcu++ proves available and useful in 12.2:**
- Prefer `cuda::memcpy_async`/pipeline abstractions over raw `__pipeline_*` PTX intrinsics in the cooperative tile load/store; this is an internal runtime decision invisible to generated code.

**If e2e test times become painful:**
- Cache compiled fixtures keyed by a hash of the generated source (content-addressed `t.TempDir()` alternative under `os.TempDir()`), and keep `go test` unit layers free of any exec calls.

## Version Compatibility

| Component | Compatible With | Notes |
|-----------|-----------------|-------|
| CUDA 12.2 nvcc + g++ 11.4 | Supported: g++ 6.x–12.2 on x86_64 | 11.4 is comfortably in-range; do not upgrade g++ past 12.2 without upgrading CUDA (12.2 rejects newer g++ without `--allow-unsupported-compiler`). |
| CUDA 12.2 + `--std=c++20` | Device C++20 supported with caveats | Coroutines (device) and modules (anywhere) unsupported; prefer C++17-style device code + `if constexpr`. |
| CMake 4.4.3 + CUDA 12.2 | Fully compatible | `CUDA_ARCHITECTURES` (3.18+), `native` (3.24+), `check_language(CUDA)` all predate 4.x. |
| CMake 4.x policy | `cmake_minimum_required(VERSION 4.4)` or `3.31` with policy max | CMake 4.x removed compatibility with `<3.5`; keep `cmake_minimum_required` at 4.x to avoid deprecation warnings. |
| GCC 11.4 C++20 | Core language ~complete; library gaps: `std::format` (needs GCC 13), partial `std::ranges` factories | Keep runtime diagnostics off `std::format`; `std::ranges` views/algorithms present are usable but unnecessary. |
| Go 1.26.2 + stdlib-only | Self-consistent | Use `slices`/`maps` generics (1.21+), `slog` (1.21+); no version-sensitive stdlib features beyond that. |
| Driver 580.178 + CUDA 12.2 runtime | Forward-compatible (driver is CUDA-13 capable) | Runtime compiled against 12.2 works on newer drivers; the occupancy-API bug fix rides the driver, so 580.178 likely includes it — keep the defensive path anyway. |

## Sources

- [CUDA C++ Programming Guide — C++ Language Support appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html) — C++20 host/device support level, coroutine/module limits (HIGH)
- [NVIDIA CUDA 12.2 Linux Installation Guide](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-installation-guide-linux/) — supported host compiler ranges for 12.2, C++20 minimums (HIGH)
- [ax3l's CUDA compilers gist](https://gist.github.com/ax3l/9489132) — nvcc 12.0–12.4 C++20-mode regressions fixed in 12.4.1 (MEDIUM, community-maintained matrix)
- [NVIDIA Forums — cudaOccupancyAvailableDynamicSMemPerBlock returning incorrect value](https://forums.developer.nvidia.com/t/cudaoccupancyavailabledynamicsmemperblock-returning-incorrect-value/329573) — confirmed sm_86 off-by-1KB bug, NVBUG 5209234, driver-side fix in 13.1 (MEDIUM)
- [Stack Overflow — Using maximum shared memory in CUDA](https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda) — sm_86 per-block opt-in max 101,376 B, per-SM 102,400 B, 1 KB reserved (HIGH, multiple agreeing sources)
- [CUDA Runtime API — Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html) — `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyAvailableDynamicSMemPerBlock` semantics (HIGH)
- [CCCL — cuda::memcpy_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html) — CCCL async-copy abstraction (MEDIUM)
- [CUDA pipelines cheatsheet](https://cppcheatsheet.com/notes/cuda/cuda_pipelines.html) — `__pipeline_memcpy_async` intrinsic family (MEDIUM)
- [McGill CS520 parsing slides (2024/2026)](https://cs.mcgill.ca/~cs520/2026/slides/parsing-bottom-up.pdf) — LLVM/GCC both hand-written recursive descent today (HIGH)
- [Go compiler thesis (Twente)](https://essay.utwente.nl/fileshare/file/107354/van den Berg_BA_EEMCS.pdf) — stdlib text/scanner + predictive recursive descent pattern in Go (MEDIUM)
- [rd — recursive descent builders in Go](https://github.com/shivamMg/rd) — canonical Enter/Exit/Match parser-builder pattern (MEDIUM)
- [Fil Pizlo — How I implement SSA form](https://gist.github.com/pizlonator/cf1e72b8600b1437dda8153ea3fdb963) — SSA design tradeoffs (use lists, phi/upsilon) (MEDIUM)
- [Understanding the Go Compiler: The SSA Phase](https://internals-for-interns.com/posts/the-go-ssa/) — cmd/compile/internal/ssa structure, FwdRef/phi insertion, inter-pass verification (MEDIUM)
- [ASU LLVM resources — Def-Use & Use-Def chains](https://labs.engineering.asu.edu/mps-lab/resources/llvm-resources/llvm-def-use-use-def-chains/) — LLVM use-list model our IR mirrors (HIGH)
- [llir/llvm](https://github.com/llir/llvm) — pure-Go LLVM IR library; design reference (dependency banned) (HIGH as reference)
- [CMake CUDA_ARCHITECTURES docs](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) — `native`/`all` semantics, CMP0104 (HIGH)
- [JLab CMake roundtable slides](https://indico.jlab.org/event/420/contributions/7961/attachments/6507/8734/CMakeSandCroundtable.slides.pdf) — check_language(CUDA), separable compilation, FindCUDAToolkit best practices (MEDIUM)
- [CMake GoogleTest module](https://cmake.org/cmake/help/latest/module/GoogleTest.html) — `gtest_discover_tests` for CUDA test executables (HIGH)
- [GCC 11 release notes](https://gcc.gnu.org/gcc-11/changes.html) and [GCC C++20 status](https://gcc.gnu.org/projects/cxx-status.html) — C++20 feature completeness in host compiler (HIGH)
- [cppreference — C++20 compiler support](https://en.cppreference.com/w/cpp/compiler_support/20) — cross-compiler C++20 feature matrix (HIGH)

---
*Stack research for: pure-Go CUDA kernel compiler emitting CUDA C++ (sm_86, CUDA 12.2, g++ 11.4, CMake 4.4.3, Go 1.26.2)*
*Researched: 2026-10-15*
