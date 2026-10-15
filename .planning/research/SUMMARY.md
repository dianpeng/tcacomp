# Project Research Summary

**Project:** tcacomp — pure-Go CUDA kernel compiler (TCA frontend → single-tier SSA IR → CUDA C++ + C++20 runtime)
**Domain:** Compiler construction / GPU DSL (RTX A6000, sm_86, CUDA 12.2)
**Researched:** 2026-10-15
**Confidence:** HIGH overall (STACK verified against NVIDIA/CMake/GCC official docs; ARCHITECTURE patterns from LLVM/MLIR/Triton/Halide are stable and well-sourced; FEATURES gap analysis is first-party against in-repo authoritative docs; PITFALLS hardware facts corroborated by multiple sources on identical hardware)

## Executive Summary

tcacomp is a correctness-first optimizing compiler: a tiny C++-style tensor kernel language is elaborated into a single-tier SSA IR (one basic block per kernel, no control flow), run through a fixed pass pipeline (simplify → merge → fusion → linearization → parallelization presets → iteration/reduce/cache lowering), and emitted as CUDA C++ linking a C++20 runtime. Research confirms the existing 5-phase correctness-first plan (IR core → naive codegen+runtime → parallelization → fusion/caches → frontend) is the right shape — it matches how LLVM/MLIR/Triton are structured ("the pipeline is the product") and how Halide proved schedule/algorithm decoupling. Every phase lands e2e-verifiable on the A6000, which is what makes the flagship gate (`matmul_scale` matching a CPU reference) a per-phase signal rather than a final exam.

The stack is fully determined and low-risk: pure Go stdlib (hand-written recursive-descent parser mirroring `go/parser`, mutable pointer SSA with intrusive use-lists mirroring llir/llvm and `cmd/compile/internal/ssa`), nvcc 12.2 + g++ 11.4 + CMake 4.4.3 pinned to `sm_86`. The genuinely load-bearing design risks are internal, not external: (1) keeping the single-tier IR from becoming a soup of abstract and lowered nodes — solved by a legality-window discipline transplanted from MLIR's legality model (node families + per-stage windows checked by the pass manager); (2) making "verifier after every pass" structural rather than conventional — solved by MLIR-style pass instrumentation where the PM runs verification, dumps, and timing as callbacks; (3) the correctness infrastructure (diagnostics, determinism, tolerance policy, sanitizer stage) must be priced into Phases 1–2, because retrofitting it after parallelization lands is where real projects lose months. The specs (`doc/syntax.md`, `doc/tca.md`) are strong but leave concrete gaps the roadmap must close: pinned FP policy, comparison tolerances, CLI surface, pass-dump flags, and the Loc decision.

## Key Findings

### Recommended Stack

Pure Go stdlib-only compiler (the hard constraint) is not a limitation — Go's own toolchain (`go/scanner`, `go/parser`, `cmd/compile/internal/ssa`) is a world-class in-repo reference for every subsystem. Key decisions, all HIGH confidence:

- **Parser:** hand-written recursive descent with precedence climbing, one-token lookahead, mirroring `go/parser` architecture. External generators (goyacc, ANTLR, tree-sitter) are all disqualified by the stdlib-only rule. Do not use `text/scanner` for the lexer — copy `go/scanner`'s *structure* instead.
- **IR:** mutable pointer-based SSA with explicit intrusive use-lists (LLVM-style), mutation only through a choke-point API (`ReplaceAllUsesWith`, `RemoveUse`, `Erase`). Single basic block per kernel means **no phis in v0** — the use-list machinery is what matters. `llir/llvm` and `x/tools/go/ssa` are banned as dependencies but are the definitive pure-Go design references (read-only). Single-tier IR requires a **versioned, stable `Kind` enumeration from day one** and pass-manager rejection of unknown kinds.
- **CUDA/runtime facts (sm_86, CUDA 12.2):** `cp.async` is the async-copy ceiling (TMA is sm_90+); dynamic SMEM >48 KB requires per-kernel `cudaFuncSetAttribute` opt-in and the A6000 ceiling is **101,376 B/block (99 KB), not 164 KB** (that's A100/sm_80) — the design's 64 KB SMEM+FRAG budget *exceeds 48 KB, so opt-in is mandatory*; `cudaOccupancyAvailableDynamicSMemPerBlock` has a confirmed sm_86 off-by-1 KB bug — compute the limit defensively from device attributes; `cooperative_groups` is fully present but grid-sync needs cooperative launch — **do not pull `cudaLaunchCooperativeKernel` into v0** (S6 runs reductions inside one CTA). C++20 device code: supported with caveats — write device code as "C++17 + `if constexpr`", avoid device-side concepts/requires, coroutines, modules, `std::format`, heavy `std::ranges`.
- **Build integration (already a logged project decision):** the Go compiler emits `.cu` + runtime files and **never invokes nvcc**. Runtime builds with CMake (`CUDA_ARCHITECTURES 86` pinned — never `native`; `CUDA_SEPARABLE_COMPILATION ON`). E2E harness: a Go `testutil` writes generated sources to `t.TempDir()`, drives `cmake -G Ninja` via `os/exec`, runs the binary, compares against a CPU reference computed in Go — gated by `TCACOMP_E2E=1`, `t.Skip` when no GPU.

**Core technologies:** Go 1.26.2 (compiler; zero external modules, enforce with `go list -m all | wc -l` == 1) · nvcc 12.2 / g++ 11.4 (fixed by environment; g++ 11.4 inside CUDA 12.2's supported host range) · CMake 4.4.3 + Ninja (runtime; CUDA language support mature since 3.18/3.24) · CUDA 12.2 runtime headers (`cooperative_groups`, `cuda_pipeline_primitives`, `cudaFuncSetAttribute`) · Graphviz `dot` (dev-only; compiler only emits `.dot` text).

### Expected Features

**Table stakes (the trust infrastructure — all P1, all currently MISSING from the specs):**
- Canonical textual IR printer (deterministic ordering) — substrate for golden tests
- Golden IR regression tests (FileCheck-style patterns + update script)
- Pass dump/trace flags: `--dump-after`, `--dump-graphviz`, `--pass-timing`, `--verify-each=0` escape hatch (MLIR convention)
- Pinned numeric policy: IEEE default, fmad documented as permitted, ftz off, no silent precision reduction (spec has a CONFLICT — punts to "the compiler's numerical policy" which is never stated)
- e2e driver (compiler → nvcc build → GPU run → f64 CPU reference → allclose with per-type tolerances; multi-shape × multi-seed matrix)
- compute-sanitizer as a required harness stage (racecheck/synccheck/memcheck on small shapes) — currently downgraded to "may be used for debugging" in PROJECT.md; upgrade it
- CLI surface (`--emit-ir/--emit-cuda`, `--dump-*`, `--preset`, `-o`, `--version`)
- Diagnostic severity model (warnings/remarks beyond errors — tca.md §2.7's uncoalesced-store perf warning needs a channel)

**Differentiators (already designed — keep and publish):**
- Race-freedom proven by construction + verifier-checked, with the invariant list published as a doc section (the flagship trust story — almost no production DSL states and machine-checks this)
- Run-to-run bitwise determinism for a fixed compiled kernel (free given no atomics/fixed schedules; needs one sentence in the docs)
- Once-per-logical-output seed semantics (validity tags, tca.md §5.10 — most DSL runtimes get this subtly wrong)
- IR-level reference interpreter in Go (nvcc-free oracle; schedule after e2e works)
- Source-span → generated CUDA `#line` mapping (decision: Loc on IR nodes — must land in IR-core phase; retrofit is a cross-cutting rewrite)
- Performance remarks channel (`-Rpass` style; tca.md §2.7 rule 6 already mandates the first one)

**Anti-features (v0 exclusions — do not add even when asked):** autotuning/cost-model search (keep the `select()` interface so a solver drops in later; add presets, not search), JIT/embedded compilation, Python bindings, tensor-core/TF32 paths (the exact silent-precision cliff that makes Triton irreproducible), user scheduling annotations, device print/assert, multi-GPU/streams/graphs/cooperative grid sync, grammar fuzzing (v1.x), full IEEE f64 support.

### Architecture Approach

The consensus across LLVM/MLIR/Triton/Halide: **the pipeline is the product** — small declarative ordered passes, all cross-cutting concerns (verification, dumps, timing, caching) live in pass-manager infrastructure, not in passes. Load-bearing patterns for tcacomp specifically:

1. **Pass instrumentation runs verification** (MLIR `PassInstrumentation`): verifier + IR dump + timing are PM callbacks. "A verifier for every transformation pass" becomes mechanically guaranteed; transformation passes contain zero verification boilerplate.
2. **Node-family legality gating** — *the* answer to the single-tier-IR soup risk (MEDIUM confidence: applied design judgment, direct transplant of MLIR's legality model, no off-the-shelf precedent): group node kinds into families (`abstract`, `loop`, `tile-op`, `collective`, `scalar-expr`), declare per-stage legality windows matching tca.md §5.2–5.12's pipeline, and have the PM check the IR against the window after every pass. Each transformation pass also declares consumed/produced families (doubles as a free verifier increment).
3. **Schedule-as-annotation (Halide decoupling):** compute structure never rewritten by scheduling; parallelism/tile/cache decisions are annotations written by the preset pass, read by lowering passes, cleared after consumption. Verifier checks annotation consistency.
4. **Runtime-owned synchronization; codegen-owned structure:** codegen never emits `__syncthreads`, shuffles, or fences — the race-freedom argument is only checkable if every barrier lives in runtime-audited code. Codegen holding an IR mutator or emitting sync is a code-review reject.
5. **Conservative analysis invalidation (LLVM new-PM simplified):** default = invalidate all cached analyses after every transform pass; preservation whitelist opt-in. At single-BB kernel scale, recompute cost is noise; a stale cache is the canonical far-from-cause crash.
6. **Single source of truth for constants:** tile sizes, budgets, presets live in `internal/flags` (or `internal/ir/cost`); the runtime receives values as template/launch parameters, never re-derives them.

**Major components:** `internal/ir` (node suite, SSA chains, mutation API, legality table — the only sanctioned exception to the immutability rule) · `internal/pass` (PM, instrumentation hooks, analysis cache) · `internal/analysis` (read-only queries; mutation-free is a package-level rule) · `internal/opt` (one file per pass, each 1:1 with a tca.md section and an invariant set) · `internal/codegen` (per-family emission registry, read-only on IR, emits only runtime entry points registered in a Go-side `rtapi` mirror of `runtime/include/tca/rt/` — drift caught by one e2e compile test) · `runtime/` (reducers, cooperative tile copies + ALL barriers, iterators, launch/shape validation, workspace) · `internal/syntax` (Phase 5 only; elaborator → builder calls) · `pkg/` (public `Compile()`; the elaborator's target, mostly built in Phase 1) · `internal/diagnostic`, `internal/log`, `internal/flags`, `internal/visualization` (cross-cutting, Phase 1).

### Critical Pitfalls

1. **SSA use-def chain corruption (P4, Phase 1)** — ad-hoc mutation of a mutable IR is the classic silent-miscompile. One choke-point mutation API (`SetDef`, `AddUse`/`RemoveUse`, `ReplaceAllUsesWith`, `Delete`), mandatory post-pass verifier checking **both** chain directions, stable integer IDs for node identity (never pointers in pass-local maps), rich verifier diagnostics (node ID, last-touched-by pass, invariant violated). Highest-leverage Phase-1 decision.
2. **Warp-shuffle mask/participation bugs (P1, Phase 3)** — `__shfl_*_sync` with out-of-range/idle lanes is UB on Volta+: undefined values or hangs. The runtime reducer owns the entire shuffle tree; every lane calls Init/Reduce/Combine unconditionally and only the *contribution* is predicated; full mask `0xffffffff` is correct only because whole-warp groups guarantee participation; validity tags mandatory for non-identity seeds (`init=5`, `min` — an identity contribution of 0 is wrong).
3. **`__syncthreads()` in divergent or non-uniform-trip-count code (P2, Phase 2)** — Volta+ independent thread scheduling makes barriers truly block-wide; pre-Volta code samples that "worked" are now deadlock templates. Hard rule: all loops containing a barrier have CTA-uniform trip counts, all predication in the body; runtime tile helpers take the tile extent and compute iterations from `blockDim` themselves; no early `return` in generated device code; `synccheck` in CI on every e2e test.
4. **Numerical comparison flakiness — bitwise vs tolerance (P3, planted in Phase 2, detonates in Phase 3)** — parallel reduction reassociation + nvcc's default `--fmad=true` guarantee last-ULP differences vs the CPU reference. Pin the policy in Phase 2 *before parallelization exists*: min/max/bitwise ops demand bitwise equality; fp +/* chains use relative tolerance (~1e-5 f32) with documented bounds; the naive Phase-2 test lane compiles with `--fmad=false` so bitwise comparison is valid at the phase boundary; small well-conditioned inputs so a dropped element (O(1/N) error) is far above tolerance.
5. **Dynamic SMEM >48 KB opt-in and silent launch failure (P5, Phase 4, discipline from Phase 2)** — three independent failure points (`extern __shared__`, per-specialization `cudaFuncSetAttribute` with an almost-never-checked return, matching `<<<>>>` bytes); A6000 ceiling is 101,376 B not 163 KB; errors surface asynchronously at the *next* CUDA call. The launch stub owns the whole sequence: query device attributes once, validate the compiler's plan against the device constant, check every return code loudly with kernel name + sizes; use `extern __shared__` + compiler-computed layout offsets, never static `__shared__`; `CUDA_LAUNCH_BLOCKING=1` in e2e.
6. **cp.async completion-group handling (P6, Phase 4)** — groups are per-thread state machines; `commit_group`/`wait_group` call counts are a collective invariant. Uniform commit counts (commit of an empty group is legal and cheap), `cp.async` zero-fill operand for predicated tails without divergence, and `__syncthreads()` after `wait_group N` (completion visibility ≠ CTA-wide data visibility).

## Implications for Roadmap

Research validates the existing 5-phase correctness-first plan and pins what each phase **must include beyond the obvious** — the correctness infrastructure is priced in early or it never lands.

### Phase 1: IR Core + Pass Infrastructure
**Rationale:** Everything hangs off `internal/ir` + `internal/pass`; the debugging backbone (verifier, diagnostics, visualization) must exist before any compiler logic (MLIR lesson: build IR + passes + printer first, parse last). Also deliberately first per syntax.md §11.1 — the `pkg/` builder API *is* the elaborator's target.
**Delivers:** SSA core (SSAVar spine with use-def/def-use chains), node suite with versioned `Kind` enum, choke-point mutation API, legality table + per-stage windows (Pattern 3), PM with instrumentation hooks (verify-after-every-pass structural, dumps, timing), conservative analysis cache, diagnostic + log + flags + visualization packages, analysis package skeleton, `pkg/` builder API, textual IR printer, golden-test harness, determinism byte-compare test.
**Must include (beyond the obvious):** the **Loc decision** (recommendation: `Loc{frontend span, pass origin}` on every node — cheap now, cross-cutting rewrite later); stable integer node IDs (a correctness decision, not perf); per-pass invariant sets written down *while the pipeline is small* (P12 — invariants never written later); warning/remark severity levels in diagnostics; Go determinism discipline (ordered structures, single sort before emission). Guard against scope creep into hash-consing/arenas (P16 — Phase 1 criteria are structural, not performance).
**Avoids:** P4 (chain corruption), P11 (map-iteration nondeterminism), P12 (pass-order coupling), P13 (diagnostic erosion), P16 (premature IR optimization).
**Research flag:** `/gsd-research-phase` worthwhile for the legality-window API shape (MEDIUM-confidence applied design judgment) and the mutation-API surface.

### Phase 2: Naive Codegen + Runtime + e2e Harness
**Rationale:** Riskiest phase — crosses every boundary at once (Go compiler, codegen, runtime, nvcc, GPU numerics). Must stay lean: sequential lowering + scalar reduce only; the entire Go↔C++ contract surface is built at its simplest and tested before any parallelism. First e2e signal is the project's core value gate.
**Delivers:** sequential emission (seq_for, scalar-expr → CUDA, sequential reducer calls, host launch stub with extent validation + zero-extent guards + per-launch workspace), runtime sequential Reducer + seq iterators + launch validation, `internal/codegen/rtapi` runtime-API registry (with the drift-catching compile test), `testutil` e2e driver (cmake/ninja via `os/exec`, `TCACOMP_E2E` gate), pinned FP policy + comparison policy, naive test lane with `--fmad=false`, determinism statement in docs.
**Must include (beyond the obvious):** the **Init/Reduce/Combine reducer contract defined here** (Phase 3 must only swap the collective implementation, not invent the contract); uniform-trip-count + predicated-body codegen discipline established *now* (P2 — before Phase 3 multiplies collectives); every-CUDA-return-checked launch-stub discipline (P5 starts here); tolerance policy written down *before* parallelization (P3); zero-extent launch guards (P9); `ShapeError` structured error surface.
**Avoids:** P2 (divergent barriers), P3 (tolerance crisis at the 2→3 boundary), P9 (zero-extent launches), P10 layer 1 (uncontrolled fmad).
**Research flag:** verify `__pipeline_*` intrinsic signatures and the C++20-subset corners against the actual CUDA 12.2 headers during this phase's planning (research is MEDIUM confidence on exact signatures).

### Phase 3: Parallelization Presets + Reduce Lowering
**Rationale:** Only after Phase 2's skeleton is proven. Preset pass writes annotations only (Halide decoupling); iter-lower is the first legality-window expansion; reduce-lower requires iter-lower's loop structure (tca.md §5.9's ordering constraint is a hard dependency — enforce via PM pass preconditions, not convention).
**Delivers:** `opt/parallelize` (S1–S6 behind `select(StatementCluster) -> Assignment`), `opt/iter-lower` (loop nodes, tile factoring), `opt/reduce-lower` (Init/Combine placement, accumulator tiles); runtime warp Reducer (shuffle tree, SMEM partials) + thread/CTA iterators; verifier's store-indexing and collective-uniformity invariants; §2.7 rule-6 uncoalesced-store warning; e2e extent matrix {0, 1, 255, 256, 257, non-multiple-of-tile, strided tails 0..3 elements/lane}.
**Must include (beyond the obvious):** validity-tag semantics for non-identity seeds tested explicitly (once-per-output init — differentiator); full-mask shuffle discipline with the "contribution is predicated, participation is not" contract asserted in debug builds; `synccheck` in CI from this phase on; the ownership-predicate representation decision for reduction stores (see Open Questions); f16/bf16 accumulator-precision warnings (>256 elements without f32 cast).
**Avoids:** P1 (shuffle masks), P8 (as/store mismatch — v0 surfaces the warning, preset-selection response deferred), P9 degenerate reducer cases, P10 layer 2 (cast-deleting simplifier).
**Research flag:** skip — patterns well-documented; the pitfalls list is the planning input.

### Phase 4: Fusion/Merge + SMEM Caches
**Rationale:** merge/fusion are IR-pure (abstract legality window, depend only on Phase 1) but only matter once parallelism exists (fusion decides launch count); cache-alloc needs parallelization annotations; cache-lower needs iteration lowering done. This is where the 64 KB budget is exercised.
**Delivers:** `opt/simplify` (DCE, identity-map removal, canonicalization — never deleting widening/narrowing casts around reductions), `opt/loop-merge`, `opt/fusion`, `opt/effect-linearize`, `opt/cache-alloc` (hoisting rule, FRAG/SMEM, 64 KB budget counting **padded** sizes), `opt/cache-lower` (variant selection); runtime cooperative LoadTileCta/LoadTileThread with `cp.async` + all barriers, SMEM layout accessor functions.
**Must include (beyond the obvious):** cp.async group discipline (uniform commits, zfill tails, barrier after wait); SMEM opt-in sequence with device-constant validation and loud failures; runtime-owned padded/swizzled tile accessors — generated code passes logical shapes only, never open-codes index arithmetic (P7: compiler/runtime layout agreement); padded sizes counted against budget; effect linearization must precede parallelization (coupling enforced by PM).
**Avoids:** P5 (SMEM opt-in), P6 (cp.async groups), P7 (layout drift).
**Research flag:** verify CCCL 12.2 header availability if preferring `cuda::memcpy_async` over raw `__pipeline_*` intrinsics (internal runtime decision, invisible to generated code).

### Phase 5: Frontend
**Rationale:** Deliberately last — parser against a proven IR is sugar, parser against an unproven IR rewrites the IR every time design tightens (MLIR's own `.mlir` parser postdates its core). The hand-built Phase 1–4 graphs are the parser's oracle.
**Delivers:** elaborator → builder calls (mostly done in Phase 1), then lexer (`go/scanner` structure port), recursive-descent parser (precedence climbing), name/shape resolution, scalar-expr checking with shared builtin/cast registries, `cmd/` CLI (`--emit-ir/--emit-cuda`, `--dump-*`, `--preset`, `-o`, `--version`).
**Must include (beyond the obvious):** milestone ends at "all documented example kernels parse to IR identical to the hand-built graphs"; one sensible error message per failure class (no error-recovery polish creep); golden diagnostics tests from syntax.md §10's 16-message table.
**Avoids:** P17 (parser-before-oracle), P15 (test creep).
**Research flag:** skip — recursive descent is the best-documented pattern in the corpus.

### Cross-Cutting Open Questions (need user decisions, with recommendations)

| Question | Where decided | Recommendation |
|---|---|---|
| **Loc tracking on IR nodes** | Phase 1, irrevocably | Include a minimal `Loc{frontend span, pass origin}` on every node; emit `#line` in codegen later. Retrofit after three phases is a cross-cutting rewrite; deciding "no" is also acceptable but must be a *decision*. |
| **FMA/numeric comparison policy** | Phase 2, pinned in docs | IEEE default everywhere; fmad contraction explicitly permitted and documented (per-op IEEE-exact); ftz off; no fast-math ever. Naive test lane compiles `--fmad=false` (bitwise vs CPU reference valid); perf lanes keep `--fmad=true` with tolerance. One paragraph in tca.md §4 or syntax.md §6 closes the spec CONFLICT. |
| **Ownership-predicate representation for reduction stores** | Phase 2 contract, Phase 3 exercised | Validity tags are mandatory for non-identity seeds (tca.md §5.10) — never represent ownership as an identity contribution (wrong for `init=5`, wrong for `min`). Encode as one runtime contract: "every lane participates unconditionally; contribution + validity is predicated." |
| **SMEM+FRAG 64 KB budget interpretation** | Phase 4 allocator (constraint from Phase 2) | Budget counts **padded** tile sizes (padding costs SMEM and occupancy), not logical sizes; runtime validates the compiler's plan against `cudaDevAttrMaxSharedMemoryPerBlockOptin` per device (101,376 B on A6000 — never hardcode 163 KB); clamp/fail loudly at launch with kernel name + requested vs available bytes. |
| **compute-sanitizer in CI** | Phase 2 harness, Phase 3 required | Upgrade from "may be used for debugging" to a required harness mode (racecheck/synccheck/memcheck on small shapes). For a compiler whose claim is race-freedom-by-construction, the sanitizer is the independent check that the argument holds. |
| **Run-to-run determinism statement** | Phase 3 docs (free) | Add one sentence: "for a fixed compiled kernel, TCA execution is deterministic (no atomics, no data-dependent scheduling)." Scope the promise to fixed artifacts before any future autotuner lands. |

### Phase Ordering Rationale

- Dependencies force the order: ir/pass have zero deps and everything hangs off them; codegen needs the read API + rtapi registry; parallelization needs a proven sequential skeleton; caches need annotations and lowered loops; the parser needs the proven IR (hand-built graphs as oracle).
- Correctness infrastructure front-loaded: diagnostics, determinism, verifier, tolerance policy, and the reducer contract all land in Phases 1–2 because every later phase's debugging depends on them and retrofit costs are cross-cutting.
- Merge/fusion *could* land earlier (IR-pure) but buy nothing e2e-visible until launch count matters — Phase 4 is correct.
- Each phase is e2e-verifiable on the A6000 against the flagship gate, so the roadmapper should keep phases small and gated rather than batching.

### Research Flags

Needs deeper research during planning:
- **Phase 1:** legality-window API design (MEDIUM confidence, applied judgment) and mutation-API surface detail.
- **Phase 2:** verify `__pipeline_*` / `cooperative_groups` signatures and C++20-subset corners against actual CUDA 12.2 headers (research-level facts, not yet compile-verified).

Standard patterns (skip `/gsd-research-phase`):
- **Phase 3:** well-documented warp-reduction and preset patterns; pitfalls list is the planning input.
- **Phase 5:** recursive descent is the best-documented pattern in the corpus.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Fixed toolchain verified against NVIDIA/CMake/GCC official docs; a few CUDA-specific facts (C++20 device-code bug corners, occupancy-API bug fix riding the driver) flagged MEDIUM |
| Features | MEDIUM-HIGH | Ecosystem facts verified against 2024–2025 official sources; gap analysis is first-party against in-repo authoritative docs (the highest-confidence kind) |
| Architecture | HIGH | LLVM/MLIR/Triton/Halide patterns are stable and well-documented; MEDIUM specifically on the single-tier legality-gating recommendation (applied design judgment) |
| Pitfalls | HIGH | Hardware/runtime semantics verified against NVIDIA docs + corroborating bug reports on identical hardware (sm_86/A6000); compiler-construction pitfalls from Go compiler SSA source + community practice |

**Overall confidence:** HIGH. This project has unusually strong inputs: fixed toolchain, in-repo authoritative design docs, a proven same-machine reference scaffold (scalfold), and single-GPU scope. The residual uncertainty is concentrated in two spots: fine-grained IR API design (ours to design) and CUDA 12.2 device-code corners (verify by compiling).

### Gaps to Address

- **FP policy CONFLICT in specs** (syntax.md §9.1 punts to an unstated "numerical policy"): resolve with the one-paragraph recommendation above before Phase 2 planning.
- **Tolerance/reference-precision spec missing:** no rtol/atol policy, no f64-reference statement; the e2e harness phase must close this (recommendation: f64 CPU reference, per-type tolerances, multi-shape × multi-seed matrix).
- **CLI surface and pass-dump flags unspecified** ("global flags, logging" only): close in Phase 1 (flags) and Phase 5 (CLI), with the MLIR flag set as the model.
- **compute-sanitizer downgraded to optional** in PROJECT.md: upgrade to a required harness mode (CI), or the race-freedom-by-construction claim has no independent check.
- **Determinism guarantee unstated:** free win, one sentence in docs; scope it to fixed compiled artifacts before any future autotuner.
- **Exact `__pipeline_*` intrinsic signatures and C++20 device-code bug corners:** reconfirm against the 12.2 headers during Phase 2 (facts are research-level, not compile-verified).
- **Legality-window design has no off-the-shelf precedent:** treat as Phase 1's one genuine design exercise; the cheaper fallback (single global "is lowered?" flag) loses per-stage precision and is not recommended.
- **`#line`/Loc decision:** make it explicitly at IR-core time even if negative; silence is the worst outcome.

## Sources

### Primary (HIGH confidence)
- doc/tca.md, doc/syntax.md, doc/design.md, PROJECT.md (this repo) — authoritative for gap analysis and all first-party design constraints
- [CUDA C++ Programming Guide — C++ Language Support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html) — C++20 host/device support level
- [NVIDIA CUDA 12.2 Linux Installation Guide](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-installation-guide-linux/) — supported host compiler ranges
- [CUDA Runtime API — Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html) — occupancy API semantics
- [Stack Overflow — Using maximum shared memory in CUDA](https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda) — sm_86 opt-in max 101,376 B
- [Using CUDA Warp-Level Primitives (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) — mask discipline, undefined values outside mask
- [sglang issue #29897](https://github.com/sgl-project/sglang/issues/29897) — unchecked `cudaFuncSetAttribute` failure pattern
- [Go compiler SSA backend README](https://go.dev/src/cmd/compile/internal/ssa/README) — per-pass verify pattern, opcode tables
- [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/) / [LLVM New Pass Manager](https://llvm.org/docs/NewPassManager.html) — instrumentation, invalidation, legality model
- [Halide PLDI '13](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) — schedule/algorithm decoupling
- [llir/llvm](https://github.com/llir/llvm) — pure-Go LLVM IR design reference
- [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) / [GoogleTest module](https://cmake.org/cmake/help/latest/module/GoogleTest.html)
- [GCC 11 release notes](https://gcc.gnu.org/gcc-11/changes.html) / [cppreference C++20 support](https://en.cppreference.com/w/cpp/compiler_support/20)
- [Triton debugging docs](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html) / [Triton tl.dot docs](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) — interpreter mode; TF32 default
- scalfold README (first-party, `/home/ps/Workspace/cuda/project/scalfold`) — methodology notes
- [NERSC compute-sanitizer guide](https://docs.nersc.gov/tools/debug/compute-sanitizer/) — sanitizer tool suite

### Secondary (MEDIUM confidence)
- [ax3l's CUDA compilers gist](https://gist.github.com/ax3l/9489132) — nvcc 12.0–12.4 C++20 regressions fixed 12.4.1
- [NVIDIA Forums — cudaOccupancyAvailableDynamicSMemPerBlock bug](https://forums.developer.nvidia.com/t/cudaoccupancyavailabledynamicsmemperblock-returning-incorrect-value/329573) — NVBUG 5209234
- [CCCL cuda::memcpy_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html) — async-copy abstraction availability
- [CUTLASS discussion #500](https://github.com/NVIDIA/cutlass/discussions/500) — cp.async wait_group group-counting discipline
- [Go compiler SSA phase internals](https://internals-for-interns.com/posts/the-go-ssa/) / [Fil Pizlo SSA notes](https://gist.github.com/pizlonator/cf1e72b8600b1437dda8153ea3fdb963) — SSA design tradeoffs
- [Triton-San, LLVM Dev Mtg Oct 2025](https://llvm.org/devmtg/2025-10/slides/technical_talks/lu.pdf) — debug-info preservation as open problem
- [GPU testing robustness study](https://svac-llc.com/case-studies/gpu-race-conditions-low-precision.html) — single-shape/single-seed testing catches 0/10 realistic bugs (LOW-MEDIUM)

### Tertiary (LOW confidence)
- None relied upon for load-bearing decisions; all LOW-confidence sources were corroborated or used only as cautionary context.

---
*Research completed: 2026-10-15*
*Ready for roadmap: yes*
