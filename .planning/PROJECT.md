# TCA Compiler (tcacomp)

## What This Is

An optimizing compiler that turns an extremely simple, C++-style tensor kernel language (TCA frontend, `doc/syntax.md`) into high-performance CUDA C++, via a single-tier SSA IR called TCA (Tensor Computation Algebra, `doc/tca.md`). The compiler is written in pure Go; the generated code links against a C++20 runtime library (CMake) that provides reducers, cooperative tile load/store, iterators, and launch/shape validation. The runtime design takes scalfold (`/home/ps/Workspace/cuda/project/scalfold`) as its reference scaffold: explicit storage tiers, generator-driven iteration, compiler-owned scheduling.

## Core Value

`matmul_scale` (and the other documented example kernels) compiles end-to-end and runs correctly on the target GPU with numerical results matching a CPU reference — from frontend source, through the TCA IR and its optimization passes, to CUDA C++ plus runtime, in one automated flow.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] TCA IR core: SSA graph (SSADef/SSAUse/SSAVar with use-def and def-use chains), single basic block per kernel, typed values, all IR tiers in one node suite (lowering introduces new node kinds, never a separate IR)
- [ ] Pass infrastructure: pass manager, analysis passes, optimization/transformation passes, global flags, logging, diagnostics, Graphviz IR visualization
- [ ] IR verification: a verifier pass for every transformation pass, checking stated invariants before/after, with rich diagnostic output
- [ ] Codegen: Go compiler emits CUDA C++ (.cu) + host launch stubs; correctness-first lowering (sequential loops, scalar reduction), then the full documented pipeline
- [ ] Runtime library (C++20, CMake): reducer specializations (sequential, warp/whole-warp thread, CTA-deferred), cooperative tile load/store with all synchronization, seq/thread/CTA iterators, launch-time shape validation, per-launch gmem workspace
- [ ] Parallelization: fixed preset table (S1–S6, whole-warp flat thread groups only), iteration lowering (seq_for/thread_for/cta_for), reduce lowering with Init/Combine placement and accumulator tiles
- [ ] Optimizations on the single IR: simplification (DCE, identity-map removal, canonicalization), loop merge, loop fusion, effect linearization, CTA cache allocation (SMEM/FRAG, 64 KB budget)
- [ ] Frontend: elaborator/graph-builder API first (programmatic construction + IR-level tests), then lexer/parser for the full EBNF grammar, name/shape resolution, scalar expression checking with shared registries
- [ ] End-to-end on-GPU verification: nvcc-compiled generated kernels checked numerically against CPU reference (A6000 available); tests limited to fundamentals per design doc — no excessive test suites
- [ ] Project layout: `runtime/` (C++), `pkg/` (public Go), `internal/` (compiler internals: syntax, ir, util, opt, pass, analysis, codegen, diagnostic, visualization, log, flags), `cmd/` (CLI)

### Out of Scope

- TMA (sm_90+) — target GPU is A6000/sm_86; `cp.async` is the async-copy ceiling — hardware limitation
- CTA/global (multi-CTA) reduction — no preset assigns CTA to a reduced dimension in v0; S6 runs full reductions on one CTA — v0 scheduling policy, split-kernel is future work
- Mixed-type reduction combiners — v0 reducers are homogeneous; deferred with a defined extension path (declared, type-checked signatures)
- 2-D product-coordinate thread groups (`16 × 16` style) — violate whole-warp assignment; removed from presets; re-entry only via a deliberately defined `(warp, lane)` → coordinate mapping
- Root `map()` / root `map(...) as ()` — IR zero-iteration root-map semantics deliberately excluded from v0 frontend
- Arbitrary CUDA calls, atomics, explicit sync, user control flow (`for`/`if`/…), pointers, in-place aliasing — breaks the structural/no-race-by-construction model
- `f8` formats — concrete formats unspecified; frontend excludes them
- Kernel-calling-kernel, multiple return values, namespaces/templates/preprocessing — v0 frontend boundary
- Cross-cluster fusion and alternative synchronization — v0 execution interface is conservative (cluster-per-kernel, gmem workspace)
- Frontend parser as the first milestone — elaborator/graph-builder API comes first per `doc/syntax.md` §11.1

## Context

- **Design docs** (authoritative, in-repo): `doc/design.md` (requirements/modules), `doc/syntax.md` (frontend v0 spec: grammar, semantics, lowering contract), `doc/tca.md` (IR algebra, pass pipeline, presets, lowering). All three were audited and revised during project initialization: presets are now whole-warp-only, the zero-arity-map semantics, homogeneous-reducer, and execution-interface questions are resolved in-text.
- **Reference scaffold**: scalfold DSL (`/home/ps/Workspace/cuda/project/scalfold`) — header-only C++20 CUDA DSL, proven at TinyLlama-1.1B scale on this machine's A6000. TCA's runtime mirrors its philosophy (explicit storage, generators, typed views, storage-vs-compute precision boundary) but is compiler-driven rather than user-driven.
- **Target hardware**: single NVIDIA RTX A6000 (sm_86, Ampere), 48 GB. CUDA toolkit 12.2 (nvcc), driver 580.178 (CUDA 13 capable). C++20 device code is fine; TMA is not (sm_90+).
- **Toolchain**: Go 1.26.2 (compiler implementation, stdlib-only), CMake 4.4.3 + g++ 11.4 (runtime library), Ninja available.
- **Verification environment**: GPU present on the dev machine, so e2e tests compile generated CUDA with nvcc and run numerical checks against CPU references. compute-sanitizer/nsys/ncu may be used for debugging per scalfold's methodology notes.
- **Documentation-driven workflow**: user requires that any uncertain, ambiguous, or seemingly incorrect design item stops implementation for discussion before proceeding.

## Constraints

- **Tech stack**: Compiler in pure Go (standard library only, no cgo) — design.md requirement; runtime in C++20 built with CMake — design.md requirement.
- **Architecture**: single-tier IR — no separate high/mid/low IRs; lowering introduces nodes into the same IR suite (design.md).
- **IR style**: SSA; CFG has exactly one basic block per kernel/function (no control flow in v0 language).
- **Parallelization**: compiler-owned only; users never bind iterations to CUDA threads/blocks; every thread group is a whole number of physical warps (design decision, revised into `doc/tca.md`).
- **Correctness model**: race-freedom by construction — structural stores (one owner per address), reduced dimensions only through reducers, in-kernel data flow through pipes, CTA-uniform collectives. Verifier checks invariants after every pass.
- **Testing**: only necessary/fundamental tests — IR unit tests plus e2e numerical checks; explicitly no excessive test suites (design.md).
- **Backend**: generated backend code is CUDA C++ for now (design.md); compiler emits code + runtime files, it does not drive nvcc (user/test harness compiles).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Correctness-first phasing: IR core → naive codegen + runtime → presets/parallelization → fusion/caches | Working compiler early; every phase verifiable e2e on the A6000; optimizations land on a proven base | — Pending |
| Compiler emits `.cu` + runtime + host stubs; tests drive nvcc via exec | Separation of translation and build; user keeps control of the build | — Pending |
| Flagship success gate: `matmul_scale` correct end-to-end on GPU vs CPU reference | The doc's central example exercises map + reduce + piped map + stores + runtime reducers | — Pending |
| Whole-warp flat thread groups only; 2-D product-coordinate presets removed (S2/S3 rewritten; uniform 256 threads/CTA) | tca.md's own whole-warp rule; strided loops give multi-element accumulators without 2-D mappings | — Pending |
| v0 execution interface: cluster-per-kernel, runtime-owned gmem workspace, host launch function per kernel with launch-time extent validation | tca.md §5.1 left it undefined; conservative choice keeps race-freedom argument intact | — Pending |
| Elaborator/graph-builder API before parser | syntax.md §11.1; IR-level tests before grammar exists | — Pending |
| Doc issues fixed in-text during init (zero-arity map semantics, homogeneous reducers, preset reconciliation, typos) | User directive: design docs must be unambiguous before implementation | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-15 after initialization*
