# Roadmap: TCA Compiler (tcacomp)

## Overview

tcacomp is built correctness-first, in five phases, each landing a working and e2e-verifiable capability on the A6000: (1) the verified IR + pass-infrastructure foundation everything hangs off, (2) a naive sequential compiler + runtime + e2e harness that reaches the core-value gate (`matmul_scale` on GPU vs CPU reference), (3) whole-warp parallelization presets with iteration/reduce lowering and the warp reducer, (4) the remaining documented optimization pipeline (merge/fusion/effect linearization/SMEM caches), and (5) the frontend (lexer/parser/elaborator) plus CLI, parsing the documented examples to IR identical to the hand-built oracle graphs. The correctness infrastructure (verifier, diagnostics, determinism, tolerance policy, reducer contract, sanitizer harness) is priced into Phases 1–2 because every later phase's debugging depends on it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: IR Core, Pass Infrastructure & Tooling** - Verified SSA IR, pass manager with structural verification, diagnostics/visualization/golden-test backbone, and the builder API oracle
- [ ] **Phase 2: Naive Codegen, Runtime & e2e Harness** - Sequential end-to-end compiler: generated CUDA + C++20 runtime + host stubs + e2e driver, reaching the flagship gate
- [ ] **Phase 3: Parallelization Presets, Iteration & Reduce Lowering** - Whole-warp preset scheduling (S1–S6), loop lowering, warp reducer, edge-shape verification
- [ ] **Phase 4: Fusion, Merge, Effect Linearization & SMEM Caches** - Full documented optimization pipeline with cp.async cooperative tiles under the 64 KB budget
- [ ] **Phase 5: Frontend — Lexer, Parser, Elaboration & CLI** - Full v0 syntax compiles; every documented example kernel parses to the oracle IR via the `tca` CLI

## Pipeline Staging (pass-order contract)

The final assembled pass pipeline, enforced by PM preconditions (PASS-01) per doc/tca.md, is:

```
simplify → loop-merge → fusion → effect-linearize → parallelize
  → iter-lower → reduce-lower → cache-alloc → cache-lower → codegen
```

Phases validate this pipeline in stages: Phase 2 runs codegen alone on canonical hand-built graphs; Phase 3 validates `parallelize → iter-lower → reduce-lower → codegen` on builder-constructed oracle graphs (the builder materializes pipes into explicit SSA data dependencies at construction time, so the linearize-before-parallelize precondition is trivially satisfied until the pass exists); Phase 4 lands `simplify/merge/fusion/effect-linearize` and validates the full pipeline order end-to-end. Hard orderings carried into phase plans: effect linearization before parallelization; iteration lowering before reduce lowering; reducer contract (Init/Reduce/Combine) defined in Phase 2, only the collective implementation swaps in Phase 3; cache allocation after reduce lowering, cache lowering after iteration lowering.

## Phase Details

### Phase 1: IR Core, Pass Infrastructure & Tooling
**Goal**: A verified, debuggable IR and pass-pipeline foundation with a programmatic graph-builder API — the substrate every later phase hangs off, with the correctness infrastructure (verifier, diagnostics, determinism, golden tests) structural from day one.
**Depends on**: Nothing (first phase)
**Requirements**: IR-01, IR-02, IR-03, IR-04, IR-05, IR-06, IR-07, PASS-01, PASS-02, PASS-03, PASS-04, PASS-05, PASS-06, TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05, TOOL-06, FE-01, VER-08
**Success Criteria** (what must be TRUE):
  1. A hand-built graph (via the `pkg/` builder API — e.g. the `map | reduce | map | store` chain of `matmul_scale`'s statement) round-trips through the canonical textual IR printer byte-identically across repeated runs and processes; golden FileCheck-style regression tests pass with a working update flow.
  2. Every IR mutation flows through the choke-point API and keeps both directions of the SSA use-def/def-use chains consistent — the post-pass verifier reports chain consistency, and a deliberately corrupted chain is caught with a diagnostic naming the node ID, the last-touched-by pass, and the invariant violated.
  3. The pass manager runs ordered pipelines and rejects out-of-order ones by enforced preconditions (e.g. reduce-lower before iter-lower); verify-after-every-pass, IR dumps, and pass timing run as PM callbacks, not inside passes, with a `--verify-each=0` escape hatch.
  4. Node-family legality windows per pipeline stage (MLIR-style) reject out-of-window IR after every pass, and the PM rejects unknown node `Kind`s.
  5. The builder API constructs the flagship `matmul_scale` graph programmatically; Graphviz `.dot` dumps render any pipeline stage; diagnostics emit error/warning/remark with source spans.
**Plans**: 5 plans
**Research flag**: `/gsd-research-phase` worthwhile before planning — legality-window API shape (MEDIUM-confidence applied design judgment, no off-the-shelf precedent) and the mutation-API surface detail.

Plans:
- [x] 01-01-PLAN.md — IR core — SSADef/SSAUse/SSAVar with use-def/def-use chains, single-BB CFG, versioned stable `Kind` enum, choke-point mutation API (SetDef/AddUse/RemoveUse/ReplaceAllUsesWith/Erase), stable integer node IDs, `Loc{frontend span, pass origin}` on every node
- [x] 01-02-PLAN.md — Legality windows + verifier framework (per-pass declared invariants, rich diagnostics) + pass manager (preconditions, instrumentation callbacks, conservative analysis cache) + `internal/flags` + `internal/analysis` skeleton
- [x] 01-03-PLAN.md — Tooling backbone — `internal/diagnostic` (severity model), `internal/log`, Graphviz emission, canonical deterministic IR printer, golden-test harness, `--dump-after`/`--dump-graphviz`/`--pass-timing` flags
- [x] 01-04-PLAN.md — `pkg/` elaborator/graph-builder API (the parser's future target) + `matmul_scale` oracle graph + determinism byte-compare test
- [x] 01-05-PLAN.md — Wire real dump emitters (`ir.Print`, `EmitDot`) into `pass.IRDump` via `pkg.DefaultInstrumentations` — closes TOOL-06 dump flags

### Phase 01.1: IR architecture refinement — extensible Type interface (NumericType), SSAUse/SSADef/SSAVar handle redesign, visitor-pattern graph walk (INSERTED)

**Goal:** [Urgent work - to be planned]
**Requirements**: TBD
**Depends on:** Phase 1
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 01.1 to break down)

### Phase 2: Naive Codegen, Runtime & e2e Harness
**Goal**: A correct end-to-end compiler on a deliberately sequential skeleton — generated CUDA C++, C++20 runtime, host launch stubs, and a Go e2e driver that compiles, runs on the A6000, and checks against a CPU reference — achieving the project's core-value gate.
**Depends on**: Phase 1
**Requirements**: CG-01, CG-02, CG-03, CG-04, RT-01, RT-02, RT-08, VER-01, VER-02, VER-03, VER-05, VER-06, NUM-01
**Success Criteria** (what must be TRUE):
  1. `matmul_scale` (hand-built via the builder API) compiles end-to-end in one automated flow: generated `.cu` + host stub + runtime build under CMake/Ninja driven by the Go e2e testutil (`TCACOMP_E2E=1` gate, skips without GPU), runs on the A6000, and matches the f64 CPU reference — bitwise in the `--fmad=false` naive lane.
  2. The runtime sequential Reducer honors the Init/Reduce/Combine contract: the initializer is applied exactly once per logical output, an empty reduction returns the declared initializer, and validity tags (never identity-padding) handle non-identity seeds; the host stub validates dynamic extents, guards zero-extent launches, owns per-launch gmem workspace, and surfaces structured `ShapeError`s.
  3. compute-sanitizer (racecheck/synccheck/memcheck) runs as a required harness mode on small shapes and reports clean; every CUDA return code is checked loudly with kernel name + sizes on failure.
  4. Generated code contains no barriers, shuffles, or fences — synchronization exists only as runtime calls, auditable in the emitted `.cu`; the Go-side `rtapi` registry mirrors `runtime/include/tca/rt/` and header drift is caught by a compile test.
  5. The numeric policy is pinned in docs and code: IEEE 754 per operation, FMA contraction permitted and documented per operation, no ftz/fast-math/silent precision reduction; the comparison policy (bitwise for min/max and integer ops, ~1e-5 relative for f32 +/× chains) is stated with a multi-shape × multi-seed matrix.
**Plans**: 4 plans
**Research flag**: verify `__pipeline_*` intrinsic signatures, `cooperative_groups` availability, and C++20-subset corners against the actual CUDA 12.2 headers during this phase's planning (research-level facts, not yet compile-verified).

Plans:
- [ ] 02-01: Codegen — sequential CUDA emission (seq loops, scalar expressions, chain-ordered statements, read-only on IR), host launch stub per kernel, `internal/codegen/rtapi` runtime-API registry + drift-catching compile test
- [ ] 02-02: Runtime — CMake skeleton (`CUDA_ARCHITECTURES 86` pinned, separable compilation ON, C++17 + `if constexpr` device subset), sequential Reducer specialization defining the Init/Reduce/Combine contract with validity tags, seq iterators, launch/shape validation, per-launch gmem workspace
- [ ] 02-03: e2e harness — Go `testutil` driver (cmake/ninja via `os/exec` in `t.TempDir()`), pinned FP + comparison policy, `--fmad=false` naive lane, compute-sanitizer mode, multi-shape × multi-seed matrix
- [ ] 02-04: Flagship gate — `matmul_scale` naive end-to-end on the A6000 vs f64 CPU reference (bitwise in the naive lane); determinism byte-compare across repeated runs

### Phase 3: Parallelization Presets, Iteration & Reduce Lowering
**Goal**: The compiler assigns whole-warp parallel schedules from the fixed preset table (schedule-as-annotation), lowers iterations and reductions into collective-correct loop nests, and the runtime grows the warp reducer — all verified on the edge-shape matrix.
**Depends on**: Phase 2
**Requirements**: PAR-01, PAR-02, PAR-03, PAR-04, PAR-05, PAR-06, CG-05, RT-03, RT-04, TOOL-08, VER-04, VER-07, NUM-02
**Success Criteria** (what must be TRUE):
  1. All six presets S1–S6 are selectable behind `select(StatementCluster) -> Assignment` and apply only annotations — compute structure is never rewritten by scheduling (verifiable by before/after IR dumps); every thread group is a whole number of physical warps at 256 threads/CTA.
  2. Iteration lowering emits `seq_for`/`thread_for`/`cta_for` with tile-loop factoring per tca.md §5.9 (golden IR verified); reduce lowering places Init/Combine around the reduced dimension's tile-between loop, materializes accumulator tiles, re-iterates fused consumers after Combine, and the `reduce<@r>` annotation stays stable across tiling.
  3. The warp reducer produces correct results for non-identity initializers (e.g. `init=5`, `min`) with validity tags; every lane participates in Init/Reduce/Combine unconditionally and only the contribution is predicated (asserted in debug builds); full-mask shuffle discipline never consumes out-of-range shuffle sources.
  4. The e2e edge-shape matrix (extents {0, 1, 255, 256, 257, non-multiple-of-tile, strided tails}; empty output extents tested separately from empty reduction extents) passes on GPU within pinned tolerances with compute-sanitizer synccheck clean; generated `thread_for` loops have CTA-uniform trip counts with predicated bodies and no early `return`.
  5. The verifier checks store-indexing, collective-uniformity, and annotation-consistency invariants after this phase's passes; the tca.md §2.7 rule-6 uncoalesced-store performance remark fires through the `-Rpass`-style remarks channel; the docs state the run-to-run determinism guarantee (scoped to fixed compiled artifacts).
**Plans**: 5 plans
**Research flag**: none — patterns well-documented; the pitfalls list (P1 shuffle masks, P9 degenerate reducers) is the planning input.

Plans:
- [ ] 03-01: `opt/parallelize` — preset table S1–S6 behind `select()`, schedule-as-annotation, whole-warp group discipline, legality-window expansion
- [ ] 03-02: `opt/iter-lower` — `seq_for`/`thread_for`/`cta_for` nodes with tile-between/tile-inside factoring per §5.9
- [ ] 03-03: `opt/reduce-lower` — Init/Combine placement, accumulator tiles, fused-consumer re-iteration per §5.10, plus the ownership-predicate representation design (focused discussion before lowering, per project directive)
- [ ] 03-04: Runtime warp Reducer (shuffle tree + SMEM cross-warp partials, participation contract) + thread/CTA iterators + runtime `select()`-driven specialization swap (contract unchanged from Phase 2)
- [ ] 03-05: Verifier invariants (store-indexing, collective-uniformity, annotation-consistency) + performance remarks channel + edge-shape e2e matrix + Go IR reference interpreter (nvcc-free oracle) + determinism doc statement

### Phase 4: Fusion, Merge, Effect Linearization & SMEM Caches
**Goal**: The complete documented optimization pipeline runs — simplification, loop merge, loop fusion, effect linearization, and CTA cache allocation/lowering into cp.async cooperative tiles under the 64 KB SMEM+FRAG budget — with correctness held e2e on the flagship gate.
**Depends on**: Phase 3
**Requirements**: OPT-01, OPT-02, OPT-03, OPT-04, OPT-05, OPT-06, RT-05, RT-06, RT-07
**Success Criteria** (what must be TRUE):
  1. Simplify, loop-merge, fusion, and effect-linearize transform golden IR per tca.md §5.2–§5.6: fusion enforces the `as`-prefix precondition and fuses only same-iteration-point pipe consumers (transpose-style reloads stay unfused); DCE never deletes widening/narrowing casts around reductions; the full pipeline order is PM-enforced and validated end-to-end.
  2. Cache allocation hoists loads per the §5.8 rule — at most 2 cached operands per map, at most 2 dimensions per cached operand, FRAG/SMEM placement chosen by covered-iteration parallelism, padded sizes counted against the 64 KB SMEM+FRAG budget, with the halve-T(R)-then-drop fallback (selection never fails).
  3. Cooperative `LoadTileCta`/`LoadTileThread` (+ storetile) with `cp.async` pass e2e honoring the runtime contract (visibility, reuse-safety, bounds, uniform call counts, empty-group commits, zero-fill predicated tails); the dynamic-SMEM >48 KB opt-in sequence validates against device attributes (101,376 B sm_86 ceiling, never hardcoded) and fails loudly with requested-vs-available bytes.
  4. `matmul_scale` runs e2e with caches enabled, matching the CPU reference within pinned tolerances, sanitizer-clean; generated code passes logical shapes only to runtime-owned padded/swizzled tile accessors — no open-coded tile index arithmetic in the emitted `.cu`.
**Plans**: 5 plans
**Research flag**: verify CCCL 12.2 header availability if preferring `cuda::memcpy_async` over raw `__pipeline_*` intrinsics (internal runtime decision, invisible to generated code).

Plans:
- [ ] 04-01: `opt/simplify` — DCE (chain removed unless it contains a store), arity-1 identity-map removal, canonicalization
- [ ] 04-02: `opt/loop-merge` (longest common `as`-prefix for data-independent statements) + `opt/fusion` (`as`-prefix precondition, same-iteration-point consumption)
- [ ] 04-03: `opt/effect-linearize` — pipe removal, explicit linear dependencies, full-pipeline order enforcement through PM preconditions
- [ ] 04-04: `opt/cache-alloc` — hoisting rule, candidate selection, ≤2/≤2 limits, FRAG/SMEM placement, padded-size 64 KB budget, halve-T(R)-then-drop fallback
- [ ] 04-05: `opt/cache-lower` + runtime cooperative tile load/store with `cp.async` and all synchronization, dynamic SMEM opt-in sequence, runtime-owned padded/swizzled accessors + e2e re-gate with caches on

### Phase 5: Frontend — Lexer, Parser, Elaboration & CLI
**Goal**: The full documented v0 syntax compiles: every example kernel parses through the lexer, recursive-descent parser, and elaborator to IR identical to the hand-built oracle graphs, driven by the `tca` CLI.
**Depends on**: Phase 4
**Requirements**: FE-02, FE-03, FE-04, FE-05, FE-06, FE-07, TOOL-07
**Success Criteria** (what must be TRUE):
  1. The lexer tokenizes per syntax.md §7.2 — longest-match tokens, line/block comments, lexical errors for malformed numerics and unterminated comments.
  2. The recursive-descent parser with precedence climbing covers the full v0 EBNF and recovers at semicolons and closing braces.
  3. Name/shape resolution manages the dim/axis/tensor/scalar/builtin namespaces, collects dynamic extent-equality checks for launch-time validation, and scalar expressions check against shared operator/cast/builtin registries; `&&`/`||`/`?:` preserve lazy conditional evaluation through to codegen.
  4. Every documented example kernel (syntax.md §2, §9.1–9.9) parses to IR identical (golden-compare) to the hand-built Phase 1 oracle graphs; golden diagnostics tests cover the message table in §10.
  5. The `tca` CLI supports `--emit-ir`/`--emit-cuda`, `--dump-*`, `--preset`, `-o`, and `--version`; `matmul_scale` compiles from frontend source end-to-end on the A6000 (final flagship gate).
**Plans**: 4 plans
**Research flag**: none — recursive descent is the best-documented pattern in the corpus.

Plans:
- [ ] 05-01: Lexer (`go/scanner` structure port) + recursive-descent parser with precedence climbing and error recovery
- [ ] 05-02: Name/shape resolution + scalar-expression checking against shared registries (lazy `&&`/`||`/`?:` preserved)
- [ ] 05-03: Pipeline elaboration (§8.3) + example-kernel oracle tests (§2, §9.1–9.9) + golden diagnostics tests (§10 table)
- [ ] 05-04: `cmd/tca` CLI surface + final flagship gate from frontend source

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. IR Core, Pass Infrastructure & Tooling | 0/4 | Not started | - |
| 2. Naive Codegen, Runtime & e2e Harness | 0/4 | Not started | - |
| 3. Parallelization Presets, Iteration & Reduce Lowering | 0/5 | Not started | - |
| 4. Fusion, Merge, Effect Linearization & SMEM Caches | 0/5 | Not started | - |
| 5. Frontend — Lexer, Parser, Elaboration & CLI | 0/4 | Not started | - |

---
*Roadmap created: 2026-10-15*
