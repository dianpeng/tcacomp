# Requirements: TCA Compiler (tcacomp)

**Defined:** 2026-10-15
**Core Value:** `matmul_scale` (and the other documented example kernels) compiles end-to-end and runs correctly on the target GPU with numerical results matching a CPU reference — from frontend source, through the TCA IR and its optimization passes, to CUDA C++ plus runtime, in one automated flow.

## v1 Requirements

Requirements for initial release. Each maps to exactly one roadmap phase.

### IR Core

- [ ] **IR-01**: SSA graph core implements SSADef/SSAUse/SSAVar — shared underlying pointer, use-def and def-use chains, trace-all-uses/defs, replace-def (all refs update), delete
- [ ] **IR-02**: CFG model holds exactly one basic block per kernel/function; statement RHS is an SSAUse or a Value (any value-producing node)
- [ ] **IR-03**: Versioned, stable `Kind` enumeration covers all node families (abstract, loop, tile-op, collective, scalar-expr); the pass manager rejects unknown kinds
- [ ] **IR-04**: All IR mutation flows through a choke-point API (SetDef, AddUse/RemoveUse, ReplaceAllUsesWith, Erase) — the only sanctioned exception to the immutability rule
- [ ] **IR-05**: Stable integer node IDs; deterministic iteration and emission ordering (no map-iteration-order dependence anywhere)
- [ ] **IR-06**: Minimal `Loc{frontend span, pass origin}` on every IR node (enables `#line` in generated CUDA later)
- [ ] **IR-07**: Node-family legality windows per pipeline stage (MLIR-style), checked by the pass manager after every pass

### Pass Infrastructure

- [ ] **PASS-01**: Pass manager runs ordered pass pipelines with enforced pass preconditions (e.g., iteration lowering before reduce lowering, effect linearization before parallelization)
- [ ] **PASS-02**: Pass instrumentation is structural: verify-after-every-pass, IR dumps, and pass timing run as PM callbacks, not inside passes
- [ ] **PASS-03**: Conservative analysis cache — invalidate-all default after transform passes, explicit preservation whitelist opt-in
- [ ] **PASS-04**: Global flags package is the single source of truth for compiler constants (tile sizes, budgets, presets, policies)
- [ ] **PASS-05**: Analysis package skeleton exists with a package-level read-only / mutation-free rule
- [ ] **PASS-06**: Every transformation pass declares its before/after invariants; verifier passes check them with rich diagnostics (node ID, last-touched-by pass, invariant violated)

### Diagnostics & Tooling

- [ ] **TOOL-01**: Diagnostic package with severity model (error / warning / remark), source spans, and structured context
- [ ] **TOOL-02**: Log infrastructure capable of compiler-scale debug output (per-pass, per-node)
- [ ] **TOOL-03**: Graphviz IR visualization (`.dot` emission) for any pipeline stage
- [ ] **TOOL-04**: Canonical deterministic textual IR printer
- [ ] **TOOL-05**: Golden IR regression tests (FileCheck-style patterns with an update flow)
- [ ] **TOOL-06**: Pass dump/trace flags: `--dump-after`, `--dump-graphviz`, `--pass-timing`, `--verify-each=0` escape hatch
- [ ] **TOOL-07**: `tca` CLI: `--emit-ir` / `--emit-cuda`, `--dump-*`, `--preset`, `-o`, `--version`
- [ ] **TOOL-08**: Performance remarks channel (`-Rpass` style; tca.md §2.7 rule-6 uncoalesced-store warning is the first client)

### Codegen

- [ ] **CG-01**: CUDA C++ emission from lowered IR; codegen is read-only on the IR
- [ ] **CG-02**: Runtime-API registry (Go-side mirror of `runtime/include/tca/rt/`); header drift is caught by a compile test
- [ ] **CG-03**: Host launch stub per frontend kernel: dynamic-extent validation, zero-extent launch guards, per-launch workspace allocation, dependency-ordered launches, every CUDA return code checked loudly (kernel name + sizes on failure)
- [ ] **CG-04**: Codegen never emits barriers, shuffles, or fences — all synchronization is runtime calls (race-freedom argument stays auditable)
- [ ] **CG-05**: `thread_for` loops are emitted with CTA-uniform trip counts and predicated bodies; no early `return` in generated device code

### Runtime

- [ ] **RT-01**: C++20 runtime library builds with CMake (`CUDA_ARCHITECTURES 86` pinned, never `native`; separable compilation ON); device code written as C++17 + `if constexpr` subset
- [ ] **RT-02**: Reducer contract `Init/Reduce/Combine` is defined at naive-codegen time; sequential specialization ships first, parallel specializations swap the collective implementation only
- [ ] **RT-03**: Seq / thread / CTA iterator objects wrapping generated loops (strided coverage, remainder guards)
- [ ] **RT-04**: Warp reducer: shuffle tree + SMEM cross-warp partials for whole-warp groups; every lane participates in Init/Reduce/Combine unconditionally, only the contribution is predicated; validity tags mandatory for non-identity seeds (never identity-padding)
- [ ] **RT-05**: Cooperative `LoadTileCta`/`LoadTileThread` (+ storetile) with `cp.async` and ALL synchronization inside the runtime; visibility, reuse-safety, bounds, and uniform-call contract per tca.md §5.11
- [ ] **RT-06**: Dynamic SMEM opt-in sequence: `extern __shared__`, per-kernel `cudaFuncSetAttribute` with checked returns, device-attribute validation against the 101,376 B sm_86 ceiling (never hardcoded), loud failure with requested-vs-available bytes
- [ ] **RT-07**: Runtime-owned padded/swizzled tile accessors — generated code passes logical shapes only, never open-codes tile index arithmetic
- [ ] **RT-08**: Launch/shape validation surfaces a structured `ShapeError`; runtime owns per-launch gmem workspace for materialized intermediates

### Parallelization

- [ ] **PAR-01**: Preset table S1–S6 behind `select(StatementCluster) -> Assignment`; the preset pass writes annotations only (schedule-as-annotation; compute structure never rewritten by scheduling)
- [ ] **PAR-02**: Every thread group is a whole number of physical warps (multiple of 32 threads, flat 1-D shard); 256 threads/CTA; no sub-warp or 2-D product-coordinate groups
- [ ] **PAR-03**: Iteration lowering emits `seq_for`/`thread_for`/`cta_for` nodes with tile-loop factoring per tca.md §5.9 (tile-between/tile-inside table)
- [ ] **PAR-04**: Reduce lowering implements Init/Combine placement, accumulator tiles, and fused-consumer re-iteration per tca.md §5.10; `reduce<@r>` annotation stable across tiling
- [ ] **PAR-05**: Ownership-predicate representation for reduction-result stores is designed and documented (focused design discussion before reduce lowering, per project directive)
- [ ] **PAR-06**: Verifier checks store-indexing, collective-uniformity, and annotation-consistency invariants after this phase

### Optimizations

- [ ] **OPT-01**: Simplification pass: DCE (chain removed unless it contains a store), identity-map removal (arity-1 only), canonicalization — never deleting widening/narrowing casts around reductions
- [ ] **OPT-02**: Loop merge for data-independent statements (longest common prefix in each statement's `as` order)
- [ ] **OPT-03**: Loop fusion with the as-prefix precondition and same-iteration-point pipe consumption (transpose-style reloads stay unfused)
- [ ] **OPT-04**: Effect linearization removes all pipes and makes dependencies explicit, ordered before parallelization
- [ ] **OPT-05**: CTA cache allocation implements the hoisting rule, ≤2 cached operands per map, ≤2 dimensions per cached operand, FRAG/SMEM placement by covered-iteration parallelism, 64 KB SMEM+FRAG budget counting **padded** sizes with halve-T(R)-then-drop fallback
- [ ] **OPT-06**: Cache lowering selects runtime variants by enclosing loop type (cta_for/thread_for → SMEM cooperative; seq_for → FRAG per-thread)

### Frontend

- [ ] **FE-01**: `pkg/` elaborator / graph-builder API — programmatic construction that produces graphs identical to the parser's target; available from Phase 1 as the oracle for all later work
- [ ] **FE-02**: Lexer per syntax.md §7.2: longest-match tokens, line/block comments, lexical errors for malformed numerics and unterminated comments
- [ ] **FE-03**: Recursive-descent parser with precedence climbing covering the full v0 EBNF; error recovery at semicolons and closing braces
- [ ] **FE-04**: Name and shape resolution: dim/axis/tensor/scalar/builtin namespaces; dynamic extent-equality checks collected for launch-time validation
- [ ] **FE-05**: Scalar expression checking against shared operator/cast/builtin registries; `&&`/`||`/`?:` preserve lazy conditional evaluation through to codegen
- [ ] **FE-06**: Pipeline elaboration per syntax.md §8.3; every documented example kernel (§2, §9.1–9.9) parses to IR identical to the hand-built graphs
- [ ] **FE-07**: Golden diagnostics tests covering the message table in syntax.md §10

### Verification

- [ ] **VER-01**: e2e driver in Go testutil: compile → cmake/ninja via `os/exec` in `t.TempDir()` → GPU run → f64 CPU reference computed in Go → allclose with per-type tolerances; gated by `TCACOMP_E2E=1`, skips when no GPU
- [ ] **VER-02**: Naive-codegen test lane compiles with `--fmad=false` so bitwise comparison against the CPU reference is valid at the Phase 2 boundary
- [ ] **VER-03**: compute-sanitizer is a required harness mode (racecheck / synccheck / memcheck on small shapes) — the independent check of the race-freedom-by-construction claim
- [ ] **VER-04**: Edge-shape matrix in e2e tests: extents {0, 1, 255, 256, 257, non-multiple-of-tile, strided tails}; empty output extents tested separately from empty reduction extents
- [ ] **VER-05**: Multi-shape × multi-seed test matrix; comparison policy pinned: bitwise for min/max and integer ops, ~1e-5 relative tolerance for f32 +/× chains
- [ ] **VER-06**: Flagship gate: `matmul_scale` compiles and runs end-to-end on the A6000 with results matching the CPU reference within the pinned tolerances
- [ ] **VER-07**: Go IR reference interpreter (TRITON_INTERPRET-style, nvcc-free oracle) executable over the tiny IR
- [ ] **VER-08**: Determinism byte-compare test: a fixed compiled kernel produces bitwise-identical outputs across repeated runs

### Numeric Policy

- [ ] **NUM-01**: Floating-point evaluation follows IEEE 754 per operation; FMA contraction permitted by default, applied per operation only, and documented; no flush-to-zero, no fast-math, no silent precision reduction (TF32 / tensor-core down-conversion excluded)
- [ ] **NUM-02**: Determinism guarantee stated in docs: a fixed compiled kernel executes deterministically (no atomics, fixed schedule); promise scoped to fixed artifacts

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Scheduling & Performance

- **SCHED-01**: Split-kernel / CTA-global (multi-CTA) reductions behind cooperative launch or an explicit physical execution plan
- **SCHED-02**: 2-D `(warp, lane)` → coordinate mapping enabling 2-D thread-group presets and first-class per-thread register tiles
- **SCHED-03**: ILP / LP / polyhedra strategy selection replacing the preset table behind the `select()` interface
- **SCHED-04**: Tensor-core instruction recognition and fast bulk-load / copy-rewrite rules (design.md future development)
- **SCHED-05**: 2-D CTA grid sharding to fix the single-dimension grid-parallelism limitation

### Language & IR Extensions

- **LANG-01**: Mixed-type reduction combiners with declared, type-checked signatures
- **LANG-02**: `f8` element types (once concrete formats are chosen)
- **LANG-03**: Root `map()` / zero-iteration root-map semantics
- **LANG-04**: Multiple kernel return values; kernel-calling-kernel
- **LANG-05**: Control flow (would introduce phis/dominance — major IR extension)
- **LANG-06**: Cross-cluster fusion and alternative synchronization mechanisms

### Tooling

- **TOOLV2-01**: Autotuning / cost-model search (presets remain the v1 answer)
- **TOOLV2-02**: JIT / embedded compilation and Python bindings
- **TOOLV2-03**: Grammar fuzzing for the frontend
- **TOOLV2-04**: f64 element type support

## Out of Scope

Explicitly excluded for v1. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Autotuning / cost-model search | v0 uses the fixed preset table; the `select()` interface is the deliberate upgrade point — add presets, not search |
| JIT / embedded compilation, Python bindings | Compiler emits artifacts; user/test harness builds them — a logged project decision |
| Tensor-core / TF32 paths | The exact silent-precision cliff that makes other DSLs irreproducible; violates the pinned numeric policy |
| User scheduling annotations | Parallelism is compiler-owned by design (tca.md §5.7) |
| Device-side print/assert | Side effects break the pure-lambda model and the race-freedom argument |
| Multi-GPU / streams / CUDA graphs / cooperative grid sync | Single-GPU, single-stream v0; S6 runs full reductions on one CTA by preset policy |
| Sub-warp or 2-D product-coordinate thread groups | Violates the whole-warp assignment rule; re-entry only via a deliberately defined mapping |
| Grammar fuzzing | v1.x tooling; golden tests + edge matrix cover v1 |
| f64 element types | Frontend v0 spec excludes them; no f64 literals |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| IR-01 | Phase 1 | Pending |
| IR-02 | Phase 1 | Pending |
| IR-03 | Phase 1 | Pending |
| IR-04 | Phase 1 | Pending |
| IR-05 | Phase 1 | Pending |
| IR-06 | Phase 1 | Pending |
| IR-07 | Phase 1 | Pending |
| PASS-01 | Phase 1 | Pending |
| PASS-02 | Phase 1 | Pending |
| PASS-03 | Phase 1 | Pending |
| PASS-04 | Phase 1 | Pending |
| PASS-05 | Phase 1 | Pending |
| PASS-06 | Phase 1 | Pending |
| TOOL-01 | Phase 1 | Pending |
| TOOL-02 | Phase 1 | Pending |
| TOOL-03 | Phase 1 | Pending |
| TOOL-04 | Phase 1 | Pending |
| TOOL-05 | Phase 1 | Pending |
| TOOL-06 | Phase 1 | Pending |
| TOOL-07 | Phase 5 | Pending |
| TOOL-08 | Phase 3 | Pending |
| CG-01 | Phase 2 | Pending |
| CG-02 | Phase 2 | Pending |
| CG-03 | Phase 2 | Pending |
| CG-04 | Phase 2 | Pending |
| CG-05 | Phase 3 | Pending |
| RT-01 | Phase 2 | Pending |
| RT-02 | Phase 2 | Pending |
| RT-03 | Phase 3 | Pending |
| RT-04 | Phase 3 | Pending |
| RT-05 | Phase 4 | Pending |
| RT-06 | Phase 4 | Pending |
| RT-07 | Phase 4 | Pending |
| RT-08 | Phase 2 | Pending |
| PAR-01 | Phase 3 | Pending |
| PAR-02 | Phase 3 | Pending |
| PAR-03 | Phase 3 | Pending |
| PAR-04 | Phase 3 | Pending |
| PAR-05 | Phase 3 | Pending |
| PAR-06 | Phase 3 | Pending |
| OPT-01 | Phase 4 | Pending |
| OPT-02 | Phase 4 | Pending |
| OPT-03 | Phase 4 | Pending |
| OPT-04 | Phase 4 | Pending |
| OPT-05 | Phase 4 | Pending |
| OPT-06 | Phase 4 | Pending |
| FE-01 | Phase 1 | Pending |
| FE-02 | Phase 5 | Pending |
| FE-03 | Phase 5 | Pending |
| FE-04 | Phase 5 | Pending |
| FE-05 | Phase 5 | Pending |
| FE-06 | Phase 5 | Pending |
| FE-07 | Phase 5 | Pending |
| VER-01 | Phase 2 | Pending |
| VER-02 | Phase 2 | Pending |
| VER-03 | Phase 2 | Pending |
| VER-04 | Phase 3 | Pending |
| VER-05 | Phase 2 | Pending |
| VER-06 | Phase 2 | Pending |
| VER-07 | Phase 3 | Pending |
| VER-08 | Phase 1 | Pending |
| NUM-01 | Phase 2 | Pending |
| NUM-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 63 total (corrected during roadmap creation — headers previously stated 57)
- Mapped to phases: 63 (see ROADMAP.md phase details)
- Unmapped: 0


---
*Requirements defined: 2026-10-15*
*Last updated: 2026-10-15 after roadmap creation (traceability populated, coverage corrected 57 → 63)*
