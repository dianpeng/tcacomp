# Architecture Research

**Domain:** Compiler middle-end + GPU DSL compiler (tensor kernel compiler, Go → CUDA C++ + C++20 runtime)
**Researched:** 2026-10-15
**Confidence:** HIGH (patterns from LLVM/MLIR/Triton/Halide are stable, well-documented, and verified against current official sources); MEDIUM on the single-tier-IR legality-gating design recommendation (applied design judgment, no off-the-shelf precedent — flagged where used)

## Standard Architecture

### Reference Architectures Surveyed

| Reference | What tcacomp should take | What tcacomp should reject |
|---|---|---|
| **LLVM new pass manager** (`llvm.org/docs/NewPassManager.html`) | Passes return `PreservedAnalyses` and the PM invalidates the analysis cache; analyses are separate from transformations and lazily computed | Full transitive inter-analysis invalidation machinery — overkill for a single-BB, kernel-sized IR |
| **MLIR** (`mlir.llvm.org/docs/PassManagement/`) | `PassInstrumentation` hooks: IR dump after change, timing, and **automatic verifier after every pass** are callbacks on the pass manager, not per-pass boilerplate; legality/promised-ops model for lowering | Multi-dialect progressive lowering — tca.md forbids separate IR tiers |
| **Triton** (TTIR→TTGIR→LLIR→PTX) | Strict pipeline stages behind `make_ttir`/`make_ttgir`/etc. stage functions; GPU-specific encodings (layouts) added as annotations, not new IRs | The separate IR-per-stage architecture itself — tca.md explicitly mandates one IR |
| **Halide** (PLDI '13) | **Decoupled algorithm vs. schedule**: compute nodes (map/reduce) carry no placement; parallelism/tile/cache decisions are annotations applied by passes | Schedule as user-facing language feature — in tcacomp the schedule is fully compiler-owned (preset table) |

The consensus across all four: **the pipeline is the product**. Transformations are small, declarative, ordered stages; everything cross-cutting (caching, verification, diagnostics, logging) lives in the pass-manager infrastructure, not in individual passes.

### System Overview

```
source text                     (Phase 5 — NOT first)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  FRONTEND (internal/syntax)                                 │
│  lexer → parser → AST → elaboration → graph building        │
│  (Phase 5: parser. Phases 1-4: pkg/ builder API instead)    │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  IR CORE (internal/ir)  — single-tier SSA graph             │
│  CFG(1 BB) → Statement → {iter, map, reduce, load/store,    │
│  seq_for, thread_for, cta_for, loadtile/storetile,          │
│  reducer-call, scalar-expr graph}                           │
│  SSAVar / SSADef / SSAUse with use-def + def-use chains     │
└───────┬──────────────────┬──────────────────┬──────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐
│ ANALYSIS      │  │ OPT PASSES    │  │ VERIFIER (analysis-   │
│ (internal/    │  │ (internal/opt)│  │  family, run by PM    │
│  analysis)    │  │ simplify→merge│  │  after every transform)│
│ race/purity/  │  │ →fusion→      │  └──────────┬───────────┘
│ dep analysis  │  │ linearize→    │             │
└───────┬───────┘  │ parallelize→  │             │
        │          │ cache alloc→  │             │
        │          │ iter lower→   │             │
        │          │ reduce lower→ │             │
        │          │ cache lower   │             │
        │          └───────┬───────┘             │
        │                  │                     │
        └──────────────────┼─────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  CODEGEN (internal/codegen)                                 │
│  IR → CUDA C++ text (.cu) + host launch stub                │
│  Emits: loop nests (via runtime iterators), lambda bodies,  │
│  reducer/tile declarations + call sites. NEVER emits sync.  │
└──────────────┬─────────────────────────────────────────────┘
               │ textual contract: tca::rt:: API surface
               ▼
┌────────────────────────────────────────────────────────────┐
│  RUNTIME (runtime/, C++20, CMake)  — standalone library     │
│  reducers (seq/warp), cooperative tile load/store (+ all    │
│  synchronization), iterators, launch/shape validation,      │
│  per-launch gmem workspace                                  │
└────────────────────────────────────────────────────────────┘

Cross-cutting: internal/pass (PM + instrumentation), internal/diagnostic,
internal/log, internal/flags (single source of truth for constants),
internal/visualization (Graphviz dumps), internal/util.
Public surface: pkg/ (compile entry + options); cmd/ (CLI → pkg/).
```

### Component Responsibilities

| Component | Owns | Talks to | Never talks to |
|-----------|------|----------|----------------|
| `internal/ir` | Node suite, SSA chains, types/shapes, mutation API, per-node-kind legality metadata | pass, analysis, opt, codegen (read-only), syntax (via builder) | codegen does not mutate IR |
| `internal/pass` | Pass manager, pass registry, instrumentation hooks (post-pass verify, IR dump, timing), analysis cache + invalidation | ir, all passes | no IR semantics knowledge |
| `internal/analysis` | Read-only queries on IR (defs/uses, statement clusters, iteration-cluster structure, purity/effect info) | ir, diagnostic | must not mutate IR |
| `internal/opt` | Transformation passes only; each declares consumed/produced node families + invalidation | ir (via mutation API), pass, flags, diagnostic, log | runtime, codegen |
| `internal/codegen` | IR → CUDA text; node-family emission registry; host launch stub | ir (read-only), runtime API contract (header/ID registry), flags | must not invent sync or scheduling |
| `runtime/` (C++) | Reducer specializations, tile copy + ALL barriers, iterators, validation, workspace | generated .cu only (compiled together by test harness) | no Go; no compiler internals |
| `internal/syntax` | Lexer/parser/AST (Phase 5); elaborator → builder calls | pkg/ builder API, ir | opt, codegen |
| `pkg/` | Public `Compile(kernel graph/source, options) → artifacts` surface | internal/* | — |
| `internal/flags` | Global knobs: preset table, tile sizes, budgets, log levels | everyone reads; only CLI/pkg writes | — |

## Recommended Project Structure

Already fixed by design.md; research confirms it, with one addition:

```
tcacomp/
├── runtime/            # C++20 runtime lib, CMake — standalone
│   ├── include/tca/rt/ # public API surface (the Go↔C++ contract, in headers)
│   └── src/
├── pkg/                # public Go API: Compile(), Options, artifact bundle
├── internal/
│   ├── ir/             # node suite, SSA core, builder, legality table
│   ├── pass/           # PM, instrumentation, analysis cache/invalidation
│   ├── analysis/       # read-only IR queries
│   ├── opt/            # one file per transformation pass
│   ├── codegen/        # emission registry, CUDA emitter, host stubs
│   ├── syntax/         # lexer/parser/AST/elaborator (Phase 5)
│   ├── diagnostic/     # structured errors attached to IR positions
│   ├── visualization/  # Graphviz dump of any IR snapshot
│   ├── log/  flags/  util/
├── cmd/tcacomp/        # CLI
└── test/e2e/           # harness: run compiler → nvcc → run → compare vs CPU
```

**Addition recommended:** `runtime/include/tca/rt/` headers are the *machine-readable* half of the codegen↔runtime contract. Codegen emits text against those declarations; a Go-side mirror (a small `internal/codegen/rtapi` registry listing runtime entry points, signatures, and variants) is the *compiler-side* half, so codegen never free-hands a call name that doesn't exist. One e2e test compiles every registered entry point once to catch drift.

### Structure Rationale

- **`opt/` one file per pass:** passes are independently auditable against tca.md sections 5.2–5.11, and each pairs 1:1 with a verifier check. This mapping must stay legible — it is the project's core review surface.
- **`analysis/` mutation-free:** the #1 historical source of compiler bugs is a "read-only" analysis secretly mutating or caching stale pointers (MLIR community experience; analysis-cache corruption surfaces as crashes far from the cause). Make it a package-level rule: analysis imports only `ir`, `diagnostic`, `log`.
- **`pass/` knows IR shape but not semantics:** the PM walks statements/nodes generically; it schedules and instruments. Semantic rules live in `ir` (legality) and `analysis` (queries).

## Architectural Patterns

### Pattern 1: Pass instrumentation runs verification, not the passes themselves

**What:** The pass manager exposes hooks (`BeforePass`, `AfterPass`, `AfterPassFailed`). The verifier, the IR dump (`-dump-ir-after=passname`), and timing are instrumentation callbacks. Transform passes contain zero verification boilerplate — satisfying design.md's "verifier for every transformation pass" requirement structurally instead of by discipline.

**When to use:** always. This is the MLIR model (`PassInstrumentation`), and it makes "every pass has a verifier" mechanically guaranteed.

**Trade-offs:** PM needs enough IR awareness to snapshot/print; the verifier must be written as an analysis-family pass that can check stated invariants generically (each opt pass registers its invariant set).

```go
// internal/pass
type Instrumentation interface {
    AfterPass(name string, snap func() *ir.Graph) error // nil = ok
}
// Registered by default: verifier (from each pass's declared invariants),
// graphviz dump on change, timing.
```

### Pattern 2: Conservative analysis invalidation with an explicit preservation whitelist

**What:** LLVM's new-PM insight: passes declare what they *preserved*, the PM drops everything else. tcacomp should take the simplest sound version: default = invalidate all cached analyses after every transformation pass; a pass may return a preservation set (e.g., "CFG shape unchanged", "SSA chains still valid").

**When to use:** always. Skip LLVM's transitive inter-analysis invalidation graph — kernels are single-BB, analyses are cheap to recompute, and an over-conservative cache beats a stale one. Recompute is the debugging-friendly default.

**Trade-offs:** slightly slower pipelines; in exchange, no stale-cache heuristics to debug. At kernel scale (one BB, dozens of statements) recompute cost is noise.

### Pattern 3: Node-family legality gating — how a single-tier IR avoids the soup

**What:** The central architectural risk the question names: map/reduce/iter (abstract) and seq_for/thread_for/cta_for/loadtile/reducer-calls (lowered) coexist in one node suite. MLIR's answer is dialects plus `ConversionTarget` legality; tcacomp can't have dialects, but it can have the same discipline:

1. **Group node kinds into families** (`abstract` = iter/map/reduce/pipe/load/store; `loop` = seq_for/thread_for/cta_for; `tile-op` = loadtile/storetile; `collective` = reducer Init/Reduce/Combine; `scalar-expr` = lambda body).
2. **The pass pipeline declares a legality window per stage** — which families may *appear* in the IR when this pass runs. The pipeline definition in tca.md §5.2–5.12 maps directly to windows: simplification/merge/fusion/linearization run in the all-abstract window; iteration lowering is the first pass whose window includes `loop`; cache allocation adds `tile-op`; reduce lowering adds `collective`; cache lowering may eliminate `tile-op` in favor of emitted calls.
3. **Every transformation pass declares consumed/produced families** and the PM checks the actual IR against the window after the pass (this doubles as a free verifier increment).

**When to use:** from Phase 1. The legality table is data (a map from pipeline-stage → allowed families), stored in `internal/ir`, checked by `internal/pass`.

**Trade-offs:** costs one up-front design pass over node kinds; buys the guarantee that "lowering introduces new node kinds into the same IR" never degenerates into arbitrary mixing. **This is the load-bearing pattern for the single-tier decision — MEDIUM confidence, applied design judgment; no direct off-the-shelf precedent, but it is a direct transplant of MLIR's legality model.** A cheaper fallback (flag on each node "is this lowered?", checked by one global verifier) loses the per-stage precision and is not recommended.

```go
// internal/ir/legality.go
type Family int // Abstract, Loop, TileOp, Collective, ScalarExpr
var PipelineWindows = map[string]map[Family]bool{
    "simplify":        {Abstract: true, ScalarExpr: true},
    "iter-lowering":   {Abstract: true, Loop: true, ScalarExpr: true},
    "reduce-lowering": {Abstract: true, Loop: true, Collective: true, ScalarExpr: true},
    "codegen-input":   {Loop: true, Collective: true, TileOp: true, ScalarExpr: true},
}
```

### Pattern 4: Schedule-as-annotation (Halide decoupling)

**What:** Compute structure (map/reduce chains, `as` clauses, stores) is never rewritten by scheduling decisions. Parallelism properties (`Sequential`/`Thread(K)`/`CTA(B,K)`), tile sizes, and cache placements are **annotations on iter/store nodes**, written by the preset-selector pass, read by lowering passes. The iteration lowering pass consumes annotations and emits loop nodes; it does not make decisions.

**When to use:** always — tca.md's pipeline already implies this (§5.7 assigns, §5.9–5.11 lower). Make it explicit in the IR: an `Annotations` struct per node, populated in one pass, consumed in later passes.

**Trade-offs:** annotations can drift from the nodes they describe if mutated carelessly — the verifier checks annotation consistency (e.g., "every iter has exactly one parallelism property after §5.7", "every loop node's annotation set is empty after §5.9").

### Pattern 5: Runtime-owned synchronization; codegen-owned structure

**What:** tca.md §5.11 already fixes the contract. Reinforced by both Triton (the runtime/launcher owns barriers, mbarriers, TMA) and scalfold's precedent: **the compiler emits control structure and call sites; the runtime owns every `__syncthreads`, shuffle, and `cp.async` fence.** This makes the race-freedom argument (tca.md §5.7.8) checkable at two independent surfaces: codegen checks structural invariants (CTA-uniform trip counts, predicated tails), the runtime checks its own internal contract.

**When to use:** always; deviating is the single easiest way to silently break the no-race-by-construction model.

## Data Flow

### Kernel Journey (end-to-end)

```
Phase 5:  source text → tokens → AST → elaborator
Phase 1-4: builder API calls (pkg/)          ─┐
                                             ▼
         ┌─── ir.Builder ─────────────────────────────┐
         │ typed SSA graph: statements with iter-      │
         │ clusters, map/reduce chains, scalar exprs   │
         └───────────────┬────────────────────────────┘
                         ▼
   pass pipeline (fixed order, tca.md §5.2→5.11):
   simplify → loop merge → fusion → effect linearization
   → parallelization presets (annotations) → CTA cache alloc
   → iteration lowering (iter→loop nodes, tile factoring)
   → reduce lowering (reducer obj + Init/Reduce/Combine placement)
   → cache lowering (loadtile/storetile → runtime variant selection)
                         ▼
   ┌─── codegen ──────────────────────────────────────┐
   │ per node-family emission registry:                │
   │  loop nodes → runtime iterator loops              │
   │  map → lambda body (scalar expr → CUDA)           │
   │  reduce → reducer decl + Init/Reduce/Combine      │
   │  loadtile/storetile → tca::rt variant call + buf  │
   │  host stub: validate extents, alloc workspace,    │
   │  launch cluster sequence in dependency order      │
   └───────────────┬───────────────────────────────────┘
                   ▼  artifacts: kernel .cu + host stub .cu + runtime path
   test harness: nvcc (runtime + .cu) → run on A6000 → numeric compare vs CPU
```

### Key Data Flows

1. **SSA chain flow:** `SSAVar` is the shared spine (design.md): passes mutate via `ReplaceDef` (all uses follow) and `DeleteDef` (use-def walk for DCE). All passes go through these two mutation entry points — never touch use/def links directly — so chain invariants hold by construction and the verifier only checks semantics, not pointer hygiene.
2. **Annotation flow:** flags/preset table → `opt/parallelize` writes `node.Annotations` → `opt/iter-lower`, `opt/reduce-lower`, `opt/cache-lower` read them → `codegen` reads lowered nodes only (never annotations). One-directional; each lowering pass clears the annotations it consumed.
3. **Diagnostic flow:** every pass reports through `diagnostic` (structured: pass name, IR position, invariant violated, expected vs. actual). The Graphviz dump attaches to the same position so a verifier failure prints IR + graph together. This is the debugging backbone design.md demands ("rich information during verification").
4. **Codegen↔runtime contract flow:** codegen emits only names present in the Go-side runtime-API registry (Pattern above); the e2e harness is the enforcement point that the registry matches `runtime/include/tca/rt/`.

### State Management

Single compile session = one `ir.Graph` mutated in place by the pipeline, with the analysis cache (recomputed conservatively) and the pass instrumentation (verify/dump) layered on top. No concurrency inside one compilation; `flags` is the only global, read-mostly after option parsing (set once by `pkg.Compile`).

## Build Order (mapped to the 5-phase correctness-first plan)

Dependencies between components, and what each phase must deliver so the next can stand on it:

| Phase | Build | Component dependencies | What it proves e2e |
|---|---|---|---|
| **1. IR core + builder + verifier** | `internal/ir` (SSA core, node suite, legality table, builder), `internal/diagnostic`, `internal/log`, `internal/flags`, `internal/visualization`, `internal/pass` skeleton (PM + instrumentation + conservative invalidation), `internal/analysis` first queries, `pkg/` builder API | ir has zero deps; pass depends on ir only; everything else hangs off these two | IR construction/mutation correct; verifier + graphviz dump work (the debugging backbone exists before any compiler logic) |
| **2. Naive codegen + runtime → e2e** | `internal/codegen` (seq_for + scalar-expr emission + sequential reducer calls + host stub), `runtime/` (sequential Reducer, seq iterators, launch validation, gmem workspace), `test/e2e` harness (nvcc via exec) | codegen needs: IR read API, runtime-API registry, flags. Runtime needs: nothing (standalone C++). **This phase intentionally builds the entire Go↔C++ contract surface at its simplest** — both sides of the boundary exist and are tested before any parallelism | `matmul_scale` compiles, runs on A6000, matches CPU reference. Whole skeleton (frontend-skip → IR → passes → CUDA → runtime → launch) exercised |
| **3. Parallelization presets + reduce lowering** | `opt/parallelize` (preset table S1–S6 behind `select(StatementCluster) -> Assignment` interface), `opt/iter-lower` (loop nodes, tile factoring), `opt/reduce-lower` (Init/Combine placement, accumulator tiles); `runtime/` adds warp Reducer (shuffle tree, SMEM partials), thread/cta iterators | preset pass only writes annotations (Pattern 4); iter-lower is the first legality-window expansion (Pattern 3); reduce-lower needs iter-lower's loop structure — **tca.md §5.9's ordering constraint is a hard dependency** | Parallel kernels race-free by construction; verifier checks store-indexing and collective-uniformity invariants after parallelize |
| **4. Fusion/merge + SMEM caches** | `opt/simplify`, `opt/loop-merge`, `opt/fusion`, `opt/effect-linearize` (actually runnable first in Phase 4, or partially in 2 — see note), `opt/cache-alloc` (hoisting rule, FRAG/SMEM, 64 KB budget), `opt/cache-lower`; `runtime/` adds cooperative `LoadTileCta`/`LoadTileThread` (+`cp.async`, barriers) | merge/fusion operate in the all-abstract legality window — they are IR-pure and depend only on Phase 1, which is why they can land late; cache-alloc needs parallelization annotations (storage class derives from loop parallelism); cache-lower needs iteration lowering done | Fusion reduces launches; SMEM caching gives the first real speedup; runtime reuse-safety contract tested |
| **5. Frontend** | `internal/syntax` elaborator → builder calls; then lexer/parser (full EBNF), name/shape resolution, scalar-expr checking with shared builtin/cast registries; `cmd/` CLI | parser output feeds the Phase-1 builder API — the frontend is deliberately the thinnest layer, "sugar over the same IR" (tca.md §5.1) | Source-text kernels reproduce all e2e results from Phases 2–4 |

**Ordering notes / opinions:**

- **Phase 2 is the riskiest phase and must stay lean.** It crosses every boundary at once (Go compiler, codegen, runtime, nvcc, GPU numerics). Resist adding anything beyond sequential lowering + scalar reduce. Every extra pass added to Phase 2 delays the first e2e signal, which is the project's core value gate (PROJECT.md: `matmul_scale` on GPU vs CPU).
- **Merge/fusion passes (tca.md §5.3–5.5) have no dependency on codegen** — they are IR-to-IR rewrites in the abstract legality window. They could run in Phase 2 for trivial cases, but they only matter once parallelism exists (fusion decides launch count, which only exists after Phase 3's cluster model). Keep them in Phase 4; putting them earlier buys nothing e2e-visible.
- **The verifier framework is Phase 1, but individual invariant sets accrue per pass** as passes land. Budget for it: each opt pass's cost is roughly 50% transformation, 50% declaring/checking its invariants.
- **The elaborator (syntax.md §11.1) is mostly built in Phase 1** — the `pkg/` builder API *is* the elaborator's target. Phase 5 adds text parsing on top, not a new IR path.

## Scaling Considerations

For a compiler, "scale" is IR size (statements per kernel, fused-chain depth), kernel count per compile, and pipeline length — not users.

| Scale | Architecture adjustments |
|-------|--------------------------|
| 1 kernel, ≤10 statements (v0, `matmul_scale` + examples) | Everything above is sufficient; recompute-all analyses, dump IR after every pass, no caching cleverness |
| Dozens of statements, deep fused chains | Revisit analysis invalidation whitelist (Pattern 2) — def-use walk queries become hot; add memoized queries per statement cluster. Graphviz dumps become unreadable; add `-dump-ir-after=<pass>` filtering |
| ILP/polyhedra selector replaces preset table (design.md "Future Development") | Already accommodated: `select()` is one interface; annotations are the only channel; lowering passes are untouched. This is Halide-autoscheduler/TVM precedent, deliberately mirrored |
| Multiple kernels per compilation unit | Statement clusters already map to launches; host stub emission is per-frontend-kernel. No change until cross-kernel fusion is on the table (out of scope v0) |

**First bottleneck:** analysis recompute cost as chains deepen — visible first in the verifier (it runs after every pass). **Second:** legality-window checking if node families proliferate — keep the family enum closed and force new node kinds to declare a family at definition time.

## Anti-Patterns

### Anti-Pattern 1: The ungated single-tier soup

**What people do:** one node suite, no legality concept; passes `switch` over whatever nodes happen to be present; abstract and lowered nodes interleave freely.
**Why it's wrong:** this is exactly the failure mode the single-tier decision invites. Passes start defensively coding around states that should be impossible; the verifier can't state invariants ("is `reduce` legal here?" depends on which passes ran); debugging a mis-ordered pipeline becomes archaeology.
**Do this instead:** Pattern 3 — family grouping + per-stage legality windows checked by the PM after every pass. The pipeline becomes data; illegal states are caught at the pass that created them.

### Anti-Pattern 2: Verification written inside transformation passes

**What people do:** each opt pass checks its own invariants inline, ad hoc.
**Why it's wrong:** design.md requires a verifier per transformation pass *with rich output* — inline checks degrade to `if bad { return err }`, dumps don't happen on success paths, and passes forget checks under flag combinations. The LLVM-legacy-PM experience is precisely that passes lie about what they preserved.
**Do this instead:** Pattern 1 — instrumentation runs the verifier after every pass, unconditionally, with graphviz snapshot on failure.

### Anti-Pattern 3: Codegen emitting synchronization

**What people do:** codegen emits `__syncthreads()`, shuffle intrinsics, or fences around generated loops "for efficiency."
**Why it's wrong:** tca.md §5.11's contract ("codegen never emits barriers for caches") exists because the race-freedom argument is only provable if all synchronization is in runtime-audited code. A single codegen-emitted barrier breaks the audit surface and invariant 5 (CTA-uniform collectives) can no longer be checked structurally.
**Do this instead:** runtime owns all sync; codegen emits call sites; if a new sync pattern is needed, it becomes a runtime variant.

### Anti-Pattern 4: Two sources of truth for tile/group/budget constants

**What people do:** tile sizes hardcoded in the preset table *and* echoed as literals in codegen; budget in a flag *and* re-checked in runtime with a different number.
**Why it's wrong:** the budget-fallback rule (tca.md §5.7.6) and store-coalescing warning (§2.7 rule 6) both require the compiler to reason about the same constants the runtime enforces; divergence produces kernels that compile and deadlock or spill.
**Do this instead:** `internal/flags` (or a dedicated `internal/ir/cost` table) is the single owner; preset table, cache allocator, and codegen all read it; the runtime receives values it needs as template arguments / launch parameters, never re-derives them.

### Anti-Pattern 5: Parser-first frontend

**What people do:** build lexer/parser/AST first because it's the visible "compiler" part.
**Why it's wrong:** syntax.md §11.1 explicitly orders elaborator/graph-builder first; a parser against an unproven IR rewrites the IR every time IR design tightens, and there are no IR-level tests to catch it. This is the MLIR community's repeated lesson: build the IR + passes + printer first, parse last (MLIR's own `.mlir` parser postdates its core).
**Do this instead:** the current plan (Phase 5 last) — pkg/ builder API is the elaborator, tested IR-first, parser is a thin sugar layer.

### Anti-Pattern 6: Caching raw analysis pointers across passes

**What people do:** a pass stashes `*analysis.Foo` for a later pass to reuse.
**Why it's wrong:** the MLIR debugging literature's canonical stale-cache failure — symptoms crash passes far from the mutation that invalidated them.
**Do this instead:** all analysis access through the PM (Pattern 2); passes declare what they preserved; anything else is recomputed.

## Integration Points

### External Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Go compiler → artifacts | `.cu` text files + runtime include path written to disk | Compiler never drives nvcc (design.md); test harness does |
| Generated code → runtime | `#include <tca/rt/...>`, calls into `tca::rt` namespace | Contract = runtime headers + Go-side API registry; e2e compile catches drift |
| Test harness → GPU | exec nvcc, run binary, compare vs CPU reference | A6000/sm_86; `cp.async` ceiling, no TMA (hardware limit) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| syntax → ir | builder API calls only | keeps parser thin; elaborator = builder client |
| opt ↔ analysis | PM-mediated: opt requests analyses by name; analysis never sees opt | analysis package imports ir only |
| opt → ir | mutation API (`ReplaceDef`, `DeleteDef`, insert/remove node) only | SSA chain invariants hold by construction |
| pass → verifier | each opt pass registers invariant set at pipeline definition time | "verifier for every pass" is structural, not conventional |
| codegen → ir | read-only | codegen holding a mutator is a code-review reject |
| flags → everything | read after `pkg.Compile` parses options; writes only from pkg/cmd | prevents pass-order-dependent flag mutation |

## Sources

- [Using the New Pass Manager — official LLVM docs](https://llvm.org/docs/NewPassManager.html) (HIGH)
- [The New Pass Manager — LLVM Project Blog (2021)](https://blog.llvm.org/posts/2021-03-26-the-new-pass-manager/) (HIGH)
- [llvm-dev: inter-analysis dependencies thread (2016)](https://groups.google.com/g/llvm-dev/c/4m_Lv3Rfylg) (MEDIUM — historical motivation)
- [MLIR Pass Management — official docs](https://mlir.llvm.org/docs/PassManagement/) (HIGH)
- [The MLIR Transform Dialect (arXiv:2409.03864)](https://arxiv.org/html/2409.03864v2) — lowering legality/pre-post-condition best practice (MEDIUM-HIGH)
- [A tour of MLIR dialect stack — hiraditya.github.io](https://hiraditya.github.io/posts/mlir-dialect-stack-for-ml/) (MEDIUM — practitioner sharp edges)
- [Cornell CS6120: MLIR — progressive lowering](https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/mlir/) (MEDIUM)
- [Triton compilation stages — PyTorch blog](https://pytorch.org/blog/triton-kernel-compilation-stages/) (HIGH)
- [Deep Dive into Triton Internals, Part 3 — add_stages pipeline](https://kapilsh.github.io/posts/deep-dive-into-triton-internals-3/) (MEDIUM)
- [triton-lang/triton Discussion #9051 — FP8 through TTIR→TTGIR→LLIR→PTX](https://github.com/triton-lang/triton/discussions/9051) (MEDIUM)
- [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation — PLDI '13](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) (HIGH — schedule/algorithm decoupling, lowering pipeline order)
- [Tensor Comprehensions: relation to Halide](https://facebookresearch.github.io/TensorComprehensions/halide_integration.html) (MEDIUM — reusing a lowering pipeline per stage)
- In-repo authoritative docs: `doc/tca.md` (IR algebra, pass pipeline, presets, lowering contract), `doc/design.md` (module layout, verification requirements), `doc/syntax.md` §11.1 (elaborator-first ordering), `.planning/PROJECT.md` (constraints, phasing decisions) (HIGH)

---
*Architecture research for: tcacomp — Go CUDA kernel compiler, single-tier SSA IR, correctness-first 5-phase build*
*Researched: 2026-10-15*
