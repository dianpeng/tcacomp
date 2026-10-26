# tcacomp — the TCA tensor-kernel compiler

**tcacomp** is an optimizing compiler that turns a deliberately simple, C++-style
tensor kernel language (the TCA frontend, [doc/syntax.md](doc/syntax.md)) into
high-performance CUDA C++, via a single-tier SSA IR called **TCA** (Tensor
Computation Algebra, [doc/tca.md](doc/tca.md)). The generated code links
against a small header-only C++20 runtime ([doc/rt.md](doc/rt.md)) that owns
synchronization, iteration, and communication.

The core value, demonstrated end-to-end on the target GPU: the documented
example kernels (e.g. `matmul_scale`) compile from frontend source through the
TCA IR and its optimization passes to CUDA C++, and run correctly on an NVIDIA
A6000 (sm_86, CUDA 12.2) with numerical results matching a CPU reference — in
one automated flow, with zero manual kernel authoring.

- **Compiler**: pure Go 1.26, standard library only — no cgo, no external
  modules (`go list -m all` is a single line, by design; [doc/design.md](doc/design.md)).
- **Runtime**: C++20 header-only CMake INTERFACE library.
- **Target**: A6000 / sm_86; default launch envelope 100 CTAs x 512 threads;
  the async-copy ceiling on this part is `cp.async` (Ampere) — no TMA.

## Why this design

Most CUDA DSL compilers put the burden in the wrong place: the user annotates
parallelism, hand-tunes fusion boundaries, or the compiler leans on polyhedral
dependence analysis over loop nests. TCA takes the opposite position
([doc/tca.md](doc/tca.md) §1):

- **No explicit control flow.** Loops are *iteration spaces*, not branches and
  back edges. The frontend is a chain of `map`/`reduce` stages over tensor
  index spaces, piped with `|`.
- **Compiler-owned parallelization.** The user never binds iterations to CUDA
  threads or blocks. Every thread group the compiler assigns is a whole number
  of physical warps. Tiling and parallelization are *searched*, not annotated.
- **A single IR for everything.** Fusion, tiling, caching, scheduling, and
  parallelization all operate on one node suite — there is no high/mid/low IR
  conversion machinery. Lowering introduces new node *kinds* into the same
  suite ([doc/design.md](doc/design.md), [doc/ir.md](doc/ir.md) §3).
- **Race freedom by construction.** Correctness comes from structural
  invariants the verifier checks after *every* pass — not from runtime
  sanitizers alone.

The frontend kernel is small enough to state in one example
([doc/syntax.md](doc/syntax.md)):

```
@name(cxx="MatMul")
kernel matmul(
  lhs: tensor<dtype=f32, shape=[M, N]>,
  rhs: tensor<dtype=f32, shape=[N, Q]>,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);
  let output = map on(lhs[a][b], rhs[b][c]) as (a, b, c)
                 (l: f32, r: f32) -> f32: l * r |
               reduce(init=0.0f)
                 (current: f32, %0: f32) -> f32: current + %0;
  return output;
}
```

Locals are immutable (no in-place mutation in v0), types are optional but
strongly checked, and the `|` pipe chains producer stages into fused kernels.

## Architecture overview

```
frontend source (.tca)
      |
      v
internal/syntax            hand-rolled lexer, recursive-descent parser,
  (lexer/parser/ast/       untyped AST + go/ast-shaped visitor,
   semacheck/build)        semacheck rule catalog, module loader (@include)
      |
      v                    AST lowered onto the public rungs
pkg/                       public graph-builder API (Builder, Local, Root)
      |                    Graph.Dematerialize -> Level-1 sea
      v
internal/ir                single-tier SSA node suite (sea-of-nodes L1 +
      |                    scheduled AST CFG L2), verifier, mutation choke points
      v
internal/pipeline          THREE EXPLICIT WINDOWS over the pass registry
  + internal/opt           (dce, simplify, typeinfer, gvn, memopt, tiling,
      |                    tile-lower, schedule, final-lower, ...)
      v
internal/cg/cxx            deterministic CUDA C++ emitter
      |                    (internal/cg: backend-neutral options/result)
      v
<kernel>.cu + CMakeLists.txt     ----links----> runtime/ (tcacomp_runtime)
```

`internal/compiler` is the facade — the single syntax-to-backend API
(`CompileKernel`/`CompileFile`) that `cmd/tcacomp` and the e2e harness call.
Codegen is **not** a pipeline pass: it is a pure consumer invoked after the
manager returns, and it never mutates the graph.

### The pass pipeline (exact, from `internal/pipeline/pipeline.go`)

The orchestrator composes three windows; every window is an explicit pass-ID
list resolved through the registry factories — fail-closed on an unknown ID,
never inheriting whatever else happens to be registered.

| Window | Priority | Pass | Role |
|--------|----------|------|------|
| HIR | 100 | `dce` | early cleanup before anything observes the graph |
| | 101 | `dce-verify` | post-sweep invariant check |
| | 102 | `simplify-lower` | IterCluster format to Iter format |
| | 103 | `simplify-inline` | lambda parameter inlining |
| | 104 | `simplify-verify` | post-simplify invariant check |
| | 105 | `typeinfer` | every node typed after simplification |
| | 110 | `gvn` | global CSE — and **loop fusion, free** |
| | 111 | `gvn-verify` | post-dedup invariant check |
| | 112 | `memopt` | store-to-load forwarding + dead-store elimination |
| | 113 | `memopt-verify` | post-memopt invariant/fixpoint check |
| LOWER | 114 | `tiling` | overlap analysis + combinatorial search + decision record |
| | 115 | `tiling-verify` | decision-record consistency check |
| | 116 | `tile-lower` | tiled loop nests + SMEM/FRAG cache insertion |
| | 900 | `schedule` | sea-of-nodes to AST-style CFG (LICM here) |
| | 950 | `final-lower` | reducers, maps, runtime objects over the AST CFG |
| TAIL | — | `const-dedup` | pipeline-owned literal const dedup |
| | — | `memopt` (2nd instance) | fill-liveness cleanup over the lowered graph |
| | — | `dce` (AST-roots) | collects dead non-item nodes + swept items |

One `opt.Manager` runs the concatenated windows, so declarative preconditions
(`Pass.Requires`) are enforced across window boundaries. A verifier check
follows every transformation pass — the `*-verify` IDs above are the explicit
instances; the IR verifier itself also runs per pass
([doc/design.md](doc/design.md): "verification pass per transformation pass").

## IR design

The IR ([doc/ir.md](doc/ir.md)) is a **single-tier, versioned node suite** —
21 HIR kinds (the abstract kernel graph the builder produces: `map`, `reduce`,
`iter`, the three memory verbs, scalar expressions, ...) and 15 LIR kinds
(lowered `forloop`/`dimvar`/IVs, launch-map queries, runtime-object
constructions; `KindVersion = 7`). There are no separate high/mid/low IR
systems; lowering introduces LIR kinds into the same suite, and the pass
manager rejects kinds it does not know (`>= KindEnd`) — forward compatibility
inside our own pipeline.

**Two levels, one suite.** Between construction and codegen the graph moves
between two forms:

- **Level 1 — sea-of-nodes.** During construction and between passes there is
  no statement list and no implicit program order. Dependencies are explicit
  edges forming a poset: data edges (`Input` operand slots) and effect edges
  (`ImpactBy`). Passes rewrite the sea in place.
- **Level 2 — scheduled AST-style CFG.** Exactly one basic block per kernel
  (v0 has no control flow, hence no PHI nodes), holding an intrusive statement
  list — produced by the `schedule` pass before codegen. Statement order is an
  observable, golden-pinned artifact; *choosing* a good order is an
  optimization decision that belongs to a pass, not to construction.

**SSA handles over one pointee.** Every node owns exactly one `SSAVar`
carrying its def and its live use records; `SSADef`/`SSAUse` are value-type
handles over that shared pointee. Because a use handle points at the *record*
rather than the var, rewiring the record propagates to every handle copy —
and `MutContext` funnels all chain edits through eight choke points
(`ReplaceAllUsesWith`, `Erase`, `SetDef`, ...) that stamp the responsible pass
on every touched node.

**Input/ImpactBy-only convention.** Every reference in the graph lives in
exactly one of two places: the node's Input operand list, or the `ImpactBy`
payload of the three effect bearers (`memload`, `memstore`, `return`). This is
machine-checked by the verifier with a reflect sweep — a new field carrying
hidden SSA references fails verification, not code review.

**Effect model.** A GMEM `memstore` and the kernel's single `return` node are
the only fully effectful kinds; the lowered SMEM/FRAG tier is in-program
memory and therefore dead-code-eligible. Per-tensor side-effect order is
carried by explicit `ImpactBy` chains (reads chained with writes), maintained
by a construction-time `Env` — sound because kernel tensors never alias (the
restrict-alias rule, which also puts `__restrict__` on every tensor parameter
in generated code). The **one end node** (`return`) carries every tensor's
final chain state, so the whole store chain stays live through it.

**Two-root-set DCE.** Liveness at Level 1 means: reachable from the
`RoleTensor` params or the end node. Stores are never roots themselves — an
unreachable store is construction debris, wiped by the first `dce`.

**The verifier.** After every pass: use-def/def-use chain integrity, node
shape checks per kind (fail-closed), loop-scope (a dimvar used only inside
its loop's subtree) and positive-step invariants, the convention sweep, and
the single-end-node rule. Rich violation reports name nodes, slots, and the
last pass that touched a node.

## Optimization passes — the specialized choices

This is where tcacomp deliberately diverges from common CUDA DSL compilers.
([doc/opt.md](doc/opt.md) is the pass-goal overview; deeper specs live in
[doc/optana/](doc/optana).)

### Loop fusion via GVN — there is no fusion pass

`Iter` nodes are **side-effect free**: control-flow regularity is represented
as data dependency, so loop headers are values like any other
([doc/optana/loop_fusion.md](doc/optana/loop_fusion.md)). GVN at priority 110
value-numbers identical loop headers — and two stages over identical headers
with identical operands are the same value. Identical loop nests deduplicate;
**fusion falls out for free**, with no dependence analysis and no dedicated
pass. What fusion makes redundant (the intermediate store/load pair through
GMEM) is then cleaned by `memopt`'s store-to-load forwarding at 112
([doc/optana/memory_opt.md](doc/optana/memory_opt.md)). Producer-to-consumer
data inside a fused kernel flows through pipes (`|`), never through memory.

### LICM via scheduling — there is no standalone LICM pass

The `schedule` pass (priority 900) brings the sea-of-nodes back into an
AST-style CFG by placing every node at a computed scheduling number
([doc/optana/schedule.md](doc/optana/schedule.md)). Invariant code — including
cache fills — is hoisted as a **derived consequence** of placement, not by a
mandated rule: a fill's position is *dominated-by-operand-scope* and is
re-derived after any reshape, so any pass may reorder the graph and the
scheduler still lands every fill inside the scope of its consumers.

### Loop tiling as combinatorial search — not polyhedral

The `tiling` decide pass (114) formulates tile-shape and parallelization
assignment as a **discrete search** over a traffic-avoided objective —
explicitly *not* a polyhedral model, because the IR carries no loop-carried
data dependences that a dependence analyzer would need
([doc/optana/loop_tiling.md](doc/optana/loop_tiling.md)). The search runs
over the innermost-K window of each loop-nest tree; feasibility gates cover
the SMEM byte budget and thread/CTA envelopes; the decision lands in a
side-table record consumed by `tile-lower` (116), which rewrites the nests
and inserts the caches ([doc/optana/loop_tile_lower.md](doc/optana/loop_tile_lower.md)).
The compiler never generates a non-tiling schedule — a legal tile of at least
2 is enforced fail-closed at four gates.

### Compiler-owned parallelization

Users never write thread bindings. The tiling decision assigns iteration
dimensions to the CTA grid and the warp/lanes of a CTA through a **portion
model**: CTA dimensions sum to `PortionCTA = 10`, warp portions sum to
`PortionWARP = 8`, so **every thread group is a whole number of physical
warps** (default envelope 100 CTAs x 512 threads). The mapping is carried by
per-LNT `GridMapping`/`ThreadMapping` runtime objects — the compiler emits
queries (`index`, `dimension`), never inlined linearization arithmetic.

### Structural race-freedom by construction

Parallelism assignment cannot introduce data races because both the program
and the assignment are structural ([doc/tca.md](doc/tca.md) §5.7.8):

1. each structural store indexes its target with exactly the live induction
   variables, each once — distinct iteration points write distinct addresses;
2. values along a reduced dimension meet only inside a **reducer**, whose
   lowering owns all cross-thread communication;
3. in-kernel data flows through pipes, not memory;
4. shared caches are written only by the runtime's cooperative loaders;
5. collectives run at CTA-uniform scope (uniform trip counts + predicated
   tails, never divergent barriers).

The verifier checks these invariants after every pass — the compiler proves
its own output race-free structurally; `compute-sanitizer`
(memcheck/racecheck) in the e2e harness is the confirmation gate, not the
defense.

### Memory IR = three verbs + a storage class

All memory is expressed with exactly **three verbs — `MemNew`, `MemLoad`,
`MemStore` — parameterized by a storage class** (GMEM / SMEM / FRAG). Caches
are not IR primitives: a cache fill is a *composition* of the verbs (an
allocation + a GMEM tile load + a class store), which is precisely what makes
fill placement a re-derivable scheduling decision rather than hardcoded IR.
Hardware-specific instruction selection — mapping a cooperative fill
composition to one `tca::rt::LoadTileCta(...)` or
`tca::rt::LoadTileThread(...)` call — happens **only at codegen**. This
unification (IR kind version 7) deleted the previous six per-tier memory
kinds.

### Explicit storage tiers

The tier model — **GMEM / SMEM / FRAG** — is first-class. The tiling decision
selects which loads are promoted to a cooperative SMEM tile (one CTA-owned
window, filled by a collective loader) or a per-thread FRAG fragment
(registers, filled per-thread), and `tile-lower` materializes the tiered
buffers and rewrites the accesses. Tier selection is driven by the
traffic-avoided objective, thread-varying coordinate analysis, and per-thread
byte budgets — never by user annotation.

## Code generation

The backend ([doc/cg.md](doc/cg.md)) lives in `internal/cg/cxx`:

- **Deterministic, on-the-fly SSA naming** — a counter-based symbol table
  assigns `tca_v<N>` locals at first use; repeated uses reuse the local (SSA
  identity *is* the C++ identity). Twin runs over twin graphs are
  byte-identical — the repo carries determinism tests and golden `.cu`
  fixtures under `internal/cg/cxx/testdata/`.
- **Fail-closed node whitelist** — any HIR verb surviving to codegen is a
  pipeline bug and a hard error, not best-effort emission.
- **All memory access through the runtime view shim** — generated code never
  touches raw memory. Every tensor/buffer/fragment is a `tca::rt` wrapper
  (`TensorView`/`SmemView`/`Frag`) carrying logical coordinates plus a
  compiler-emitted layout descriptor; future coalescing/swizzle passes change
  descriptors, never call sites.
- **One dynamic `__shared__` pool** — every SMEM cache (including reducer
  staging rows) is a view at a compiler-assigned offset into one
  `extern __shared__` pool; the launcher opts into dynamic shared memory via
  `cudaFuncSetAttribute` (the 64 KB design budget exceeds the 48 KB static
  ceiling).
- **Host launcher with the Tensor contract** — the user-facing signature
  speaks `tca::Tensor<T, Rank>` (device pointer + shape + layout in one
  object). Every dim parameter is derived from its first tensor occurrence
  and cross-validated with named **`TCA_REQUIRE`** checks; mismatched shapes
  are a named hard error at launch. `tca::rt::ValidateLaunch` checks the
  launch envelope against *queried* device attributes — the smem opt-in
  ceiling is never hardcoded.
- **The split: the compiler emits, the user builds.** tcacomp writes
  `<kernel>.cu` plus a `CMakeLists.txt` (arch pinned to sm_86, C++20, linking
  `tcacomp_runtime` and `CUDA::cudart`) — it never drives nvcc. The build
  belongs to the user ([doc/design.md](doc/design.md)).

## Runtime

`runtime/` is a **header-only INTERFACE CMake library** (`tcacomp_runtime`,
`runtime/tca/`, CMake >= 3.24, C++20/CUDA 20; [doc/rt.md](doc/rt.md)) — the
scalfold-lineage design ([doc/runtime_objects.md](doc/runtime_objects.md)):

- **`KernelInfo`** — dual-compile CTA/thread-count queries.
- **`GridMapping` / `ThreadMapping`** — the portion decomposition; maps are
  opaque, the compiler never inlines their arithmetic.
- **Iterators** (`Seq`/`Cta`/`Thr`) — one uniform `Has()/Next()/Index()`
  round walk; the compiler emits zero branches around them.
- **Bounds-predicated views** — `TensorView`/`SmemView`/`Frag` clamp or
  predicate out-of-range access; validity is enforced inside the views, not
  by emitted control flow.
- **Cooperative loaders** — `LoadTileCta` (CTA-cooperative, **runtime-owned
  synchronization and tail clamping**, `cp.async`-backed on sm_86) and
  `LoadTileThread` (per-thread, barrier-free).
- **Reducers** — the scalar protocol (`Init()` / `Reduce(value)` /
  `Combine()` returning `T`); warp staging is a row of the *compiler's*
  dynamic-SMEM pool handed in as a `SmemView`, never a runtime-static scratch.
- **Host side** — `Tensor`, `TCA_REQUIRE`, `TCA_CUDA_CHECK`, and
  `ValidateLaunch` with device-attribute queries.

The design principles: the runtime owns **all** synchronization and guards
("predicated, not exited" — collective calls stay uniformly reached), maps
are opaque, no allocation or host calls on device paths, and every accessor
is O(1).

## Getting started

```sh
# build the compiler (pure Go, stdlib only)
scripts/build.sh                 # -> bin/tcacomp

# format Go sources (cmd/, internal/, pkg/)
scripts/format.sh

# compile a kernel: emits <name>.cu + CMakeLists.txt into tcacomp-out/
./bin/tcacomp compile path/to/kernel.tca
./bin/tcacomp compile -o outdir path/to/kernel.tca
./bin/tcacomp help
```

The emitted directory is user-buildable as-is: point CMake at it (the
generated `CMakeLists.txt` links `tcacomp_runtime`), and the runtime headers
come from `runtime/`.

### Tests

```sh
go test ./...                    # hermetic: unit + pipeline + goldens, no GPU
TCACOMP_E2E=1 go test ./internal/e2e/   # GPU e2e: compile, run, compare vs CPU reference
```

The e2e harness is gated behind `TCACOMP_E2E=1` so ordinary test runs stay
hermetic; GPU fixtures are content-addressed under the OS temp dir and the
numerical comparison runs against a Go-computed CPU reference, with
`compute-sanitizer` memcheck/racecheck modes.

## Repository layout

| Path | Responsibility |
|------|----------------|
| `cmd/tcacomp/` | compiler CLI (`compile`, `help`) |
| `cmd/astdot/` | IR-to-Graphviz dump tool |
| `pkg/` | public TCA graph-builder API (Builder rungs, `Dematerialize`) |
| `internal/syntax/` | frontend: hand-rolled lexer, recursive-descent parser, AST, semacheck, module loader, AST-to-IR lowering |
| `internal/ir/` | the single-tier SSA node suite, verifier, mutation choke points, two-level graph |
| `internal/opt/` | optimization + lowering passes (dce, simplify, typeinfer, gvn, memopt, tiling, tile-lower, schedule, final-lower) and the pass manager |
| `internal/pipeline/` | pass orchestrator: the three explicit windows + tail composition |
| `internal/analysis/` | analysis registry (side-effect verdicts, loop-forest) |
| `internal/compiler/` | the single syntax-to-backend facade over parse/build/pipeline/codegen |
| `internal/cg/` | codegen driver surface + options |
| `internal/cg/cxx/` | the CUDA C++ emitter (deterministic naming, fail-closed gates) |
| `internal/e2e/` | GPU e2e harness (TCACOMP_E2E-gated, content-addressed fixtures) |
| `internal/golden/` | golden-test helpers (IR prints, emitted `.cu`) |
| `internal/diagnostic/` | position-attributed diagnostics |
| `internal/flags/` | global compiler flags |
| `internal/log/` | logging facade over `log/slog` |
| `internal/visualization/` | DOT emission for IR graphs |
| `runtime/` | header-only C++20 runtime library (`tcacomp_runtime`) + CPU tests |
| `doc/` | design docs: requirements, frontend, TCA algebra, IR, passes, codegen, runtime |

## Documentation index

- [doc/design.md](doc/design.md) — project requirements and constraints
- [doc/syntax.md](doc/syntax.md) — frontend language and sample
- [doc/syntax-design.md](doc/syntax-design.md) — implementation-ready frontend spec (lexer/parser/semacheck/lowering)
- [doc/tca.md](doc/tca.md) — the TCA algebra: motivation, syntax, rewrite rules, pipeline by example
- [doc/ir.md](doc/ir.md) — implementation-faithful IR reference (two-level design, catalog, SSA, effects, DCE, scheduling)
- [doc/opt.md](doc/opt.md) — optimization pass goals
- [doc/optana/loop_fusion.md](doc/optana/loop_fusion.md) — loop fusion via GVN
- [doc/optana/loop_tiling.md](doc/optana/loop_tiling.md) — tiling as combinatorial search
- [doc/optana/loop_tile_lower.md](doc/optana/loop_tile_lower.md) — tile rewrite + cache insertion
- [doc/optana/schedule.md](doc/optana/schedule.md) — sea-of-nodes to AST CFG scheduling
- [doc/optana/final_lower.md](doc/optana/final_lower.md) — final lowering before codegen
- [doc/optana/memory_opt.md](doc/optana/memory_opt.md) — store-to-load forwarding + dead-store elimination
- [doc/optana/type_inf.md](doc/optana/type_inf.md) — type inference
- [doc/cg.md](doc/cg.md) — CUDA C++ codegen design
- [doc/rt.md](doc/rt.md) — C++ runtime library design
- [doc/runtime_objects.md](doc/runtime_objects.md) — emitted runtime-object trace
