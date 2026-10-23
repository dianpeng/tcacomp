# Code Generation

- This is the final pass that takes finalized IR graph to produce backend code

- For now, we are planning to support 2 backends
  - [1] C++ cuda (this version)
  - [2] MLIR     (next version)

- Code Generation must reside in its own modules
  - internal/cg/cxx   => c++ cuda
  - internal/cg/mlir  => mlir version in the future

- Status of this document: COMPLETE DESIGN (quick-261022-u75), reviewed
  against the post-final-lower codebase. The C++ runtime library itself is
  NOT in scope of the codegen task — but §8 pins its full API surface as
  emitted by this backend, so the library task can implement from this doc +
  doc/runtime_objects.md without reading the Go compiler.

# CXX Code Generation

## 0. Ruled decisions (2026-10-22, confirmed by author)

| # | Decision | Ruling |
|---|----------|--------|
| D1 | Emitted artifact | **Full trio**: `<kernel>.cu` (device kernel + `extern "C"` host launcher) + a generated `CMakeLists.txt` linking the runtime library (the stack decision "emit .cu + CMake, not raw nvcc") |
| D2 | Runtime namespace | **`tca::rt`** for every runtime object (matches tca.md §5.11 `tca::rt::LoadTileCta`); `tca::` stays clean for future non-runtime host surface |
| D3 | GMEM access | **Runtime `tca::rt::TensorView` wrapper objects** — generated code NEVER touches raw memory (see §3, the shim mandate) |
| D4 | SMEM declaration placement | **Kernel-top hoist** — one dynamic `__shared__` pool + typed views at kernel top (see D5: static arrays are impossible at our budget) |
| D5 | SMEM mechanism | **Dynamic shared memory** (`extern __shared__` + `cudaFuncSetAttribute` opt-in in the launcher). NOT static arrays: the design SMEM budget MaxCache = 64 KB exceeds the 48 KB static `__shared__` ceiling, and the worked example's lhs window is exactly 64 KB. Opt-in dynamic smem is the only compliant mechanism on sm_86 (per-block opt-in max 101,376 B, queried at runtime — never hardcoded) |
| D8 | SMEM pool sizing | ONE hoisted `extern __shared__` pool for the whole kernel; every SMEM cache is a VIEW at a compiler-assigned offset into it. **v0 (no SMEM RA): assume worst case — zero reuse — pool size = Σ all buffer extents, views at disjoint offsets** (sub-optimal max, correct). A future opt pass performs liveness RA over all SMEM uses, sizes the pool to the MAXIMUM concurrently-live bytes, and lets non-overlapping live ranges share offsets (each cache becomes an overlapping reference into the pool). The emitted shape (one pool + offset views) is FINAL — the RA pass changes only the offsets and the pool size, never the emitter or the IR |
| D6 | Memory shim | **ALL memory (GMEM/SMEM/FRAG) is accessed through `tca::rt` C++ wrapper objects**, never raw pointers. The wrappers take a compiler-emitted **layout descriptor**; v0 emits plain row-major / no-swizzle descriptors. The future GMEM coalescing (address shuffling) and SMEM swizzle (bank-conflict avoidance) passes change ONLY the descriptor the compiler emits — never the emitter, never the IR access nodes |
| D9 | Google-style symbols | Function symbols are PascalCase (quick-261023-cgq): kernel `<Name>`, launcher `<Name>Launch`, combiner `<Kernel>Combine<i>`. The names come from the frontend via the **KindKernelMeta** instrumentation node (Prototype-anchored, DCE-rooted, string payloads; empty payload = derive); the fallback derives PascalCase from the graph name. Invalid identifiers fail closed. Locals are snake_case (Google-compatible): `v<n>`/`c<n>` values, `<dim>_it<n>` iterators, `<tensor>_tile`/`<tensor>_frag` buffers, `grid_map<n>`/`thread_map<n>`/`reducer<n>`/`kinfo` objects — all collision-guarded against kernel parameter names. Tensor params carry `__restrict__` (the restrict-alias rule is structural); the kernel carries `__launch_bounds__(MaxThreadsPerCTA)` |
| D11 | Reducer scratch | **Amended by the R1/R2 ruling (2026-10-23):** the reducer constructions are scalar-only (`WarpReducer<float>` — NO Slots template argument) and their warp staging lives in the COMPILER'S dynamic-SMEM pool: final-lower allocates one `KindMemNewSMem` staging row per warp/global reducer (`#warps × sizeof(T)` bytes, `#warps` = MaxThreadsPerCTA/32 from the tiling provenance) at a disjoint D8 pool offset, and the ctor receives it INLINE as `tca::rt::SmemView<T, 1u>(reinterpret_cast<T*>(tca_smem + N), #warps u)` plus the owning LNT's `ThreadMapping` (R2: warp/lane/count identity is runtime-internal, derived from the map). The runtime keeps NO static `__shared__` scratch. The staging protocol is pinned in doc/rt.md §4.6 |
| D12 | Loader window→axis mapping | The fill's destination window covers the source tensor's TRAILING `dst.Rank()` axes; `baseCoords` carries one coordinate per source axis in order (trailing axes get round bases, leading axes fixed coordinates). tile-lower's tie construction produces trailing windows; a non-trailing window would require an explicit axis-list signature change (pinned in doc/rt.md §4.5) |
| D10 | Comment instrumentation | `cg.Options.Comments` (none/sections/verbose, default sections): section comments synthesized from facts the IR already carries — the LNT header (per-loop dim/soc/tile/LoC from the soc/dosoc/loc/**tile** annotations, the tile annotation added quick-261023-cgq on the same KeyedAttribute channel), loop lines (tile-between rounds vs inside walk), fill loader contracts, reducer bracket lines, the launch header. Verbose additionally stamps hoisted locals with their minting pass (node PassOrigin). No new IR kinds for comments |
| D7 | ~~Slot-IV Combine~~ (DELETED by R1/R3, 2026-10-23) | The slot-indexed accumulator protocol is gone: `Reduce(value)` folds one element per call, `Combine()` returns `T`, and the former slot/re-iteration machinery (LR-13 slot args, the `.Get(slotIVs)` per-use reads) no longer exists. The per-element consumption reads the scalar Combine local directly; the per-element output store is single-owner gated (R2/R3: `thread_map<N>.is_first_in_cta()`) |

- The provided IR has following shape
  - 1) For IRNodes that has corresponding low level IRNode, already lowered
  - 2) Verb, map/reduce has been lowered into low level runtime primitive call
  - 3) SSAUse is not resolved
    - Which means the code generation part needs to generate local symbol on the fly


- CodeGen method
  - 1) Naturally visit all BB in the IR graph
  - 2) For each BB, based on its type generate the BB itself
  - 3) Recursively code generate all statement inside of it
  - All IRNodes naturally have very direct 1:1 mapping

- Runtime object
  - ALL runtime object stays within its own namespace tca
  - To use a runtime object, generate code that has qualified namespace
    - ie tca::XXX

## 1. Input contract — the codegen-ready IR

The backend consumes the product of the DEFAULT pipeline (opt.BuildPipeline,
15 passes, `internal/opt/all`): `*ir.Graph` after final-lower @950 plus its
attached `*ir.ASTKernel` from `g.AST()` (`internal/ir/astcfg.go:386-419`).
Codegen is NOT a pipeline pass — it is a consumer invoked after
`(*opt.Manager).Run` returns. It never mutates the graph.

### 1.1 Structure walk input

- `ASTKernel.Prototype` — `BlockPrototype` (SN -1): live `KindInputParam`
  placeholders (kernel signature) and `KindMemNew` (GMEM temporaries) —
  `astcfg.go:43-55`.
- `ASTKernel.Body` — top-level `BlockGroup` containing, per LNT in allocation
  order, that LNT's LEXICAL Group (quick-261023-lex): `[head items (the old
  prologue region), the embedded loop-root block, after-nest items (no
  separate tail BB)]`; `KindReturn` lands in the LAST LNT's after-nest
  region. The Group renders as ONE C++ scope so the head region's
  declarations scope over the nest — a sibling prologue Group would close
  before the nest opens.
- `ASTBlock` kinds: `BlockGroup` / `BlockSeqLoop` / `BlockCTALoop` /
  `BlockThrLoop`, carrying `Loop/IV/Bound/DoSoC/LoC` (`astcfg.go:73-151`).
- Items are in final placement order — the emitter walks `Items()` and emits
  top-to-bottom. NO reordering, NO scheduling decisions: the schedule pass
  already made every placement decision (this is the seed's "naturally visit
  all BB" step, concretely).

### 1.2 Node whitelist (post-final-lower; anything else = hard error)

Eliminated before codegen (their presence is a pipeline bug — fail closed):
`map`, `reduce`, `iter`, `itercluster`, `iterclusterproject`, `as`,
`runtimegridquery`, `runtimethreadquery`, `inductionvariable`.

Surviving kinds and their emission rules are the table in §6. Source of
truth: `internal/ir/kind.go:32-107`, `internal/ir/catalog.go`.

### 1.3 Kinds of metadata the emitter needs beyond the graph

- **Launch config is NOT an IR node** (LR-15, `finallower/pass.go:22-25`):
  gridDim.x = `Params.MaxCTA`, blockDim.x = `Params.MaxThreadsPerCTA` from
  the tiling record (`internal/opt/tiling/record.go:258-327`, defaults
  100/512). Codegen fetches the record the same way final-lower fetches
  `tiling-provenance` (analysis-cache side table, `Preserved` through
  schedule) or receives it as an explicit `Options` field — the driver
  passes it; the IR carries it only implicitly via mapping portions.
- **SMEM total**: Σ `MemNewSMem` extent consts (bytes) — the v0 worst-case
  pool size (D8; the future SMEM-RA pass shrinks it). The launcher emits it
  as the dynamic-smem size + `cudaFuncSetAttribute` opt-in.

### 1.4 Known driver caveats (affect the test harness, not the design)

- `pkg.Builder.Build()` materializes; the schedule gate refuses materialized
  graphs (`scheduler.go:182`). Codegen fixtures build graphs via
  `internal/ir` constructors directly (the `cmd/astdot` /
  `finallower_test.go` idiom), never through `pkg`.
- There is no CLI and no `internal/syntax` frontend yet — codegen ships as a
  Go API exercised by golden tests (§10). The CLI lands with the frontend.

## 2. Emitted artifact (D1)

```
out/
├── <kernel>.cu        // exactly one translation unit per compiled kernel
└── CMakeLists.txt     // builds it against the tcacomp runtime
```

`<kernel>.cu` layout, top to bottom:

```c++
// generated by tcacomp <version> — deterministic output
#include "tca/runtime.h"          // the runtime library umbrella (§8)

// 1. file-scope device lambdas (one per surviving KindLambda, §6 L-1)
template <typename T>
__device__ static __forceinline__ T tca_lambda_7(T tca_p0, T tca_p1) {
  return tca_p0 + tca_p1;
}

// 2. the kernel
__global__ void tca_kernel_<name>(<signature §5>) {
  <kernel-top declarations: smem pool views (D4), TensorViews (D3)>
  <Body walk: prologue Group → LNT loop roots → epilogue Groups>
}

// 3. host launcher (§9)
extern "C" void tca_<name>_launch(...);
```

Determinism is a hard requirement (the repo carries determinism tests):
the walk order is AST item order, names are assigned by first-emission
counter, and the emitter contains zero map iteration without sorting.
Twin runs over twin graphs must be byte-identical.

## 3. Memory model — the runtime shim (D3/D6)

**Generated code never emits raw memory access.** Every tensor, buffer, and
fragment is a `tca::rt` wrapper object; every read/write is a wrapper
method call carrying LOGICAL coordinates. The wrapper owns the mapping from
logical coordinates to the underlying flattened storage, guided by a
compiler-emitted layout descriptor.

```c++
// v0 descriptor vocabulary (closed; future passes extend it)
namespace tca::rt {
enum class Layout : unsigned { RowMajor = 0 };   // v0: the only layout
// future: Swizzle<XOR128>, Coalesced<...> — chosen by opt passes, emitted
// by codegen as the descriptor argument; the access-site emission is
// IDENTICAL for every layout (the whole point of the shim)
}
```

- **GMEM** (`TensorView<T, Rank>`): constructed in the kernel prologue, one
  per tensor InputParam: `tca::rt::TensorView<const float, 2> lhs_v(lhs, d0,
  d2);` — pointer + runtime dim values (the dim InputParams). Access:
  `lhs_v.Get(i0, i1)` / `out_v.Set(v, i0, i1)`. The future coalescing pass
  shuffles addresses INSIDE `Get/Set` (descriptor-driven) with zero change
  to generated access sites.
- **SMEM** (`SmemView<T, Rank>`): kernel-top typed views over the one
  dynamic pool (D4/D5) — see §6 M-3.
- **FRAG** (`Frag<T, Rank, Dims...>`): per-thread compile-time-shaped
  register tiles, wrapper-owned — swizzle/bank concerns do not apply but the
  wrapper boundary is kept uniform.
- The cooperative loaders (`LoadTileCta`/`LoadTileThread`) move data
  view-to-view THROUGH the same shim on both sides, so a swizzled SMEM
  layout is transparent to loads as well.

This is the memory-side corollary of "runtime objects are opaque": the IR's
access nodes stay logical; physical addressing is a runtime-internal detail
the compiler steers only via descriptors.

## 4. Emitter architecture

```
internal/cg/            // shared driver surface + options (backend-neutral)
internal/cg/cxx/        // the CUDA C++ backend (this document)
    emit.go             // Emit(g, opts) — entry
    walk.go             // ASTBlock/ASTKernel walk
    expr.go             // scalar-expression emission (post-order)
    memory.go           // views, smem pool, fills, loads/stores
    loops.go            // iterator materialization (soc)
    runtime.go          // runtime-object constructions/calls/attributes
    launch.go           // host launcher + CMakeLists
    naming.go           // symbol table + type formatting
```

- **Entry**: `cxx.Emit(g *ir.Graph, opts Options) (*cg.Result, error)` where
  `Result.Files` is `map[string][]byte` (`<name>.cu`, `CMakeLists.txt`).
  `Options`: kernel display name, tiling `Params` (§1.3), target sm (86).
- **Walk**: `Prototype` → signature synthesis; `Body` → items in order,
  recursing into Loop blocks as C++ scopes `{ ... }`. One AST item = one
  C++ statement or declaration; loop blocks materialize as §7.
- **Expression emission**: post-order over the operand DAG. Each
  value-producing node is materialized into a local at its FIRST use site
  (uses are effect- and scope-ordered by construction — def dominates use
  lexically; the verifier guarantees it). Repeated uses of the same node
  reuse the local (the SSA identity IS the C++ identity — no copies).
  Constants: final-lower already deduped literal consts
  (`finallower/consts.go`), so `tca_c<N>` locals are emitted once.
- **On-the-fly naming (the seed's point 3)**: a counter-based symbol table
  `node → local`. Values `tca_v<N>`, consts `tca_c<N>`, views `<param>_v`,
  smem views `tca_s<N>`, iterators `tca_it<N>`, runtime objects
  `tca_r<N>` / `tca_ki` / `tca_gm<N>` / `tca_tm<N>`, lambdas
  `tca_lambda_<nodeID>`, params `tca_p<N>` (lambda) / `<name>` (kernel).
  Counters advance in emission order → deterministic.
- **Fail-closed gates** (before emitting anything): AST attached; whitelist
  §1.2 clean; every ForLoop carries `soc`; every fill carries `loadtile`;
  `ConstTypeOf` present for every const; every SMEM extent is a u32 const;
  total SMEM ≤ the queried device ceiling at LAUNCH time (launcher-side),
  ≤ MaxCache at COMPILE time (compiler-side check).

## 5. Type & opcode mapping

`ElemType` (`internal/ir/types.go:13-25`) → C++:

| TBool | TF32 | TF16 | TBF16 | TI8 | TI16 | TI32 | TU8 | TU16 | TU32 |
|-------|------|------|-------|-----|------|------|-----|------|------|
| `bool`| `float` | `__half` | `__nv_bfloat16` | `int8_t` | `int16_t` | `int32_t` | `uint8_t` | `uint16_t` | `uint32_t` |

- The index domain is u32 throughout (runtime_objects.md §10.2) — loop
  counters, coordinates, map queries are all `uint32_t`; emitted arithmetic
  on them wraps identically to IR semantics.
- f16/bf16 compute-type policy (scalfold's `TypeTrait::Compute` float-
  promotion) is a RUNTIME-library decision deferred to the library task;
  v0's worked example is f32 end-to-end. The emitter passes the declared
  elem type only.
- `BinOpCode` (`node_expr.go:13-33`) → the identical C++ operator token
  (`|| && | ^ & == != < <= > >= << >> + - * / %`); unary → `+ - ! ~`.
  `KindCall` builtins (`internal/ir/builtins.go`):
  - f32 math set → CUDA math verbatim (`expf logf sqrtf rsqrtf fabsf tanhf
    sinf cosf fminf fmaxf fmaf`).
  - `umin`/`umax` → `tca::rt::umin/umax` (the runtime owns the u32 pair,
    runtime_objects.md §4).

## 6. The emission table (node kind → C++)

### Scalar expressions (FamilyScalarExpr — emitted inline or as `tca_v<N>` locals)

| Kind | Emission |
|------|----------|
| `const` | type-formatted literal as `tca_c<N>` local (`0.5f`, `64u`, `true`); type from `ConstTypeOf` |
| `param` | the kernel scalar param's name (its `InputParam` operand) |
| `binop` / `unaryop` | `(a OP b)` / `OP a` per §5 |
| `select` | `(cond ? a : b)` — value-level ternary, NOT a branch (one-BB IR is preserved: no control flow is introduced) |
| `cast` | `static_cast<T>(x)` (T from `CastTypeOf`) |
| `call` | §5 builtin table |
| `tensordimof` | the dim param's value if symbolic (`DimRef` name → kernel dim arg), the literal if the `DimRef` carries a constant extent |

### Variables / kernel I/O

| Kind | Emission |
|------|----------|
| `inputparam` | Prototype block → kernel signature slot (§5 signature): `RoleDim` → `uint32_t <name>`, `RoleScalar` → `<type> <name>`, `RoleTensor` → `const T* <name>` (load-only, verified by absence of stores) or `T* <name>`; `RoleLambda` never reaches the signature |
| `return` | emits NOTHING in the kernel (a CUDA kernel returns void; the Return node is metadata: its operands are the kernel's logical outputs, consumed by the launcher §9) |

### GMEM (through the §3 shim)

| Kind | Emission |
|------|----------|
| `memnew` | Prototype placeholder → a launcher-allocated device buffer (§9); inside the kernel it is just another `TensorView` over the temp pointer |
| `memload` | `T tca_v<N> = <tensor>_v.Get(<axis exprs...>);` — axes are emitted scalar expressions (they may contain dimvar reads + map attribute reads, e.g. the golden's `intrinsicgridmap(index,0)*64 + @d0 + @d0` form, now `RuntimeGetAttribute`-based) |
| `memstore` | `<tensor>_v.Set(<value>, <axis exprs...>);` |

### Loops (§7 for the iterator API)

| Kind | Emission |
|------|----------|
| `forloop` | `for (auto tca_it<N> = <iterator by soc>; tca_it<N>.Has(); tca_it<N>.Next()) { <body items> }` — the `soc` annotation selects seq/cta/thr; `dosoc`/`loc` feed the iterator ctor |
| `dimvar` | the owning loop's iterator value: `tca_it<N>.Index()` (a dimvar's uses are always lexically inside its loop — the scheduler's SN model guarantees it) |
| `bdimvar` | never referenced from bodies; consumed as the loop's bound expression (`tca_it<N>` ctor argument) |

### SMEM / FRAG (LIR memory — the tile-lower product)

| Kind | Emission |
|------|----------|
| `memnewsmem` | NOTHING at its scheduled position. Kernel-top (D4/D5/D8): ONE `extern __shared__ __align__(16) char tca_smem[];` pool for the whole kernel + per-buffer typed views at compiler-assigned offsets: `tca::rt::SmemView<float, 1> tca_s0(reinterpret_cast<float*>(tca_smem + <offset>), <window extents>);` — **v0 offsets: disjoint, in buffer allocation order (zero reuse, worst case); pool size = Σ extents**. The future SMEM-RA pass (D8) re-assigns offsets by liveness overlap and shrinks the pool to max concurrent bytes — same emitted shape, different offset/size numbers. Total pool bytes → launcher (dynamic smem size + opt-in) |
| `fragnew` | per-thread wrapper at kernel top: `tca::rt::Frag<float, 1, 32> tca_f0;` (compile-time extents from the buffer tensor type) |
| `memstoresmem` / `fragstore` (the FILL, `loadtile` attr) | variant call: `loadtile=1` → `tca::rt::LoadTileCta(tca_s0, lhs_v, <src base coords...>);` `loadtile=2` → `tca::rt::LoadTileThread(tca_f0, lhs_v, <src base coords...>);` — src coords = the fill's value `memload` axes (above-tie in-scope coords) PLUS the covered dims' round-base bivs, in TENSOR DIM ORDER (covered dim → its trailing biv operand; above-tie dim → its axis expr). The loader enumerates the window internally, bounds-clamps, and synchronizes (cooperative variant) — runtime_objects.md §6.4/6.5 |
| `memloadsmem` / `fragload` | `T tca_v<N> = tca_s0.Get(<coord exprs...>);` — coords per covered dim (WARP dim: `threadmap index·T + i` already lowered into `RuntimeGetAttribute` + dimvar exprs; SEQ dim: plain dimvar) |

### Runtime objects (final-lower product, `internal/ir/node_runtime.go`)

| Kind | Emission |
|------|----------|
| `runtimekernelinfo` | `tca::rt::KernelInfo tca_ki;` — first statement of the kernel Body Group (its SN-0 position) |
| `runtimegridmapping` | `tca::rt::GridMapping tca_gm<N>(tca_ki, <ndim, portions... const args>);` — LNT prologue, construction args are the const operands verbatim (mechanical literal representation; NOT template parameters) |
| `runtimethreadmapping` | `tca::rt::ThreadMapping tca_tm<N>(tca_ki, ...);` — immediately after the grid mapping |
| `runtimesequentialreducer` / `runtimewarpreducer` / `runtimeglobalreducer` | `tca::rt::SeqReducer<T> tca_r<N>(<init expr>, tca_lambda_<id>);` — and for warp/global (R1/R2, 2026-10-23): `tca::rt::WarpReducer<T> reducer<N>(<init>, <Kernel>Combine<i>, thread_map<N>, tca::rt::SmemView<T, 1u>(reinterpret_cast<T*>(tca_smem + <off>u), <#warps>u));` — `<T>` = the init operand's `ConstTypeOf`/param type; the combiner lambda is the surviving `KindLambda` operand (L-1); the mapping operand is the owning LNT's `ThreadMapping` local; the staging row is the construction's `KindMemNewSMem` operand rendered INLINE over the dynamic pool at its prescan-assigned disjoint offset (the row participates in pool accounting but gets NO kernelTop view declaration — its only textual form is this ctor argument). `<#warps>` = MaxThreadsPerCTA/32, cross-checked against the byte extent |
| `runtimecallmethod` | `reducer<N>.Init();` (first statement of the element loop) / `reducer<N>.Reduce(<value>);` (R1 scalar protocol — the slot dimvars are gone) / `auto tca_v<N> = reducer<N>.Combine();` (returns `T`; right after the reduced-dim walks) — and the per-element store of a reduced value is wrapped in `if (thread_map<N>.is_first_in_cta()) { ... }` (R2/R3: single-owner stores; every other thread computes the element and stores nothing) |
| `runtimegetattribute` | accessor call, pure: `tca_ki.total_cta()`, `tca_ki.total_thread_per_cta()`, `tca_gm<N>.index(<d>)`, `tca_tm<N>.dimension(<d>)` — GVN already deduped repeated reads |

### Lambda (L-1: the one survivor per reducer)

Stateless by construction (final_lower.md's lambda contract — body references
only its own params + constants): hoisted to file scope as

```c++
template <typename T>
__device__ static __forceinline__ T tca_lambda_<nodeID>(T tca_p0, T tca_p1) {
  return <body expression>;   // params → tca_p<N>, consts → tca_c<N>
}
```

Param types come from the lambda's per-parameter declaration types (the
combiner is homogeneous — checked at Reduce lowering). The template-T form
keeps the runtime free to apply the combiner in its compute type (f16
policy deferred, §5).

### Annotation carrier

`keyedattribute` emits NOTHING — it is metadata consumed at its host node
(`soc/dosoc/loc` on forloops, `loadtile` on fills, `init` already unwrapped
by final-lower into the reducer ctor).

## 7. Loop materialization (the `soc` selector)

All loops become runtime iterator objects (tca.md §5.12); the compiler emits
NO for-statements and NO branches of its own — every guard lives inside the
runtime (loop_tile_lower.md §4):

| soc | Block kind | Iterator | Semantics (runtime-owned) |
|-----|-----------|----------|---------------------------|
| 0 | BlockSeqLoop | `tca::rt::SeqIterator(init, step, bound)` | native `base = init; base < max; base += step` rounds; no bailout |
| 1 | BlockCTALoop | `tca::rt::CtaIterator(gm, dosoc, loc, init, step, bound)` | CTA-uniform round walk (step = LoC·T) |
| 2 | BlockThrLoop | `tca::rt::ThrIterator(tm, dosoc, loc, init, step, bound)` | CTA-uniform trip count, per-thread owned chunk/rank |

- Iterator ctor args (as implemented): the LNT's mapping object (by `soc`
  class: grid vs thread), the loop's `dosoc` dimension, its `loc`
  concurrency, the dimvar identity `(init, step)` as literals, and the
  bound = the `bdimvar` max expression (the lowered `umin` chain — F1 —
  emitted verbatim as a u32 expression). `it.Index()` returns the current
  round-base value; ranks (`map.index(d)·T`) are already in-body offset
  terms inside the lowered expressions — never inside the iterator.
- **Guard semantics (tca.md §5.12 "predicated, not exited"):** `Has()` is
  the CTA-UNIFORM round walk — every thread of the CTA executes the same
  iterations, so collective runtime calls inside the body stay uniformly
  reached. Per-thread/per-CTA validity is enforced INSIDE the shim — the
  views' bounds-checked Get/Set and the loaders' tail clamping suppress
  out-of-range effects — never by an emitted branch, never by the iterator
  bailing early. The whole-body no-op is the SEMANTIC contract; predication
  is the mechanism. Trip counts are CTA-uniform by construction (F1 +
  0-based between loops), which makes collective uniform-call safety
  structural (runtime_objects.md §7.2).

## 8. Runtime library surface (the contract for the FOLLOW-UP task)

Everything below is what the emitter generates calls against. Names are
`tca::rt` (D2). One umbrella header `tca/runtime.h`. Header-only,
`inline`/template, C++20 — the scalfold shape
(`/home/ps/Workspace/cuda/project/scalfold`), but with tcacomp's own
mapping/portion model (NOT scalfold's single-shot generator model):

```c++
namespace tca::rt {

// ---- launch & info -------------------------------------------------
// Host-side: validate-and-throw the launch pre-conditions (assert-and-trap:
// gridDim.x multiple of PortionCTA=10; blockDim.x whole warps, multiple of
// PortionWarp=8 warps, >= 256 threads; device SMEM ceiling >= kernel total).
void ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes);

// ---- the user-facing tensor (the launcher contract type, quick-261023-ten) ----
// Tensor wraps DEVICE memory: the data pointer, the runtime shape, and the
// layout descriptor in one object — the mutually understood contract
// between the end user and every generated launcher. Non-owning (the user
// owns the allocation); host-constructible; pass-by-reference.
template <typename T, unsigned Rank>
struct Tensor {
  Tensor(T* data, /*unsigned dims[Rank]*/...);   // row-major, contiguous
  template <typename... Dims> static Tensor From(T* data, Dims... dims);
  T* Data() const;                                // the device pointer
  unsigned Dim(unsigned axis) const;              // the runtime shape
  unsigned Rank() const;
  Layout layout() const;                          // v0: Layout::RowMajor
};

// Host-side validate-and-throw (throws std::runtime_error with the message).
[[noreturn]] void Throw(const char* what);
#define TCA_REQUIRE(cond, what) /* throw unless cond */

// Device-side: the totals supplier (constructed from gridDim/blockDim).
struct KernelInfo {
  __device__ KernelInfo();
  __device__ unsigned total_cta() const;                 // gridDim.x
  __device__ unsigned total_thread_per_cta() const;      // blockDim.x
};

// ---- per-LNT mappings (portion model, runtime_objects.md §2) --------
struct GridMapping {                                      // CTA space
  __device__ GridMapping(KernelInfo ki, int ndim, /*portions...*/);
  __device__ unsigned index(unsigned d) const;            // this CTA's rank in dim d (CTA-uniform)
  __device__ unsigned dimension(unsigned d) const;        // dim d's LoC
};
struct ThreadMapping {                                    // per-CTA thread space
  __device__ ThreadMapping(KernelInfo ki, int ndim, /*portions...*/);
  __device__ unsigned index(unsigned d) const;            // this thread's rank in dim d
  __device__ unsigned dimension(unsigned d) const;
  // warp-specialized helpers (spec'd §7bis; emitted only when a future
  // pass queries them): warp_index(), lane_index(), is_first()
};

// ---- loop iterators (§7) --------------------------------------------
struct SeqIterator { __device__ SeqIterator(unsigned init, unsigned step, unsigned bound); __device__ bool Has(); __device__ void Next(); __device__ unsigned Index(); };
struct CtaIterator { __device__ CtaIterator(GridMapping&, unsigned dosoc, unsigned loc, unsigned init, unsigned step, unsigned bound); /* + Has/Next/Index; Has() is the CTA-uniform round walk */ };
struct ThrIterator { __device__ ThrIterator(ThreadMapping&, unsigned dosoc, unsigned loc, unsigned init, unsigned step, unsigned bound); /* + Has/Next/Index; same uniform contract */ };

// ---- the memory shim (§3, D3/D6) ------------------------------------
enum class Layout : unsigned { RowMajor = 0 /* future: swizzle/coalescing */ };

template <typename T, unsigned Rank> struct TensorView {   // GMEM
  __device__ TensorView(T* data, /*unsigned dims[Rank]*/...);
  __device__ T Get(unsigned coords[Rank]-as-args...);      // layout-driven
  __device__ void Set(T v, coords...);
};
template <typename T, unsigned Rank> struct SmemView {     // SMEM pool slice
  __device__ SmemView(T* base, unsigned extents...);       // + Get/Set
};
template <typename T, unsigned Rank, unsigned... Dims> struct Frag { ... };

// ---- cooperative loaders (tca.md §5.11) ------------------------------
// own ALL synchronization + internal bounds clamping; the compiler emits none
__device__ void LoadTileCta(SmemView& dst, const TensorView& src, unsigned baseCoords...);
__device__ void LoadTileThread(Frag& dst, const TensorView& src, unsigned baseCoords...);
// (StoreTileCta/StoreTileThread: spec'd, v0 loads-only, never emitted)

// ---- reducers (final_lower.md interface; R1/R2 ruling 2026-10-23) -----
template <typename T> struct SeqReducer   { __device__ SeqReducer(T init, T (*fn)(T, T)); __device__ void Init(); __device__ void Reduce(T v); __device__ T Combine(); };
template <typename T> struct WarpReducer  { __device__ WarpReducer(T init, T (*fn)(T, T), const ThreadMapping& map, SmemView<T, 1u> staging); /* same scalar face; warp shuffle + staged warp partials + replicated local fold */ };
template <typename T> struct GlobalReducer{ /* same ctor face; v0-unreachable scaffolding */ };

// ---- u32 pair (runtime_objects.md §4) --------------------------------
__device__ unsigned umin(unsigned a, unsigned b);
__device__ unsigned umax(unsigned a, unsigned b);

}  // namespace tca::rt
```

Contracts the LIBRARY owns (all already spec'd — the library implements,
never redefines): barrier ownership + placement (runtime_objects.md §7),
the sane-mask lane-prefix closed form (§8), tail clamping inside loaders
(§10.3), Combine broadcast semantics (tca.md §5.10), smem ceiling query via
device attributes (never hardcode 101,376). `Slotable` = scalar for slotless
reducers, a slot-addressable value with `Get(ivs...)` when the reducer was
lowered with slot IVs (D7).

## 9. Host launcher + build (per kernel) — the Tensor contract protocol

The kernel lives in the ANONYMOUS namespace (quick-261023-ten): the
launcher is the translation unit's only exported symbol (the scalfold
engine precedent — kernels internal, wrappers the surface). The user-facing
signature speaks `tca::Tensor<T, Rank>` — device memory + shape + layout in
one object — never loose `(ptr, dim, dim, ...)` lists:

```c++
namespace {
__global__ void __launch_bounds__(MaxThreadsPerCTA) <Name>(uint32_t d0, ..., /* raw internal signature */ ...) { ... }
}  // namespace

void <Name>Launch(tca::Tensor<float, 3u>& out, const tca::Tensor<float, 2u>& lhs, const tca::Tensor<float, 2u>& rhs, cudaStream_t tca_stream) {
  // The contract, enforced by the launcher itself: every dim parameter is
  // DERIVED from its first tensor occurrence (signature order) and every
  // other occurrence must AGREE — mismatched shapes are a named hard error.
  const uint32_t d0 = out.Dim(0u);
  TCA_REQUIRE(lhs.Dim(0u) == d0, "lhs.Dim(0) must equal d0 (dim d0)");
  ...
  tca::rt::ValidateLaunch(dim3(MaxCTA), dim3(MaxThreadsPerCTA), <smemBytes>);
  cudaFuncSetAttribute(<Name>, cudaFuncAttributeMaxDynamicSharedMemorySize, <smemBytes>);  // only when > 48KB
  <Name><<<dim3(MaxCTA), dim3(MaxThreadsPerCTA), <smemBytes>, tca_stream>>>(d0, ..., out.Data(), lhs.Data(), rhs.Data());
  TCA_CUDA_CHECK(cudaGetLastError());
}
```

- Read-only tensors arrive as `const tca::Tensor<T, Rank>&`; stored outputs
  as `tca::Tensor<T, Rank>&` (the same read/write classification as the
  kernel signature's const-qualification).
- Scalar parameters pass through by value; a dim parameter NO tensor
  carries stays an explicit `uint32_t` launcher argument (v0 fixtures never
  hit this — every dim appears in at least one tensor).
- `MaxCTA`/`MaxThreadsPerCTA` are BAKED constants (tiling `Params`, §1.3).
- Generated `CMakeLists.txt`: `project(<Name> LANGUAGES CXX CUDA)`,
  `CUDA_ARCHITECTURES` pinned to the target SM (never `native`),
  `CXX/CUDA_STANDARD 20`, `add_library(<Name> OBJECT <name>.cu)` linking
  the tcacomp runtime + `CUDA::cudart`, `--expt-relaxed-constexpr` kept as
  a knob (scalfold needs it).

## 10. Testing

- **Golden tests** (`internal/golden`): emit the `.cu` (+ CMakeLists) for
  the worked examples (matmul_scale with the K-reduce chain; the plain
  store chain) and byte-compare `testdata/*.cu`. Regenerate via
  `-update`, exactly like the IR goldens.
- **Determinism**: twin-graph byte-identical output (the repo's standing
  determinism-test idiom).
- **Unit pins**: naming/formatting tables (§5), the fail-closed gates (§4).
- **E2E compile+run**: deferred to the runtime-library task (needs the
  library to link); when it lands it follows the stack doc's harness —
  `TCACOMP_E2E=1` gated, content-addressed fixture cache, CPU-reference
  comparison, compute-sanitizer. Codegen's job stops at: the emitted `.cu`
  compiles against the real runtime headers once they exist.

## 11. Implementation notes (quick-261022-cxx — pinned by the goldens)

- **Loader coordinate convention** (§6 M-3 fill, as implemented): the
  loader call is `LoadTile*(dst_view, src_view, coords...)` with the merged
  coordinate tuple in TENSOR DIM ORDER — `len(axes) == rank` uses the fill
  source-load's axes verbatim (the FRAG all-axes form); `len(axes) +
  len(bivs) == rank` merges (covered positions take their round-base biv,
  the rest their axis); anything else fails closed (the FRAG
  mixed-with-skips form — the v0 search policy never produces it).
- **Scheduler-anchored lambda items**: the schedule pass anchors the
  surviving combiner lambda's params and body as items (typically in the
  owning LNT's prologue). The kernel-body walk skips the whole lambda
  operand closure — the lambda emits file-scope (§6 L-1).
- **`keyedattribute` items**: the scheduler may anchor annotation nodes as
  items; they are metadata and emit nothing.
- **Dead locals**: scheduler-placed pure items (late-hoisted bound chains)
  can materialize `auto` locals that go unused in their C++ scope. Harmless
  (nvcc elides them); pinned by the goldens.
- **Duplicate attribute reads RESOLVED (quick-261023-cgq)**: the emitter
  CSEs pure item materializations by emitted-text key — two identical
  expressions reuse the first local. (The IR-level alternative, a
  post-final-lower GVN re-run, stays available if ever needed.)
- **Dead pure items RESOLVED (quick-261023-cgq)**: a referenced-closure
  precompute (statement items' operand closure + loop bound chains; dimvar/
  annotation operands sealed — their values emit as inline literals) skips
  scheduler-anchored pure items nothing reads. Const locals renumber
  compactly.
- **RESOLVED (quick-261023-lex)**: the fill-after-reads placement gap is
  FIXED in the schedule pass — rank-r statements now INSERT before their
  block's embedded sub-block (they used to append after it, and the old
  block-granular order net masked the inversion). The matmul golden now
  pins the fill at the TOP of its K-between host body, before every reader.
  The cross-block order net compares ITEM-POSITION PATHS (lexicographic),
  not block DFS-entry indices — required since one lexical Group holds
  items both before and after its embedded nest.

## 12. Out of scope / follow-ups

| Item | Note |
|------|------|
| C++ runtime library implementation | THE next task — implements §8 (this doc + runtime_objects.md are its complete input) |
| GMEM coalescing + SMEM swizzle passes | After codegen + library: low-hanging fruit — emit non-RowMajor descriptors (D6); zero emitter change |
| SMEM liveness-RA pass | After codegen: pool sizing = max concurrently-live SMEM (v0 assumes zero reuse, D8); changes only the assigned view offsets + pool size — the one-pool-plus-views emission shape is final from day one |
| MLIR backend (`internal/cg/mlir`) | Next version; the runtime-object nodes were designed mechanical-literal for exactly this |
| CLI driver (`cmd/tcacomp`) | Blocked on `internal/syntax` (frontend does not exist yet); codegen ships as the Go API |
| f16/bf16 compute-type policy | Runtime-library decision (scalfold `TypeTrait::Compute` precedent) |
| `storetile` variants, `GlobalReducer` reachability, map `warp_index`/`lane_index` queries | Spec'd in runtime_objects.md §11; v0 never emits them |
