# Runtime Objects Trace

- **Purpose.** Authoritative inventory of every runtime-side object and contract
  the tcacomp compiler emits into the IR or assumes of the C++ runtime library
  (`runtime/` per design.md). This is the seed document for implementing that
  library: the implementer should NOT need to read the Go compiler source.
- **Reader.** The engineer implementing the C++ runtime library.
- **Status legend.**
  - `[emitted]` — the object/contract appears in compiler output TODAY
    (post quick-261021-o61, commits 8f466f3..968e885): an IR node kind, a
    KeyedAttribute annotation, a builtin call, or a compiler-side threshold
    constant the IR encodes.
  - `[spec'd]` — the contract is decided in a spec but its emission lands with
    final-lower/codegen; no IR node exists today.
- **Cross-check discipline.** Every `[emitted]` entry cites its emission site
  (Go file:line or the golden dump `internal/opt/tiling/testdata/
  tile_lowered_matmul.ir`). Every `[spec'd]` entry cites its spec section.
  Nothing in this document is invented; anything not listed does not exist.
- **Golden cross-reference.** `tile_lowered_matmul.ir` is real emitted output
  (worked example at the fix4 §3.3 signature, M=1024/N=2048/K=512): statement
  #163 shows the lowered global index forms (`runtimegridquery(index, 0) *
  256 + @d0(init=0, step=1024) + @d0(init=runtimethreadquery(index),
  step=64)` — CTA coupled walk, flat-rank init, blockDim step — and
  `runtimethreadquery(index, 0) * 32 + @d2(init=0, step=64) +
  @d2(init=runtimethreadquery(lane_id) * 1, step=1)` — WARP rank + lane
  chunk), statements #165/#166 show the FRAG and SMEM fills.
- **fix4 update (2026-10-27).** Rows below are updated to the LANDED
  factorization runtime API (doc/fix/fix4.md §4; runtime/tca/mapping.h,
  iterator.h, launch.h are ground truth): absolute-extent map constructors,
  launch-DERIVED totals, the 4-arg ValidateLaunch, warp-unit loc semantics.
  The portion grammar (PORTION_CTA/PORTION_WARP, −1 remainder,
  assert-and-trap pre-conditions) is DELETED — rows 1-4 reflect that.

## Inventory

| # | Object | Category | Status | Emission site / spec reference |
|---|--------|----------|--------|--------------------------------|
| 1 | Launch dims (DERIVED from loop annotations; MaxCTAPolicy = 512 the gridDim cap) | Launch contract | [emitted] (annotation-derived literals; cg.Options.MaxCTAPolicy) | doc/fix/fix4.md §5.1; internal/cg/cxx/launch.go; internal/opt/tiling/record.go (MaxCTA = 512 policy cap, DefaultParams) |
| 2 | ~~PORTION_CTA / PORTION_WARP~~ (DELETED by fix4 §1.2/§2.1) | Launch contract | [deleted] | fix4 §2.1 Params retirements; grep 'ortion' runtime/ = 0 |
| 3 | ValidateLaunch pre-conditions (4-arg) | Launch contract | DELIVERED | runtime/tca/launch.h:37 — `ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes, unsigned maxCTA)`: block%32==0 && block/32≤32, grid ≤ maxCTA, smem ceiling QUERIED |
| 4 | Feasibility thresholds (Π CTA LoC ≤ MaxCTA, Π warp K ≤ MaxWarpsPerCTA=32) | Launch contract | [emitted] | fix4 §2.3 MALoC; record.go Params; tiling-verify re-checks (fix4 §2.7 — CTA LoC==MaxCTA and portion multiples DELETED) |
| 5 | 1-D normalized launch (gridDim.x / blockDim.x) | Launch contract | DELIVERED | literal dim3 in the emitted launcher (fix4 §5.1); runtime/tca/launch.h |
| 6 | Per-LNT remap / per-LNT exhaustion | Launch contract | DELIVERED | per-LNT GridMapping/ThreadMapping ctors in the goldens; surplus ranks wrap in-shape, guards own the skip (fix4 §3.2) |
| 7 | GridMapping `(kinfo, ndim, absolute LoC... all ≥2)` | Maps | DELIVERED | runtime/tca/mapping.h (class GridMapping); IR node KindRuntimeGridMapping; fix4 §4.1 |
| 8 | ThreadMapping `(kinfo, ndim, absolute K_d... in warp units; ndim 0 admitted)` | Maps | DELIVERED | runtime/tca/mapping.h (class ThreadMapping — WARP dims only, fix4 §1.3; zero-extent admission); IR node KindRuntimeThreadMapping |
| 9 | `runtimegridquery(field, dosoc)` query node | Maps / Intrinsics | [emitted] | internal/ir/node_runtime.go (NewRuntimeGridQuery — flat no-arg + per-dim forms); lower_rewrite.go |
| 10 | `runtimethreadquery(field, dosoc)` query node (incl. zero-operand flat + lane_id forms) | Maps / Intrinsics | [emitted] | internal/ir/node_runtime.go (NewRuntimeThreadQuery / Flat / Lane — fix4 §3.1 lane vocabulary); lower_rewrite.go; lower_cache.go |
| 11 | `warp_index` / `lane_id` / `is_first_in_warp` / `is_first_in_cta` / `warp_count` queries | Maps | DELIVERED | runtime/tca/mapping.h (ThreadMapping members; warp_count = kinfo-DERIVED blockDim/32) |
| 12 | `umin` builtin | u32 builtins | [emitted] | internal/ir/builtins.go:31; materialized at lower_rewrite.go:284-288 |
| 13 | `umax` builtin | u32 builtins | [emitted] | internal/ir/builtins.go:31 (registry; no current materialization site) |
| 14 | ceildiv identity (A + T − 1) / T | u32 builtins | [spec'd] (recipe pinned; used for the lane chunk ⌈T_d/32⌉ at lower_rewrite/lower_cache) | loop_tile_lower.md §3 "Recipe"; fix4 §3.1 |
| 15 | `thr_loop` materialization | Loop primitives | [spec'd] (soc=2 annotation [emitted]) | loop_tile_lower.md §4; annotation lower_rewrite.go |
| 16 | `cta_loop` materialization | Loop primitives | [spec'd] (soc=1 annotation [emitted]) | loop_tile_lower.md §4 (amended 2026-10-21: guarded) |
| 17 | `seq_loop` materialization | Loop primitives | [spec'd] (soc=0 annotation [emitted]) | loop_tile_lower.md §4 |
| 18 | Whole-body no-op on tile_max == 0 | Loop primitives | [spec'd] | loop_tile_lower.md §4 |
| 19 | MemNewSMem (SMEM buffer) | Memory & tile | [emitted] | internal/ir/node_lir_memory.go:158-163; allocated lower_cache.go:349 |
| 20 | FragNew (FRAG buffer) | Memory & tile | [emitted] | node_lir_memory.go:244-249; allocated lower_cache.go:347 |
| 21 | MemStoreSMem fill (cooperative) | Memory & tile | [emitted] | node_lir_memory.go:140-148; emitted lower_cache.go:446 |
| 22 | FragStore fill (per-thread) | Memory & tile | [emitted] | node_lir_memory.go:226-234; emitted lower_cache.go:444 |
| 23 | LoadTileCta runtime loader (variant 1) | Memory & tile | DELIVERED | runtime/tca/loader.h (LoadTileCta — one commit/wait pipeline pair, C9 barriers); variant tag lower_cache.go |
| 24 | LoadTileThread runtime loader (variant 2) | Memory & tile | DELIVERED | runtime/tca/loader.h (LoadTileThread — per-thread, no sync); variant tag lower_cache.go |
| 25 | MemLoadSMemChained (buffer access) | Memory & tile | [emitted] | node_lir_memory.go:121-128; lower_cache.go:474 |
| 26 | FragLoadChained (buffer access) | Memory & tile | [emitted] | node_lir_memory.go:207-214; lower_cache.go:472 |
| 27 | Window sizing (WARP dim → K_d × T_d; SEQ → T_d; C1b FRAG → ⌈T_d/32⌉) | Memory & tile | [emitted] | lower_cache.go (buffer extents); fix4 §2.4/§3.3 — the old #threads_d × T_d group window is superseded |
| 28 | MaxCache = 64 KB (kernel-wide SMEM budget) | Memory & tile | [emitted] | record.go (Params.MaxCache 65536); gated by decide-side eviction (fix4 §2.6) — the lower's halve-then-drop is an unreachable backstop |
| 29 | MaxFragPerThread = 256 B (F6) | Memory & tile | [emitted] | record.go (Params.MaxFragPerThread) |
| 30 | Barrier ownership + placement rules | Barriers | [spec'd] | loop_tile_lower.md §6 "Sync", §4 layering; tca.md §5.11 |
| 31 | Mask contract (thread_map single supplier, lane-prefix closed form, full-warp fast path) | Masks | [spec'd] | loop_tile_lower.md §4 "Mask contract"; fix4 §1.4 (lane-prefix holds under consecutive chunks) |
| 32 | `soc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go (all ForLoop construction sites) |
| 33 | `dosoc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go; numbering: lower_rewrite.go:166-196 |
| 34 | `loc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go — ABSOLUTE semantics per fix4 §4.2 (CTA: LoC; WARP: K_d warp units; coupled walk: blockDim) |
| 35 | `loadtile` KeyedAttribute | Annotations | [emitted] | lower_cache.go:441; consumed by checkNoBarriers lower.go:465-492 |
| 36 | `init` KeyedAttribute (Reduce) | Annotations | [emitted] | internal/ir/node_lambda.go:106 (NewReduce) |
| 37 | Saturating tile_max contract (F1) | Numerical / index | [emitted] | lower_rewrite.go:271-288 |
| 38 | u32 arithmetic domain + u32-extent guard (LO-05) | Numerical / index | [emitted] | lower_rewrite.go:216-241; typeinfer u32 rule internal/opt/typeinfer/rules.go |
| 39 | Tail clamping inside loaders | Numerical / index | [spec'd] | loop_tile_lower.md §4 layering, §7bis "Bounds checking" |
| 40 | KernelInfo runtime object | Delivered | DELIVERED | runtime/tca/mapping.h (struct KernelInfo — total_cta()/total_thread_per_cta() return the launch-DERIVED gridDim.x/blockDim.x); IR: KindRuntimeKernelInfo, first kernel statement |
| 41 | RuntimeGridMapping / RuntimeThreadMapping IR nodes | Delivered | DELIVERED | internal/ir/node_runtime.go (ops = [kinfo, ndim const, shape consts...] — ABSOLUTE extents ≥ 2; final-lower verifier strikes −1; zero-extent ThreadMapping admitted); emitted as tca_gm<N>/tca_tm<N> ctors |
| 42 | Reducer<T> + Seq/Warp/Global specializations | Delivered (R1/R2/R3 ruling 2026-10-23) | `runtime/tca/reducer.h`; IR kinds `runtimesequentialreducer`/`runtimewarpreducer`/`runtimeglobalreducer` | final_lower.md "Reducer Lower" as amended; doc/rt.md §4.6; doc/tca.md §5.10 |
| 43 | RuntimeCallMethod node | Delivered | DELIVERED | final_lower.md "Runtime Object Node"; IR kind `runtimecallmethod`; emitted as reducer Init/Reduce/Combine calls |
| 44 | storetile / StoreTileCta / StoreTileThread variants | Spec'd future | [spec'd] (v0 explicitly out of scope: loads-only) | doc/tca.md §5.11; loop_tile_lower.md §6 "Loads-only scope" |

Counts: 44 inventory entries — delivered/emitted 30 (rows 1, 3-13, 15-29,
31-38, 40-43), [spec'd]/future 13, deleted 1 (row 2, the portion constants).

---

## 1. Launch contract

### 1.1 Launch dims — DERIVED from loop annotations; MaxCTAPolicy the cap [fix4 §5.1]

- **Name:** launch dims (gridDim.x, blockDim.x) + the MaxCTAPolicy constant.
- **Landed API (runtime/tca/launch.h; internal/cg/cxx/launch.go):** the
  emitter walks the ForLoops' `soc/dosoc/loc/tile` annotations, groups per
  LNT, and computes `gridDim.x = max over LNTs of Π CTA LoC` (fail-closed
  CuMAP error on a CTA-less kernel), `blockDim.x = (max over LNTs of
  Π warp K) × 32`, floored at 32 (the ZERO-WARP admission: a schedule with
  no WARP dim launches one warp). LoCs are compile-time constants, so the
  launcher bakes literal `dim3(...)` values into `__launch_bounds__(<derived
  blockDim>)`, the kernel header, `ValidateLaunch(...)`, and the
  `<<<dim3(...), dim3(...)>>>` launch.
- **MaxCTAPolicy** (cg.Options, default 512) is the gridDim CAP the runtime
  validates — it replaces the retired MaxCTA-as-exhaustion and
  MaxThreadsPerCTA Params (fix4 §2.1); tiling's `Params.MaxCTA` (default
  512) is the same cap on the decide side's MALoC bound.
- **Runtime implementation notes:** `KernelInfo.total_cta()` /
  `total_thread_per_cta()` return the launch-DERIVED values (gridDim.x /
  blockDim.x under `__CUDA_ARCH__`) — multi-LNT surplus launches size at
  the max-over-LNT totals, and surplus ranks' coordinates wrap in-shape
  (the decomposition is total for any flat rank) while the compiler's loop
  guards own the skip (fix4 §3.2). NEVER hardcode sm_86 SMEM ceilings —
  query device attributes.
- **Spec reference:** doc/fix/fix4.md §5.1 (launch derivation), §4.3
  (validation), §2.1 (Params retirements).

### 1.2 ~~Portion constants~~ — DELETED [fix4 §1.2/§2.1]

- The PORTION_CTA/PORTION_WARP grammar, the −1 remainder, and the
  assert-and-trap divisibility pre-conditions are DELETED from the compiler
  (grep 'ortion' over runtime/ = 0) and from the tiling Params. The map
  constructors take absolute extents directly (§2 below). This row is kept
  as a tombstone so the numbering below stays stable.

### 1.3 ValidateLaunch — 4-arg, defense-in-depth [DELIVERED]

- **Landed signature (runtime/tca/launch.h):**
  `inline void ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes, unsigned maxCTA)`.
- **Pre-conditions (fix4 §4.3):** `blockDim.x % 32 == 0 && blockDim.x/32 <= 32`
  (whole warps, ≤ 32 of them); `gridDim.x <= maxCTA` (the compiler-emitted
  policy constant); `smemBytes + reserved <= the device opt-in max`,
  QUERIED via `cudaDeviceGetAttribute(
  cudaDevAttrMaxSharedMemoryPerBlockOptin)` — never a hardcoded ceiling.
  The launch VALUES themselves are compiler-derived literals — validation
  is defense-in-depth, not the launch decision.

### 1.4 Feasibility thresholds — MALoC/CuMAP [fix4 §2.3]

- **Emission site:** `internal/opt/tiling` constraints (fix4 §2.3): per LNT
  `Π CTA LoC <= Params.MaxCTA (512)` and `Π warp K <= MaxWarpsPerCTA (32)`;
  at least one CTA-assigned dim (CuMAP — hard error otherwise); tiling-verify
  re-checks the same bounds on the record (fix4 §2.7 — the old
  `CTA LoC == MaxCTA` exhaustion check and warp-portion multiples are
  DELETED).
- **Spec reference:** fix4 §2.3 (constraints), §2.7 (record re-checks).

### 1.5 1-D normalized launch [spec'd]

- **Semantics:** `gridDim.x` = total CTAs; `blockDim.x` = threads/CTA
  (whole-warp multiple). Since everything is remapped per LNT, the simplest
  kernel launch shape suffices.
- **Spec reference:** loop_tile_lower.md §7bis "1-D normalized launch".

### 1.6 Per-LNT remap obligation [spec'd]

- **Semantics:** each LNT constructs its OWN map objects at the start of its
  generated loop nests and destroys them at LNT exit; the next LNT
  remaps from scratch. Portion-sharding ACROSS LNTs is the ONE future place
  chained/shard maps may return.
- **Spec reference:** loop_tile_lower.md §7bis "Per-LNT scoped maps";
  loop_tiling.md "Runtime Object Support Arbitary Remapping".

---

## 2. Maps

Two mapping objects per LNT (allocator vocabulary D3, ruled 2026-10-21):
ONE **GridMapping** over the CTA space, ONE **ThreadMapping** over the
per-CTA thread space — both DELIVERED (runtime/tca/mapping.h) under the
fix4 §4.1 factorization API:

```c++
GridMapping(const KernelInfo&, unsigned ndim, /*unsigned loC...*/);   // absolute CTA counts, all >= 2
ThreadMapping(const KernelInfo&, unsigned ndim, /*unsigned warpsPerDim...*/);  // absolute warp counts, WARP dims only; ndim 0 admitted
```

- Each constructor arg is the dim's ABSOLUTE extent — no portions, no −1,
  no total division. `index(d)` = the tuple coordinate via reverse
  stride-shape decomposition of the flat rank over the ctor shape
  (ThreadMapping: WARP units — the warp rank `index()/32` decomposed);
  the no-arg `index()` = the flat rank (threadIdx.x/blockIdx.x analogs);
  `lane_id()` resolves the lane within the warp.
- ThreadMapping carries WARP dims ONLY (fix4 §1.3 dimensionality ruling):
  the CTA dim's coupled walk uses the no-arg flat rank and needs no slot.
  The ZERO-EXTENT form (ndim 0) admits one-warp launches.
- `warp_count()` = `kinfo.total_thread_per_cta()/32` and `total()` =
  `kinfo.total_thread_per_cta()` — launch-DERIVED values, not shape
  products (surplus launches wrap in-shape; guards own the skip).
- Earlier per-dim chained map description was a design typo — superseded.

### 2.1 GridConcurrencyMap / GridMapping [DELIVERED — see the §2 landed API]

> **fix4 §4.1 amendments to the text below:** the IR query node is
> `runtimegridquery` (internal/ir/node_runtime.go — renamed from
> `intrinsicgridmap`, with the zero-operand FLAT `index()` form added);
> the C++ object is `tca::rt::GridMapping` (runtime/tca/mapping.h),
> constructed `(kinfo, ndim, <absolute LoC...>)` — never "from KernelInfo
> totals + the portion partition"; a CTA dim takes a GridMapping slot only
> (NO ThreadMap slot — its coupled walk uses the flat rank, fix4 §1.3).
> Golden #163: `runtimegridquery(index, 0) * 256 + ...`. The text below
> survives for history.

- **IR emission site (queries):** `intrinsicgridmap(field, dosoc)` —
  `internal/ir/node_intrinsic.go:39-44` (NewIntrinsicGridMap). Emitted at
  five sites in `lower_rewrite.go:generateDimExprs`: single-CTA-loop rank
  (:322), CTA between-loop rank (:357), and (with the thread map) the CTA
  coupled inside walk. Golden: `intrinsicgridmap(index, 0) * 64 + ...`.
- **Signature & operands:** `IntrinsicGridMap(origin, field, dosoc)`;
  `field ∈ {"index", "dimension"}` is a STRING payload riding the
  callNode.name precedent (mixString hashing, internal/ir/value.go);
  `dosoc` is the SINGLE operand, a const u32 = the map-dimension binding
  from the pass's per-LNT numbering (lower_rewrite.go:166-196). Result
  type: **u32**.
- **Semantics:** per-LNT scoped query — the queried object is the
  GridConcurrencyMap constructed for the ENCLOSING LNT nest.
  - `index(d)`: this CTA's rank in dimension d's allocation; **CTA-uniform**
    (every thread of the CTA reads the same value). Used only as an in-body
    offset term: `offset = rank·T + round_base` — never inside an induction
    variable.
  - `dimension(d)`: dimension d's LoC (the number of CTAs mapped to that
    dim).
  - DoSoC numbering: per LNT, counters restart at 0 (outer→inner walk);
  sequential iters get no dimension; a CTA dim takes the next GridMapping
  dimension AND the next ThreadMap dimension (its coupled warp is part of
  the CTA assignment, not double-counted).
- **Runtime implementation notes:** C++ template object over the CTA space;
  constructed at LNT entry from KernelInfo totals + the portion partition
  for this LNT; `index`/`dimension` are cheap accessors. One instance per
  LNT per kernel execution.
- **Spec reference:** loop_tile_lower.md §5, §7bis; loop_tiling.md "Runtime
  Object Support"; final_lower.md (RuntimeGridMapping construction node —
  spec'd).

### 2.2 ThreadMap [DELIVERED as tca::rt::ThreadMapping — see the §2 landed API]

> **fix4 §4.1/§1.3 amendments to the text below:** the IR query node is
> `runtimethreadquery` (internal/ir/node_runtime.go — renamed from
> `intrinsicthreadmap`, with the zero-operand FLAT `index()` and
> `lane_id()` forms added, fix4 §3.1's lane vocabulary); the C++ object is
> `tca::rt::ThreadMapping` constructed `(kinfo, ndim, <absolute K_d...>)`
> in WARP units, WARP dims ONLY (the m+n counting invariant below is
> superseded — the CTA coupled walk needs no slot), zero-extent ndim 0
> admitted; `warp_index()`/`lane_id()`/`is_first_in_warp()`/
> `is_first_in_cta()`/`warp_count()` are DELIVERED (mapping.h).
> The text below survives for history.

- **IR emission site (queries):** `intrinsicthreadmap(field, dosoc)` —
  `node_intrinsic.go:49-54` (NewIntrinsicThreadMap). Emitted at
  `lower_rewrite.go` (single-WARP-loop rank :330, CTA coupled inside rank
  :374, WARP between rank :386) and at `lower_cache.go:461` — the R2
  rank-in-index access `threadmap(d)·T + i` on WARP-covered buffer dims.
  Golden: `intrinsicthreadmap(index, 1) * 64 + @d2(...)`; the buffer access
  in the golden is exactly `threadmap(index, 1)·64 + i`.
- **Signature & operands:** same shape as the grid query: field payload +
  single const-u32 DoSoC operand; result **u32**.
- **Semantics:** per-LNT scoped query over the per-CTA thread space.
  - `index(d)`: the calling thread's rank in dimension d's allocation —
    **per-thread** (CTA-relative, never global: warp allocations are PER-CTA,
    cross-product composition).
  - `dimension(d)`: dimension d's thread LoC.
  - **Tuple coordinates:** `cuda_linear_id` is computed FIRST (linear over
    the 1-D launch), THEN reverse stride-shape decomposition per the
    mapping's shape. The multi-dim remap is runtime-OWNED — the compiler
    never inlines its own copy of the linearization arithmetic.
  - Dimensionality: m coupled + n independent dims (counting invariant:
    p = m CTA dims, q = n independent warp dims; a CTA dim's coupled
    inside-warp counts toward the ThreadMap, not the GridMapping).
- **Runtime implementation notes:** C++ template object over the per-CTA
  thread space; warp ranks are CTA-relative; must provide:
  - `warp_index` / `lane_index` / `is_first` queries for warp-specialized
    reduce (spec'd, §7bis — no IR query node exists yet).
- **Spec reference:** loop_tile_lower.md §5, §7bis "Allocator vocabulary",
  "Tuple coordinates"; loop_tiling.md "Composition Semantics, CTA x WARP".

### 2.3 Argument conventions the lowered expressions rely on

The lowered index expressions consume the maps under these fixed shapes
(golden-verified):

| Form | Meaning | Where |
|------|---------|-------|
| `gridmap(d,"index")·T + round` | CTA-dim round offset (in-body local; stride LoC·T, LoC ABSOLUTE) | lower_rewrite.go (CTA between) |
| `threadmap(d,"index")·T + round` | WARP-dim chunk offset (in-body local; stride K_d·T, rank in WARP units) | lower_rewrite.go (WARP between) |
| `gridmap(d,"index")·T + round + flat + i` with `i: init = threadmap() (no-arg flat), step = blockDim` | CTA dim's coupled inside walk (flat-index form survives; ONLY the step became blockDim — fix4 §3.1) | lower_rewrite.go (CTA inside); golden #163: `@d0(init=runtimethreadquery(index), step=64)` |
| `lane·⌈T_d/32⌉ + j, j ∈ [0, ⌈T_d/32⌉)` | WARP dim's LANE CHUNK inside walk (init `lane_id()·c`, step 1, bound `umin((lane+1)·c, tile_max)`) | lower_rewrite.go (WARP inside); golden #163: `@d2(init=runtimethreadquery(lane_id) * 1, step=1)` |
| `threadmap(d,"index")·T_d + i` | R2 SMEM buffer access index on WARP dim d | lower_cache.go |
| `i − lane·⌈T_d/32⌉` | C1b forced-FRAG access index on a WARP-covered dim (CHUNK-LOCAL — the fragment holds only the lane's chunk) | lower_cache.go (access coords); fix4 §3.3 |

No map query appears in any induction variable; ranks enter ONLY as in-body
offset locals.

---

## 3. Intrinsics

Two LIR node kinds, materialized by the loop-tile-lower pass
(quick-261021-jay), exist only between tile-lower and codegen.

### 3.1 KindRuntimeGridQuery / KindRuntimeThreadQuery [emitted — renamed from the intrinsic family by fix4 T4]

- **Emission site:** node structs in `internal/ir/node_runtime.go`
  (RuntimeGridQuery/RuntimeThreadQuery — the flat no-arg and zero-operand
  `lane_id` forms alongside the per-dim DoSoC form, fix4 §3.1); catalog
  rows in `internal/ir/catalog.go`; print forms `runtimegridquery(index,
  %N)` / `runtimethreadquery(index)` / `runtimethreadquery(lane_id)`
  (internal/ir/print.go); verify shape case (operand count, DoSoC
  constness, field vocabulary — internal/ir/verify.go); typeinfer rule
  (u32 — internal/opt/typeinfer/rules.go); legality-window row
  (legal at StageConstructed by the catalog-iteration rule);
  contentHash/contentEqual (mixString of DoSoC + field, internal/ir/value.go).
  These are the spec's "six companions" (loop_tile_lower.md §5): (1) node
  struct + catalog row, (2) allocNode case, (3) verify shape case,
  (4) typeinfer rules, (5) legality-window row, (6) contentHash/contentEqual.
- **Signature & operands:** per-dim form `ops = [DoSoC const u32]`; the
  flat and lane forms are ZERO-OPERAND (field pinned `"index"` /
  `"lane_id"`); field payload `"index" | "dimension"` on the per-dim form;
  result u32; FamilyAbstract (pure value —
  EffectOf free), so **GVN-CSE is sound and desirable**: repeated
  (DoSoC, field) queries collapse to one node.
- **Semantics:** per-launch deterministic pure values; per-LNT scoped (the
  enclosing LNT's map object). See §2 of this document for the full query
  semantics.
- **Runtime implementation notes:** codegen lowers each query to a map
  accessor call; CSE means at most one accessor per (map, dim, field).
- **Spec reference:** loop_tile_lower.md §5.

---

## 4. u32 builtins

### 4.1 The builtin registry

`internal/ir/builtins.go` is the pure scalar-builtin signature registry —
the single source of truth shared by the builder DSL and typeinfer. The
table is a compile-time constant map, never iterated (no map-order
nondeterminism). Anything not registered (memory access, synchronization,
atomics, thread-index builtins) is NOT a scalar-lambda call.

| Builtin | Arity | Signature | Notes |
|---------|-------|-----------|-------|
| `umin` | 2 | (u32, u32) → u32 | **load-bearing** — see 4.2 |
| `umax` | 2 | (u32, u32) → u32 | registered; no current materialization site |
| `expf`, `logf`, `sqrtf`, `rsqrtf`, `fabsf`, `tanhf`, `sinf`, `cosf` | 1 | (f32) → f32 | device math-lib passthrough; not runtime objects |
| `fminf`, `fmaxf` | 2 | (f32, f32) → f32 | device math-lib passthrough |
| `fmaf` | 3 | (f32, f32, f32) → f32 | device math-lib passthrough |

The f32 set rides the CUDA math library verbatim; the runtime library owns
ONLY the u32 pair (and the spec'd ceildiv identity below).

### 4.2 `umin` — the load-bearing use [emitted]

- **Emission site:** materialized per inside-loop bound by
  `lower_rewrite.go:284-288` (tileMax). Type checked by typeinfer's
  KindCall arm (signature lookup + argument typing,
  internal/opt/typeinfer/rules.go:170-189).
- **Semantics:** saturating tile bound (F1, see §10.1 of this document):
  `off′ = umin(offset, extent)`, `tile_max = umin(T, extent − off′)`.
- **Runtime implementation notes:** plain `min()` on u32; both operands are
  always in-range u32 by construction (extents are u32 — LO-05 guard at
  lower_rewrite.go:226-241).
- **Spec reference:** loop_tile_lower.md §3, §6bis PHASE 3.

### 4.3 ceildiv identity [spec'd recipe, not materialized]

- `ceildiv(A, T) = (A + T − 1) / T` over the existing BinOp div. The 0-based
  element-stride between loops eliminate ceil entirely from the loop bounds
  (bound = raw extent); only min and the primitives' internal ceildiv remain.
- **Spec reference:** loop_tile_lower.md §3 "Recipe".

---

## 5. Loop primitives

The pass lowers loops to three primitive forms (loop_tile_lower.md §4).
The COMPILER emits only the selector annotations (`soc`) on KindForLoop
nodes; the materialization templates land with codegen.

### 5.1 thr_loop — thread-parallel element loop [spec'd; soc=2 annotation emitted]

- **Selector:** `soc = 2` on the ForLoop (all `soc` annotations emitted at
  lower_rewrite.go — the five construction sites).
- **Semantics:** its materialization checks whether the offset global index
  lies within the allowed iteration space; out-of-range iterations become a
  **NO-OP for THE ENTIRE iteration body** (whole-body skip — compiler-inlined
  pure computation included). The unit of no-op is the WHOLE body, never
  individual ops: codegen wraps `body` as a unit (`thr_loop(map, lo, hi,
  body)`), inlining the guard into the primitive's fixed template or
  emitting the body as a `__device__` function.
- **Trip counts:** exact by construction — CTA-uniform; the bailout lives
  INSIDE a loop whose iteration count every thread of the group agrees on
  (`ceil(tile_max / nThreads)`). Scaffolding stays convergent; only bodies
  diverge; the next iteration reconverges.
- **Runtime implementation notes:** fixed template with an internal guard;
  NO branch enters the IR (divergence exists only inside the materialized
  primitives). No compiler predication at all — not even KindSelect.
- **Spec reference:** loop_tile_lower.md §4; tca.md §5.12 (iterator-object
  wrapping: `for (auto it = <cta/thread/seq iterator>(); it.Has(); it.Next())`).

### 5.2 cta_loop — CTA-parallel round loop [spec'd; soc=1 annotation emitted]

- **Selector:** `soc = 1`.
- **Semantics:** **GUARDED, same whole-body no-op as thr_loop** (amended
  2026-10-21: the portion model allocates a full round stride (nCTA × T)
  whether or not the extent fills it, so an out-of-range CTA rank must
  whole-body no-op — the earlier "exact by launch construction, no bailout"
  wording is superseded).
- **Runtime implementation notes:** same template family as thr_loop with
  the CTA map supplying the rank.
- **Spec reference:** loop_tile_lower.md §4 (amended).

### 5.3 seq_loop — single-thread sequential [spec'd; soc=0 annotation emitted]

- **Selector:** `soc = 0`.
- **Semantics:** native bounds; no cross-thread concerns; no bailout needed.
- **Spec reference:** loop_tile_lower.md §4.

### 5.4 Shared structural contracts [emitted]

- **0-based induction:** all between loops enumerate rounds 0-based,
  `for (d = 0; d < extent; d += LoC·T)` (CTA) / `d += K_d·T` (WARP); trip
  count uniform for every CTA/thread of the allocation. NO map index in ANY
  between-round induction variable; ranks appear ONLY as in-body offset
  locals (`offset = rank·T + round_base` — golden #163:
  `@d0(init=0, step=1024)` is the round dimvar, `runtimegridquery(index, 0)
  * 256` the rank term). The INSIDE walks' dimvars DO carry per-thread inits
  by design: the CTA coupled walk's `@d0(init=runtimethreadquery(index),
  step=64)` (flat rank, blockDim stride) and the WARP lane chunk's
  `@d2(init=runtimethreadquery(lane_id) * 1, step=1)` (fix4 §3.1).
- **Induction identity:** `offset = rank·T + round_base` for CTA/WARP dims;
  `index = offset + inside` for the CTA dim's coupled walk and the WARP
  dim's lane-chunk walk (both inside walks ARE induction variables under
  fix4 §3.1 — the per-thread init is the ownership mechanism).
- **No compiler predication:** branches never enter the IR (Q1: one-BB is
  permanent — control flow exists only inside runtime-materialized loop
  primitives).
- **Emitted IR nodes:** KindForLoop(iv dimvar, BoundedDimVar bound, body,
  trailing KeyedAttribute annotations) — `internal/ir/node_loop.go:91-111`;
  the annotation vocabulary the verifier pins: `soc` (0=seq, 1=cta, 2=thr —
  the materialization primitive selector), `dosoc` (the loop's map
  dimension), `loc` (that loop's materialization concurrency).
- **Annotation value table** (emitted per dim kind, lower_rewrite.go
  generateDimExprs):

| Dim kind | Loop | soc | dosoc | loc |
|----------|------|-----|-------|-----|
| SEQ | between + inside | 0 | — | 1 (both) |
| CTA | between | 1 | gridDim | LoC (ABSOLUTE CTA count) |
| CTA | inside (coupled walk) | 2 | — (flat rank needs no slot, fix4 §1.3) | the LAUNCH blockDim |
| WARP | between | 2 | threadDim | K_d (ABSOLUTE, WARP units) |
| WARP | inside (lane-chunk walk) | 0 | — | 1 |

(All dims are tiled, T >= 2 — the 2026-10-24 tile-disable ruling; the
untiled single-loop forms are gone.)

- **Coupled walk (CTA dim's inside, fix4 §3.1):** init = the no-arg flat
  thread query `thread_map.index()`, **step = the launch blockDim**, bound
  = the F1 tile max; threads with flatRank ≥ tile_max take zero trips
  (single ownership). The old portion-quantum `coupledThreads` arithmetic
  and the F2 floor are DELETED (fix4 §1.2 — no "remaining warp portions"
  exist under the factorization).
- **WARP composition semantics (fix4 §3.1):** the between loop strides
  K_d×T_d; the owning warp's 32 lanes split the tile into CONSECUTIVE
  chunks of ⌈T_d/32⌉ (lane l walks [l·c, min((l+1)·c, tile_max)), trip 0
  out of range — the §1.4 lane-prefix mask contract holds). Distinct from
  the CTA dim's coupled walk (blockDim granularity).
- **Spec reference:** doc/fix/fix4.md §1.3, §3.1; loop_tile_lower.md §3,
  §6bis PHASE 3 (superseded stubs therein).

---

## 6. Memory & tile objects

Loads-only scope in v0: candidate sites are MemLoads; stores NEVER route
through the cache (`storetile`/write-through is out of scope). Buffer layout
is plain row-major, NO padding/swizzle in v0.

### 6.1 MemNewSMem — SMEM buffer allocation [emitted]

- **Emission site:** `internal/ir/node_lir_memory.go:150-163`
  (NewMemNewSMem, ops = [extent], StorageSMem class); allocated by
  lower_cache.go:349 with a u32-const byte extent.
- **Signature & operands:** `MemNewSMem(origin, extent)`; the buffer's
  tensor TYPE is bound by the pass itself (SetType — the tile TENSOR type,
  `InternTensor(elem, dims)`; the one documented exception to the no-self-
  typing rule, lower_cache.go:331-353).
- **Semantics (window sizing, fix4 §2.4/§3.3):** per covered dim — WARP
  dim d → `K_d × T_d` elements (one slab per warp coordinate: the
  factorization's cross-warp SHARED window — 32× smaller than the
  superseded `#threads_d × T_d` group window); SEQ dim → `T_d`. Total
  buffer bytes = Π over covered dims × element size. Exactly the decide
  side's window-bytes model (one budget, both halves). Buffer dim names
  ride the tensor's dim names.
- **Lifetime (M3):** ONE buffer per site, declared at the fill point,
  refilled per fill-key epoch; a static declaration reused across refills,
  never re-allocated.
- **Runtime implementation notes:** codegen emits the `__shared__`
  declaration sized by the byte extent; row-major, no swizzle (v0).
- **Spec reference:** loop_tile_lower.md §6 "Buffer materialization", §6bis
  PHASE 4 (R2 sizing).

### 6.2 FragNew — FRAG buffer allocation [emitted]

- **Emission site:** `node_lir_memory.go:236-249` (NewFragNew, ops =
  [extent], StorageFrag class); allocated by lower_cache.go:347.
- **Semantics:** per-thread register fragment; extent = the per-thread
  window product over covered dims — SEQ dims contribute `T_d`; a
  WARP-covered dim under the C1b forced-FRAG precedence contributes
  `⌈T_d/32⌉` (the LANE's chunk, fix4 §3.3 — e.g. the §3.3 worked example's
  lhs at T_k=32: 1 element × 4 B = **4 B per thread**); all-SEQ FRAG sites
  price Π T_d × elem bytes.
- **FRAG preconditions (§6):** the covered tile dimensions must FULLY
  UNROLL — register arrays stay in registers only when all indices are
  compile-time constants after unrolling; a non-unrolled FRAG tile spills to
  local memory, strictly worse than the GMEM load it replaced. Plus the
  per-thread byte budget (§6.7).
- **Runtime implementation notes:** codegen emits a per-thread array /
  register block; no synchronization.
- **Spec reference:** loop_tile_lower.md §6; doc/tca.md §5.8.

### 6.3 The fill store — MemStoreSMem / FragStore [emitted]

- **Emission site:** `node_lir_memory.go:130-148` (NewMemStoreSMem,
  ops = [value, dst, bivs...]) / `:216-234` (NewFragStore); emitted by
  lower_cache.go:441-447 with the variant annotation and round-base bivs.
  Golden: `memstoresmem(%3[@d2(init=0, step=16384)][loadtile=1], %52)`.
- **Signature & operands:** value = the tile-shaped source load
  (a MemLoad on the ORIGINAL tensor at the fill position's coordinates,
  chained onto the source tensor's effect chain — lower_cache.go:417-427);
  dst = the buffer; trailing biv operands = the covered dims' ROUND BASES
  (between dimvars only — CTA-uniform for the cooperative loader); trailing
  KeyedAttribute = the `loadtile` variant tag (§9.4).
- **Fill placement (R1, variant-dependent — the load-bearing rule):**
  - **LoadTileCta (SMEM window class):** at the innermost covered dim's
    BETWEEN-round body, immediately before the tile-inside loops;
    CTA-UNIFORM coordinates ONLY (round bases; never a per-thread ThreadMap
    term). A WARP dim above the tie has no uniform coordinate — the site
    fails closed.
  - **LoadTileThread (FRAG class):** INSIDE the element loop that DEFINES
    its per-thread coordinate — the inside loop of the innermost above-tie
    INS dim (the fill key; e.g. lhs refills per m-element inside M.inside);
    degenerate fallback (no above-tie INS dim): the innermost covered dim's
    between loop.
  - ONE structural splice mechanism for both variants (the F8 mechanism);
  the old `bodyRoot == load` special case is deleted.
- **Coordinate construction (fill axes):** covered tile dims contribute NO
  axis (the loader enumerates them internally — the compiler emits no loop
  for the fill); above-tie INS dims contribute their in-scope coordinates —
  baked as-is for FRAG, uniformly substituted (round bases) for SMEM.
- **Spec reference:** loop_tile_lower.md §6bis PHASE 4 (FILL PLACEMENT,
  R1/R2); lower_cache.go header comment (R1 ruling, verbatim).

### 6.4 LoadTileCta — cooperative CTA loader [spec'd; variant tag emitted]

- **Selector:** `loadtile = 1` annotation on the fill store
  (lower_cache.go:80, :441).
- **Semantics:** ONE cooperative load per round over the group window:
  CTA-uniform source base (the round bases), DISJOINT per-thread stripes
  enumerating the window, TAIL CLAMPING inside the loader (a partial tile
  gets filled — this is why the fill must run OUTSIDE the bailing element
  loop, with all threads participating). Uniform call rule: called by every
  thread of the CTA the same number of times. Classification: ANY covered
  dim WARP-assigned → SMEM window class; ANY covered dim CTA-assigned →
  site REFUSED (no coherent fill variant under v0 mechanics).
- **Runtime implementation notes:** all-threads copy cooperatively (cp.async
  on sm_86 where available per the runtime's own capability check); the
  variant implements the copy AND all synchronization; internal bounds
  clamping. Named `tca::rt::LoadTileCta<...>` in tca.md §5.11.
- **Spec reference:** loop_tile_lower.md §6bis PHASE 4 (R2); doc/tca.md
  §5.11 (variant table + runtime contract).

### 6.5 LoadTileThread — per-thread loader [spec'd; variant tag emitted]

- **Selector:** `loadtile = 2` (lower_cache.go:81).
- **Semantics:** per-thread copy into the FRAG; the fill is an element-level
  op that sits INSIDE the element loop defining its per-thread coordinate
  (R1 clarification: a per-thread fill is NOT a cooperative op, hence not
  bound by the outside-the-element-loop rule); refills once per epoch of its
  fill key. NO synchronization.
- **Runtime implementation notes:** plain per-thread copy loop; no barrier.
  Named `tca::rt::LoadTileThread<...>`.
- **Spec reference:** loop_tile_lower.md §4 (CLARIFIED R1), §6; tca.md §5.11.

### 6.6 Buffer accesses — MemLoadSMemChained / FragLoadChained [emitted]

- **Emission site:** `node_lir_memory.go:114-128` /
  `:204-214`; emitted by lower_cache.go:470-475.
- **Signature & operands:** ops = [buffer, coords...]; coords per covered
  dim in tensor order — WARP dim (SMEM class): `threadmap(d,"index")·T_d +
  i` (R2 rank-in-index; d = the dim's slot in the LNT's ThreadMap — NOT a
  hardcoded 0, NOT the bare inside coordinate); SEQ dim: its plain inside
  coordinate. FRAG buffers: per-thread `i` per SEQ dim; a C1b WARP-covered
  dim indexes CHUNK-LOCAL (`i − lane·⌈T_d/32⌉` — the fragment holds only
  the lane's chunk; GPU-proven 2026-10-27: the raw `i` form read past the
  fragment under memcheck).
- **Ordering:** every access chains after the fill via the buffer's effect
  chain (ImpactBy edge), so the fill dominates its reads in the sea's
  explicit effect order — this chain IS the F8 structural splice mechanism
  that makes the fill a member of its host loop's body subtree.
- **Spec reference:** loop_tile_lower.md §6bis PHASE 4 (access rewrite, M4:
  the existing KindMemLoadSMem/KindFragLoad kinds are reused).

### 6.7 Budgets — MaxCache, MaxFragPerThread [emitted]

- **Emission site:** `internal/opt/tiling/record.go` — MaxCache (default
  **65536 = 64 KB**, the project's SMEM design budget) and MaxFragPerThread
  (default **256 bytes/thread**, F6); gated kernel-wide (R3: Σ SMEM window
  bytes over ALL LNTs' promoted sites ≤ MaxCache; F6: Σ per-thread FRAG
  bytes ≤ MaxFragPerThread) by decide feasibility, tiling-verify, and the
  lower's budget resolution (lower_cache.go:98-209). Fallback: halve the
  offending site's covered tiles with a floor of 2 — a site whose halved
  covered dim would fall below 2 is DROPPED to GMEM (F4, floor amended
  2026-10-24: tile 1 is not a legal schedule).
- **Runtime implementation notes:** assert the device SMEM ceiling (via
  device attributes, NEVER hardcode sm_86's 101,376 B) ≥ the compiled
  kernel's budget.
- **Spec reference:** loop_tiling.md "SMEM constraints"; loop_tile_lower.md
  §6 "Budget enforcement".

### 6.8 Spec'd-future memory objects

- **storetile / StoreTileCta / StoreTileThread** [spec'd]: matching store
  variants (store ordering contract: waits until all writes to the tile are
  finished before copying out). v0 NEVER routes stores through the cache —
  no `storetile` annotation is emitted today. Spec: tca.md §5.11;
  loop_tile_lower.md §6 "Loads-only scope".

---

## 7. Barriers

### 7.1 Ownership [spec'd]

- **The runtime owns ALL synchronization/barriers; the compiler emits
  NONE.** The tile-lower pass pins this structurally: `checkNoBarriers`
  (internal/opt/tiling/lower.go:465-492) parses the live `loadtile`
  attribute back out of every emitted fill's operand list and asserts no
  barrier node exists — never by re-reading a baked flag (F5: real
  structural self-checks only).
- **Spec reference:** loop_tile_lower.md §6 "Sync", §7bis "Sync ownership".

### 7.2 Placement rules [spec'd]

| Rule | Statement |
|------|-----------|
| Cooperative fill → barrier → first access | A cooperative `loadtile` needs ALL threads of the group to participate (this is how a partial tile gets filled); visibility must be established before any read. |
| Never inside a bailing element loop | Tile-level cooperative ops are emitted OUTSIDE the bailing element loops (§4 layering); a cooperative tile op inside a thr_loop body is a spec violation — checked structurally (lower.go §8 self-checks). |
| Reuse safety (write-after-read) | A cooperative refill waits until all reads of the previous contents are finished. Double buffering can later be added inside the runtime WITHOUT IR changes. |
| Store ordering | A cooperative `storetile` waits until all writes to the tile are finished before copying it out. |

- **Runtime implementation notes:** `__syncthreads()` (or pipeline-group
  equivalents for cp.async) inside the cooperative variants only; warp-
  synchronous ops use map-derived masks (§8), not barriers.
- **Spec reference:** loop_tile_lower.md §4 layering, §7bis "Tile-op
  obligations"; tca.md §5.11 runtime contract (Visibility / Reuse safety /
  Store ordering).

---

## 8. Masks

### 8.1 The mask contract [spec'd]

- **thread_map is the SINGLE supplier of masks.** ANY runtime object that
  needs a participation mask gets it from the map. No other mask source
  exists; the compiler never emits mask arithmetic.
- **Sane-mask properties (all three REQUIRED):**
  - **EXACT** — names precisely the lanes that will execute the collective.
  - **UNIFORMLY DERIVED** — computed only from map-uniform quantities
    (base, tile_max); lanes can never disagree about the mask (disagreement
    is the actual UB mechanism — `__shfl_down_sync(mask, ...)` is UB if any
    lane named in mask does not execute the instruction).
  - **PER-ITERATION** — a function of the CURRENT iteration's
    (base, tile_max), read from loop-scaffolding state, not a loop
    invariant.
- **Spec reference:** loop_tile_lower.md §4 "Mask contract".

### 8.2 Lane-prefix closed form [spec'd]

Ownership is lane-strided and the bound is warp-uniform, so the active set
is always a LANE PREFIX:

```
mask = (tile_max − base ≥ 32) ? 0xFFFFFFFF : ((1u << (tile_max − base)) − 1)
```

Pure arithmetic — **NO `__ballot_sync` / `__activemask`** is needed to
derive it; deterministic, divergence-proof, CPU-testable.

### 8.3 Full-warp fast path [spec'd]

Within a full tile the mask is 0xFFFFFFFF — the warp-multiple LoC rule
guarantees non-tail iterations get the fast path for free.

### 8.4 Reducer composition [amended by R1/R2/R3, 2026-10-23]

Reducer placement composes per output element: the R3 element loop hosts
[Init -> reduced-dim walks (Reduce(value)) -> Combine], so `Combine`'s warp
down-reduce is called uniformly by every CTA thread once per element. The
down-reduce runs over the FULL warp (out-of-coverage lanes hold the
identity initializer from their saturated zero-trip walks — the arithmetic
mirror gate — so a full `0xffffffff` shuffle mask is legal); lane 0 stages
the warp partial at its `warp_index()`, `__syncthreads` follows, and every
thread folds all staged warp partials in ascending order from the identity
(the replicated-local-combine broadcast: every participant receives the
element's final value, and the single-owner store gate
`is_first_in_cta()` picks the writer).

---

## 9. KeyedAttribute annotation inventory

The mechanism: `KindKeyedAttribute` — a named attribute riding an Input
list, `keyedattribute(key=value)` print form
(internal/ir/node_iterfmt.go:69-70, print.go). Every annotation below is a
u32-const-valued KeyedAttribute unless noted.

| # | Key | Host node | Producer | Consumer | Value semantics |
|---|-----|-----------|----------|----------|-----------------|
| 1 | `soc` | KindForLoop (trailing) | tile-lower, `lower_rewrite.go` (all ForLoop construction sites) | codegen primitive selection (thr_loop/cta_loop/seq_loop); tile-lower self-checks (`lower.go:268`) | 0 = seq, 1 = cta, 2 = thr — the materialization primitive selector |
| 2 | `dosoc` | KindForLoop (trailing) | tile-lower, `lower_rewrite.go` (numbering: `:166-196`) | codegen map binding; tile-lower self-checks (`lower.go:276`) | the loop's map-dimension binding (per-LNT DoSoC slot; absent on SEQ loops) |
| 3 | `loc` | KindForLoop (trailing) | tile-lower, `lower_rewrite.go` | codegen materialization concurrency; tile-lower self-checks (`lower.go:272`) | the loop's materialization concurrency — per the value table in §5.4 of this doc (SEQ: 1; CTA between: LoC; CTA inside: coupled threads; WARP between: LoC×32; WARP inside: 1) |
| 4 | `loadtile` | fill store trailing operand (MemStoreSMem / FragStore) | tile-lower PHASE 4, `lower_cache.go:441` | the runtime's loader selection (codegen, spec'd); tile-lower's `checkNoBarriers` parses it back out today (`lower.go:465-492`) | copy variant: 1 = LoadTileCta (cooperative, SMEM window), 2 = LoadTileThread (per-thread, FRAG) |
| 5 | `init` | KindReduce (operand 1) | builder, `internal/ir/node_lambda.go:106` (NewReduce) | reducer lowering (final_lower.md — spec'd): `Reducer.Init(T init = <value>)` | the user-provided reduction init value (typed, not u32) |

Notes:

- The `soc`/`dosoc`/`loc` triple is M1 (implementer default): the
  annotation mechanism for loop materialization metadata, so codegen
  selects the cta_loop/thr_loop/seq_loop materialization.
- The `loadtile` value rides the fill store's TRAILING operand slot after
  the round-base bivs; the runtime selects the loader from it (the tca.md
  §5.11 variant table).
- No other KeyedAttribute keys are emitted anywhere in the compiler
  (grep-verified over `internal/`).

---

## 10. Numerical / index contracts

### 10.1 The saturating tile_max contract (F1) [emitted]

- **Form:** `off′ = umin(offset, extent)`; `tile_max = umin(T, extent − off′)`.
- **Emission site:** `lower_rewrite.go:271-288` (tileMax); golden bounds
  chain visible in every lowered inside loop's BoundedDimVar.
- **Semantics:** pure u32, wraps nowhere. The clamp BEFORE the subtract is
  load-bearing: the bare `extent − offset` WRAPS in u32 on tail rounds (an
  out-of-range CTA rank 3168 against extent 1024 computes 4294965152, the
  min picks T, and the inside loop RUNS its body out of bounds). With the
  clamp: **`tile_max == 0` exactly when the rank is out of range** — bounds
  ONLY; tail guarding stays the loop primitive's whole-body no-op, never an
  IR branch.
- **Spec reference:** loop_tile_lower.md §3, §6bis PHASE 3 (F1, ruled
  2026-10-21).

### 10.2 u32 arithmetic domain [emitted]

- All materialized bound chains (sub, umin), all map queries, all dimvar
  init/step values, and all loop annotation values are **u32**.
- **u32-extent guard (LO-05):** `extentOfBiv` (lower_rewrite.go:226-241)
  fails closed on a non-u32 BIV max operand — the umin bound chain requires
  u32 extents (an extent of any other type would emit ill-typed nodes that
  only the pipeline's later typeinfer re-run catches).
- The intrinsics' result type is u32 by the typeinfer rule (§3.1); the
  verifier's shape check pins the DoSoC operand's const-u32-ness.

### 10.3 Tail clamping inside loaders [spec'd]

- Element-level guarding physically lives in the thr_loop bailout
  (whole-body no-op); tile-level partial-tile clamping lives INSIDE the
  cooperative loaders (both ruled per §4). The compiler emits no clamp of
  its own beyond the F1 bound materialization.
- Runtime implementation notes: the cooperative loader must bounds-check
  its global tensor reads internally (invalid tile slots may hold
  unspecified padding; user lambdas see only valid values) — this is what
  makes a partial-tile fill correct with all threads participating.

### 10.4 Launch-shape consequences (reference numbers, fix4 model)

The portion arithmetic below is DELETED (fix4 §1.2/§2.1). Reference
numbers under the landed model: blockDim = (max over LNTs of Π warp K)×32
floored at 32 — e.g. the §3.3 worked example (K=WARP K=2) launches
blockDim 64 = 2 warps; a zero-WARP schedule launches blockDim 32 (one
warp — the zero-warp admission, pinned by the frontend fixture's
`ValidateLaunch(dim3(64u,1u,1u), dim3(32u,1u,1u), 0u, 512u)`); gridDim =
max over LNTs of Π CTA LoC, capped by MaxCTAPolicy 512. Historical text:
Defaults at MaxCTA=100 / MaxThreadsPerCTA=512: warps per portion quantum =
(512/32)/8 = 2 warps; AutoWarpPortion (LNT carries a CTA dim) = 4 portions
→ coupled walk = (8−4)×2×32 = 256 threads. CTA count rounding: 108 → 100,
A6000 84 → 80 (rounded DOWN to the PortionCTA multiple).

---

## 11. Spec'd-future runtime surface (from final_lower.md)

Objects decided by spec whose IR emission lands with final-lower /
reducer lowering / codegen. None exist as IR nodes today; listed so the
runtime library design accounts for them.

| Object | Status | Contract (spec reference) |
|--------|--------|---------------------------|
| `KernelInfo` | DELIVERED | runtime/tca/mapping.h — dual-compile (device reads gridDim.x/blockDim.x; host ctor for CPU tests); `total_cta()`/`total_thread_per_cta()` return the launch-DERIVED values (fix4 §4.1). IR: KindRuntimeKernelInfo, first kernel statement. |
| `RuntimeGridMapping` / `RuntimeThreadMapping` IR nodes | DELIVERED | internal/ir/node_runtime.go — ops = `[kernel_info, ndim const, ABSOLUTE shape consts...]`, extents ≥ 2, −1 remainder STRUCK by the final-lower verifier; the zero-extent ThreadMapping (ndim 0) admitted. Emitted as the per-LNT `tca_gm<N>`/`tca_tm<N>` ctor locals (goldens + conformance_new_api.cu). |
| `Reducer<T>` + Seq/Warp/Global specializations | DELIVERED (R1/R2, 2026-10-23) | Scalar-only protocol: ctor `(T init, T (*fn)(T, T))` for Seq and `(T init, T (*fn)(T, T), const ThreadMapping& map, SmemView<T, 1u> staging)` for Warp/Global; `Init()` argument-free per output element; `Reduce(value)` one element per call; `Combine()` returns `T` (warp: shuffle down-tree -> `is_first_in_warp()` stages at `warp_index()` -> `__syncthreads` -> replicated fold over `warp_count()` from the identity). The staging row is COMPILER-OWNED (one `#warps x sizeof(T)` dynamic-pool allocation per reducer, R1/D8); warp/lane/count identity derives from the mapping (R2) — no slot arguments, no runtime-static scratch. The combiner lambda body is NOT inlined — the reducer calls it internally. |
| `RuntimeCallMethod` node | [spec'd] | Method call on a runtime IR node: input #0 = the runtime object, #1 = method name, #2-K = parameters (final_lower.md "Runtime Object Node"). |
| `storetile` variants (StoreTileCta/StoreTileThread) | [spec'd] | Matching store-side variants; v0 loads-only, never emitted (tca.md §5.11). |
| Map `warp_index`/`lane_index`/`is_first` queries | DELIVERED (flat-index() ruling 2026-10-23) | `ThreadMapping::warp_index()`/`lane_id()`/`is_first_in_warp()`/`is_first_in_cta()`/`warp_count()` on the v0 surface (runtime/tca/mapping.h) — runtime-internal control flow for reducer staging and the R3 single-owner store gate; generated code never inlines threadIdx arithmetic. |
| Concurrent-LNT portion sharding | [spec'd] | The ONE future place chained/shard maps may return (loop_tile_lower.md §7bis). |

---

## Appendix A. Golden dump cross-checks

`internal/opt/tiling/testdata/tile_lowered_matmul.ir` (worked example at
the fix4 §3.3 signature, M=1024/N=2048/K=512; materialize+print round-trip
golden):

- Statement #163 (`memstore` of the output) carries the lowered global
  index forms: `runtimegridquery(index, 0) * 256 + @d0(init=0, step=1024)
  + @d0(init=runtimethreadquery(index), step=64)` (CTA dim M: grid rank ×
  T + round + the coupled walk — flat-rank init, blockDim step) and
  `runtimethreadquery(index, 0) * 32 + @d2(init=0, step=64) +
  @d2(init=runtimethreadquery(lane_id) * 1, step=1)` (WARP dim: warp-unit
  rank + the lane-chunk init lane·⌈T/32⌉).
- Statement #165 (`memstore[frag](%3, %56)`) is the per-thread lhs fill
  (LoadTileThread); statement #166 (`memstore[smem](%16[@d1(init=0,
  step=256)][@d2(init=0, step=64)], %59)`) is the cooperative rhs fill
  (LoadTileCta) — buffer, covered round-base bivs, `loadtile` variant.

## Appendix B. Emission-site index (Go files)

| File | What it owns |
|------|--------------|
| internal/ir/node_runtime.go | RuntimeGridQuery/RuntimeThreadQuery (flat + per-dim + lane forms), RuntimeKernelInfo, RuntimeGridMapping/RuntimeThreadMapping construction nodes — the renamed intrinsic family (fix4 T4/T5) |
| internal/ir/builtins.go | Builtin signature registry (umin/umax + f32 set) |
| internal/ir/node_lir_memory.go | MemNewSMem/MemStoreSMem/MemLoadSMem(+Chained), FragNew/FragStore/FragLoad(+Chained), StorageClass |
| internal/ir/node_loop.go | KindForLoop + the soc/dosoc/loc annotation vocabulary |
| internal/ir/node_lambda.go | NewReduce + the `init` KeyedAttribute |
| internal/ir/node_iterfmt.go | NewKeyedAttribute mechanism |
| internal/ir/kind.go / catalog.go | Kind enum (KindVersion 5), per-kind rows incl. the two intrinsic kinds |
| internal/ir/print.go | Canonical print forms (runtimegridquery(index, %N), runtimethreadquery(lane_id), keyedattribute(k=v), @x(init, step)) |
| internal/ir/verify.go | Intrinsic shape check (operand count, DoSoC constness, field vocabulary) |
| internal/opt/typeinfer/rules.go | u32 typing rule for both intrinsics; KindCall builtin signature check |
| internal/opt/tiling/record.go | fix4 §2.1 Params (WindowK, TileSizes, CtaLoCPresets/WarpLoCPresets absolute, MaxPartitionCta/Thr, MaxCTA policy cap, MaxWarpsPerCTA, MaxCache, MaxFragPerThread, CacheDims, MaxCS, ReducedDimWarp) + Validate + DefaultParams — the portion constants are RETIRED |
| internal/opt/tiling/verify.go | tiling-verify (fix4 §2.7): LoC preset membership, Π CTA LoC ≤ MaxCTA, Π warp K ≤ 32, ≥1 CTA dim, tile ≥ 2, kernel-wide budgets, judge agreement |
| internal/opt/tiling/search.go | fix4 §2.2 feasible-area enumeration: multi-dim same-category factorizations, incremental MAPM/MALoC/CuMAP pruning, SoC/LoC-desc/tile-asc determinism, WARP-on-reduced pruned while ReducedDimWarp=false |
| internal/opt/tiling/lower_rewrite.go | DoSoC numbering; dimvar/bound/rank/offset/index generation (F1 tileMax, LO-05 u32 extent guard); ForLoop construction with soc/dosoc/loc |
| internal/opt/tiling/lower_cache.go | Budget resolution (R3/F4/F6); buffer/fill/access surgery (R1 placement, R2 window sizing + rank-in-index, loadtile variant) |
| internal/opt/tiling/lower.go | Pass skeleton; §8 structural self-checks incl. checkNoBarriers (F5) |
