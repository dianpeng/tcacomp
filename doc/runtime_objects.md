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
  (worked example, M=1024/N=2048/K=512): statement #137 shows the lowered
  global index forms (`intrinsicgridmap(index, 0) * 64 + @d0(...)`,
  `intrinsicthreadmap(index, 1) * 64 + @d2(...)`), statement #139 shows the
  cooperative fill `memstoresmem(%3[@d2(init=0, step=16384)][loadtile=1], %52)`.

## Inventory

| # | Object | Category | Status | Emission site / spec reference |
|---|--------|----------|--------|--------------------------------|
| 1 | Launch totals (MaxCTA, MaxThreadsPerCTA) | Launch contract | [emitted] (compiler thresholds) / runtime assert-and-trap [spec'd] | internal/opt/tiling/record.go:258-265 (Params), :314-327 (DefaultParams: 100 / 512) |
| 2 | PORTION_CTA = 10, PORTION_WARP = 8 | Launch contract | [emitted] (compiler constants; runtime mirror spec'd) | internal/opt/tiling/record.go:237-252 |
| 3 | Runtime assert-and-trap launch pre-conditions | Launch contract | [spec'd] | doc/optana/loop_tiling.md "Runtime hard pre-condition, ASSERT-AND-TRAP"; compiler mirror record.go:338-344 |
| 4 | Max.cta / Max.threads_per_cta feasibility thresholds | Launch contract | [emitted] | record.go:197 (tiling-verify CTA LoC == MaxCTA), :201-208 (warp portion multiples) |
| 5 | 1-D normalized launch (gridDim.x / blockDim.x) | Launch contract | [spec'd] | doc/optana/loop_tile_lower.md §7bis "1-D normalized launch" |
| 6 | Per-LNT remap / per-LNT exhaustion | Launch contract | [spec'd] | loop_tile_lower.md §7bis "Per-LNT scoped maps" |
| 7 | GridConcurrencyMap (a.k.a. GridMapping) | Maps | [spec'd] (object) — queries [emitted] | final_lower.md "Per LNT's Mapping Object Construction"; queries: NewIntrinsicGridMap |
| 8 | ThreadMap | Maps | [spec'd] (object) — queries [emitted] | final_lower.md (same); queries: NewIntrinsicThreadMap |
| 9 | `intrinsicgridmap(field, dosoc)` query node | Maps / Intrinsics | [emitted] | internal/ir/node_intrinsic.go:39-44; lower_rewrite.go (5 call sites) |
| 10 | `intrinsicthreadmap(field, dosoc)` query node | Maps / Intrinsics | [emitted] | internal/ir/node_intrinsic.go:49-54; lower_rewrite.go; lower_cache.go:461 |
| 11 | `warp_index` / `lane_index` / `is_first_thread` queries | Maps | [spec'd] | loop_tile_lower.md §7bis "Tuple coordinates" |
| 12 | `umin` builtin | u32 builtins | [emitted] | internal/ir/builtins.go:31; materialized at lower_rewrite.go:284-288 |
| 13 | `umax` builtin | u32 builtins | [emitted] | internal/ir/builtins.go:31 (registry; no current materialization site) |
| 14 | ceildiv identity (A + T − 1) / T | u32 builtins | [spec'd] (recipe pinned; not materialized — 0-based between loops eliminate ceil) | loop_tile_lower.md §3 "Recipe" |
| 15 | `thr_loop` materialization | Loop primitives | [spec'd] (soc=2 annotation [emitted]) | loop_tile_lower.md §4; annotation lower_rewrite.go |
| 16 | `cta_loop` materialization | Loop primitives | [spec'd] (soc=1 annotation [emitted]) | loop_tile_lower.md §4 (amended 2026-10-21: guarded) |
| 17 | `seq_loop` materialization | Loop primitives | [spec'd] (soc=0 annotation [emitted]) | loop_tile_lower.md §4 |
| 18 | Whole-body no-op on tile_max == 0 | Loop primitives | [spec'd] | loop_tile_lower.md §4 |
| 19 | MemNewSMem (SMEM buffer) | Memory & tile | [emitted] | internal/ir/node_lir_memory.go:158-163; allocated lower_cache.go:349 |
| 20 | FragNew (FRAG buffer) | Memory & tile | [emitted] | node_lir_memory.go:244-249; allocated lower_cache.go:347 |
| 21 | MemStoreSMem fill (cooperative) | Memory & tile | [emitted] | node_lir_memory.go:140-148; emitted lower_cache.go:446 |
| 22 | FragStore fill (per-thread) | Memory & tile | [emitted] | node_lir_memory.go:226-234; emitted lower_cache.go:444 |
| 23 | LoadTileCta runtime loader (variant 1) | Memory & tile | [spec'd] (variant tag [emitted]) | lower_cache.go:79-82, :441; doc/tca.md §5.11 |
| 24 | LoadTileThread runtime loader (variant 2) | Memory & tile | [spec'd] (variant tag [emitted]) | lower_cache.go:79-82, :441; doc/tca.md §5.11 |
| 25 | MemLoadSMemChained (buffer access) | Memory & tile | [emitted] | node_lir_memory.go:121-128; lower_cache.go:474 |
| 26 | FragLoadChained (buffer access) | Memory & tile | [emitted] | node_lir_memory.go:207-214; lower_cache.go:472 |
| 27 | SMEM window sizing (#threads_d × T_d / T_d) | Memory & tile | [emitted] | lower_cache.go:331-344 |
| 28 | MaxCache = 64 KB (kernel-wide SMEM budget) | Memory & tile | [emitted] | record.go:271-277, :314-327; gated lower_cache.go:159-207 |
| 29 | MaxFragPerThread = 256 B (F6) | Memory & tile | [emitted] | record.go:278-286, :314-327 |
| 30 | Barrier ownership + placement rules | Barriers | [spec'd] | loop_tile_lower.md §6 "Sync", §4 layering; tca.md §5.11 |
| 31 | Mask contract (thread_map single supplier, lane-prefix closed form, full-warp fast path) | Masks | [spec'd] | loop_tile_lower.md §4 "Mask contract" |
| 32 | `soc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go (all ForLoop construction sites) |
| 33 | `dosoc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go; numbering: lower_rewrite.go:166-196 |
| 34 | `loc` KeyedAttribute | Annotations | [emitted] | lower_rewrite.go |
| 35 | `loadtile` KeyedAttribute | Annotations | [emitted] | lower_cache.go:441; consumed by checkNoBarriers lower.go:465-492 |
| 36 | `init` KeyedAttribute (Reduce) | Annotations | [emitted] | internal/ir/node_lambda.go:106 (NewReduce) |
| 37 | Saturating tile_max contract (F1) | Numerical / index | [emitted] | lower_rewrite.go:271-288 |
| 38 | u32 arithmetic domain + u32-extent guard (LO-05) | Numerical / index | [emitted] | lower_rewrite.go:216-241; typeinfer u32 rule internal/opt/typeinfer/rules.go |
| 39 | Tail clamping inside loaders | Numerical / index | [spec'd] | loop_tile_lower.md §4 layering, §7bis "Bounds checking" |
| 40 | KernelInfo runtime object | Spec'd future | [spec'd] | final_lower.md "Initial Runtime Object Construction" |
| 41 | RuntimeGridMapping / RuntimeThreadMapping IR nodes | Spec'd future | [spec'd] | final_lower.md "Per LNT's Mapping Object Construction" |
| 42 | Reducer<T> + Seq/Warp/Global specializations | Delivered (R1/R2/R3 ruling 2026-10-23) | `runtime/tca/reducer.h`; IR kinds `runtimesequentialreducer`/`runtimewarpreducer`/`runtimeglobalreducer` | final_lower.md "Reducer Lower" as amended; doc/rt.md §4.6; doc/tca.md §5.10 |
| 43 | RuntimeCallMethod node | Spec'd future | [spec'd] | final_lower.md "Runtime Object Node" |
| 44 | storetile / StoreTileCta / StoreTileThread variants | Spec'd future | [spec'd] (v0 explicitly out of scope: loads-only) | doc/tca.md §5.11; loop_tile_lower.md §6 "Loads-only scope" |

Counts: 44 inventory entries — [emitted] 20 (rows 1, 2, 4, 9, 10, 12, 13, 19-22,
25-29, 32-38, of which rows 1/2 carry a spec'd runtime mirror), [spec'd] 24.

---

## 1. Launch contract

### 1.1 Launch totals — MaxCTA, MaxThreadsPerCTA [emitted thresholds; runtime assert-and-trap spec'd]

- **Name:** launch totals (gridDim.x = MaxCTA, blockDim.x = MaxThreadsPerCTA).
- **Emission site:** `internal/opt/tiling/record.go:258-265` (Params fields),
  `:314-327` (DefaultParams). Defaults: **100 CTAs** (Ampere's 108 SMs rounded
  DOWN to the PortionCTA multiple) / **512 threads per CTA = 16 warps**.
  Target override example: A6000 84 SMs → 80 CTAs.
- **Signature & operands:** not an IR node — compile-time static contract.
  Every LNT's schedule is computed under these totals; the lowered loops
  encode LoC-derived strides as dimvar step constants (see the golden:
  `@d0(init=0, step=6400)` etc.).
- **Semantics:** the launch config is EXTERNAL and static at compile time
  (from the target GPU and green-context SM slicing). It is a RUNTIME
  CONTRACT — the normalized 1-D launch is how the per-LNT map objects learn
  the totals they remap. It is NOT a compiler-side launch decision.
- **Runtime implementation notes:** the runtime receives the totals (CUDA
  launch config) and must (a) ASSERT-AND-TRAP the pre-conditions below,
  (b) expose the totals to the map constructors, (c) query device attributes
  for the SMEM ceiling and assert ≥ the design budget (64 KB) — NEVER
  hardcode sm_86 ceilings (101,376 B).
- **Spec reference:** loop_tiling.md "LoC concurrency bounds, the FINAL
  MODEL"; loop_tile_lower.md §7bis "Launch/budget validation at runtime".

### 1.2 Portion constants — PORTION_CTA=10, PORTION_WARP=8 [emitted; runtime mirror spec'd]

- **Name:** PORTION_CTA, PORTION_WARP.
- **Emission site:** `internal/opt/tiling/record.go:237-252`. Named portion
  constants — future adaptation points.
- **Semantics:** the grammar the launch totals are divided into. CTA-assigned
  dims' portions SUM to 10; warp-assigned dims' portions (CTA-coupled halves
  INCLUDED) SUM to 8. The LAST assignment in each category may be −1 =
  derived remainder. PORTION_WARP = 8 because at the minimum config each
  portion is exactly ONE warp (8 warps / 256 threads minimum).
- **Runtime implementation notes:** mirror these constants; the
  assert-and-trap and the map decomposition both consume them.
- **Spec reference:** loop_tiling.md "Named portion constants".

### 1.3 Runtime assert-and-trap pre-conditions [spec'd]

- **Name:** launch divisibility assert-and-trap.
- **Semantics:** provided # of CTAs MUST be a multiple of PORTION_CTA (10);
  threads per CTA MUST be a multiple of PORTION_WARP warps (8 warps) with
  **256 threads minimum**.
- **Compiler-side mirror (today):** `Params.Validate`, record.go:338-344 —
  `MaxCTA` a positive multiple of 10; `MaxThreadsPerCTA` a multiple of 32
  whose warp count is a multiple of 8.
- **Runtime implementation notes:** trap (not silently clamp) on violation;
  the map arithmetic divides by these grammars unconditionally.
- **Spec reference:** loop_tiling.md "Runtime hard pre-condition,
  ASSERT-AND-TRAP".

### 1.4 Feasibility thresholds — Max.cta / Max.threads_per_cta [emitted]

- **Emission site:** tiling-verify, `internal/opt/tiling/verify.go:171-234`:
  the single CTA dim must exhaust MaxCTA exactly (LoC == MaxCTA, portion
  −1/full); warp LoC must be a warp-portion multiple (quantum = warps per
  portion = (MaxThreadsPerCTA/32)/8; portion ∈ [1, 8]).
- **Semantics:** LoC of a dim = portion × quantum (CTA quantum = total
  CTAs / 10; warp quantum = total warps / 8). Every LNT in the kernel is
  optimized under the SAME bounds; in v0 LNTs execute sequentially and each
  remaps the FULL launch (per-LNT exhaustion).
- **Spec reference:** loop_tiling.md "LoC concurrency bounds"; "Portion
  grammar".

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
ONE **GridMapping** over the CTA space, ONE **ThreadMap** over the per-CTA
thread space. Each is MULTI-DIMENSIONAL and holds the COMPLETE partition
knowledge of its space (all CTA dims / all warp-category consumers). The
earlier per-dim chained map description was a design typo — superseded.

### 2.1 GridConcurrencyMap / GridMapping [spec'd object; queries emitted]

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

### 2.2 ThreadMap [spec'd object; queries emitted]

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
| `gridmap(d,"index")·T + round` | CTA-dim round offset (in-body local) | lower_rewrite.go:357-359 |
| `threadmap(d,"index")·T + round` | WARP-dim chunk offset (in-body local) | lower_rewrite.go:386-388 |
| `gridmap(d,"index")·T + round + threadmap(d′,"index") + i·stride` | CTA dim's coupled inside walk (i at element stride) | lower_rewrite.go:371-376 |
| `threadmap(d,"index")·T_d + i` | R2 SMEM buffer access index on WARP dim d | lower_cache.go:458-467 |

No map query appears in any induction variable; ranks enter ONLY as in-body
offset locals.

---

## 3. Intrinsics

Two LIR node kinds, materialized by the loop-tile-lower pass
(quick-261021-jay), exist only between tile-lower and codegen.

### 3.1 KindIntrinsicGridMap / KindIntrinsicThreadMap [emitted]

- **Emission site:** node struct `internal/ir/node_intrinsic.go:29-33`;
  constructors :39-54; catalog rows `internal/ir/catalog.go:68-69`;
  print form `internal/ir/print.go:749-764` (`intrinsicgridmap(index, %N)` /
  `intrinsicthreadmap(index, %N)`); verify shape case (operand count, DoSoC
  constness, field vocabulary — internal/ir/verify.go); typeinfer rule
  (u32 — internal/opt/typeinfer/rules.go:194-197); legality-window row
  (legal at StageConstructed by the catalog-iteration rule);
  contentHash/contentEqual (mixString of DoSoC + field, internal/ir/value.go).
  These are the spec's "six companions" (loop_tile_lower.md §5): (1) node
  struct + catalog row, (2) allocNode case, (3) verify shape case,
  (4) typeinfer rules, (5) legality-window row, (6) contentHash/contentEqual.
- **Signature & operands:** `ops = [DoSoC const u32]`; field payload
  `"index" | "dimension"`; result u32; FamilyAbstract (pure value —
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
  `for (d = 0; d < extent; d += LoC·T)`; trip count uniform for every
  CTA/thread of the allocation. NO map index in ANY induction variable;
  ranks appear ONLY as in-body offset locals
  (`offset = rank·T + round_base` — golden: `@d0(init=0, step=6400)` is the
  round dimvar, `intrinsicgridmap(index, 0) * 64` the rank term).
- **Induction identity:** `offset = rank·T + round_base` for CTA/WARP dims;
  `index = offset + inside` (WARP dims); `index = offset + inside + thread_rank`
  for the CTA dim's coupled walk (element-stride inside at coupled-thread
  stride).
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
| CTA | between | 1 | gridDim | LoC |
| CTA | inside (coupled warp walk) | 2 | threadDim | coupled threads |
| WARP | between | 2 | threadDim | LoC×32 |
| WARP | inside (owned-chunk walk) | 0 | — | 1 |

(All dims are tiled, T >= 2 — the 2026-10-24 tile-disable ruling; the
untiled single-loop forms are gone.)

- **Coupled warp walk (CTA dim's inside):** element stride, coupled
  threads = `(PORTION_WARP − warpPortion) × quantum × 32`, where quantum =
  (MaxThreadsPerCTA/32)/PORTION_WARP (lower_rewrite.go:248-259). F2 floor:
  ≥ 1 portion must remain for the coupled walk (a zero-portion coupled walk
  is a zero-stride inside loop = infinite loop); the rewrite fails closed
  (lower_rewrite.go:362-369) and tiling-verify rejects the record first.
- **WARP composition semantics:** each thread owns a WHOLE T-chunk of the dim
  (between stride = nThreads·T); the inside loop is SEQUENTIAL over the owned
  chunk. Distinct from the CTA dim's coupled walk (element granularity).
- **Spec reference:** loop_tile_lower.md §3, §6bis PHASE 3; loop_tiling.md
  "Mechanical rewrite for tiling", "Composition Semantics, CTA x WARP".

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
- **Semantics (window sizing, R2):** per covered dim — WARP dim d →
  `#threads_d × T_d` elements (the whole group's round window, NOT the
  single-thread chunk); SEQ dim → `T_d`. Total buffer bytes = Π over
  covered dims × element size. Exactly the decide side's window-bytes model
  (one budget, both halves). Buffer dim names ride the tensor's dim names.
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
- **Semantics:** per-thread register fragment; extent = same window-bytes
  product over covered dims (SEQ dims contribute T_d; a WARP dim would
  contribute its window, but FRAG sites are all-SEQ by the tier rule, so in
  practice Π T_d × elem bytes — worked example: lhs 32 × 4 B = **128 B per
  thread**).
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
  dim in tensor order — WARP dim: `threadmap(d,"index")·T_d + i` (R2
  rank-in-index; d = the dim's slot in the LNT's ThreadMap — NOT a
  hardcoded 0, NOT the bare inside coordinate); SEQ dim: its plain inside
  coordinate (golden-access form: `threadmap(index, 1)·64 + @d2(init=0,
  step=1)`). FRAG buffers: plain per-thread `i`.
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

### 10.4 Launch-shape consequences (reference numbers)

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
| `KernelInfo` | [spec'd] | One runtime object at kernel entry holding all cuda kernel info: `total_cta()`, `total_thread_per_cta()`. Constructed at the very first block (final_lower.md "Initial Runtime Object Construction"). |
| `RuntimeGridMapping` / `RuntimeThreadMapping` IR nodes | [spec'd] | Per-LNT mapping-object construction PREPENDED to each Loop BB Cluster's starting block: `RuntimeGridMapping(kernel_info, ndim, portion..., -1)` / `RuntimeThreadMapping(...)` — the compiler-side constructors of the §2 map objects (final_lower.md "Per LNT's Mapping Object Construction"). |
| `Reducer<T>` + Seq/Warp/Global specializations | DELIVERED (R1/R2, 2026-10-23) | Scalar-only protocol: ctor `(T init, T (*fn)(T, T))` for Seq and `(T init, T (*fn)(T, T), const ThreadMapping& map, SmemView<T, 1u> staging)` for Warp/Global; `Init()` argument-free per output element; `Reduce(value)` one element per call; `Combine()` returns `T` (warp: shuffle down-tree -> `is_first_in_warp()` stages at `warp_index()` -> `__syncthreads` -> replicated fold over `warp_count()` from the identity). The staging row is COMPILER-OWNED (one `#warps x sizeof(T)` dynamic-pool allocation per reducer, R1/D8); warp/lane/count identity derives from the mapping (R2) — no slot arguments, no runtime-static scratch. The combiner lambda body is NOT inlined — the reducer calls it internally. |
| `RuntimeCallMethod` node | [spec'd] | Method call on a runtime IR node: input #0 = the runtime object, #1 = method name, #2-K = parameters (final_lower.md "Runtime Object Node"). |
| `storetile` variants (StoreTileCta/StoreTileThread) | [spec'd] | Matching store-side variants; v0 loads-only, never emitted (tca.md §5.11). |
| Map `warp_index`/`lane_index`/`is_first` queries | DELIVERED (flat-index() ruling 2026-10-23) | `ThreadMapping::warp_index()`/`lane_id()`/`is_first_in_warp()`/`is_first_in_cta()`/`warp_count()` on the v0 surface (runtime/tca/mapping.h) — runtime-internal control flow for reducer staging and the R3 single-owner store gate; generated code never inlines threadIdx arithmetic. |
| Concurrent-LNT portion sharding | [spec'd] | The ONE future place chained/shard maps may return (loop_tile_lower.md §7bis). |

---

## Appendix A. Golden dump cross-checks

`internal/opt/tiling/testdata/tile_lowered_matmul.ir` (worked example,
M=1024/N=2048/K=512 at defaults; materialize+print round-trip golden):

- Statement #137 (`memstore` of the output) carries the lowered global
  index forms: `intrinsicgridmap(index, 0) * 64 + @d0(init=0, step=6400) +
  @d0(init=0, step=256)` (CTA dim M: grid rank × T + round + coupled walk)
  and `intrinsicthreadmap(index, 1) * 64 + @d2(init=0, step=16384) +
  @d2(init=0, step=1)` (WARP dim access + R2 rank-in-index form).
- Statement #139 (`memstoresmem(%3[@d2(init=0, step=16384)][loadtile=1],
  %52)`) is the cooperative fill: buffer `%3`, round-base biv operand
  (`@d2(init=0, step=16384)` — the between dimvar), and the `loadtile=1`
  variant annotation (LoadTileCta).

## Appendix B. Emission-site index (Go files)

| File | What it owns |
|------|--------------|
| internal/ir/node_intrinsic.go | IntrinsicGridMap/IntrinsicThreadMap node structs + constructors + field read face |
| internal/ir/builtins.go | Builtin signature registry (umin/umax + f32 set) |
| internal/ir/node_lir_memory.go | MemNewSMem/MemStoreSMem/MemLoadSMem(+Chained), FragNew/FragStore/FragLoad(+Chained), StorageClass |
| internal/ir/node_loop.go | KindForLoop + the soc/dosoc/loc annotation vocabulary |
| internal/ir/node_lambda.go | NewReduce + the `init` KeyedAttribute |
| internal/ir/node_iterfmt.go | NewKeyedAttribute mechanism |
| internal/ir/kind.go / catalog.go | Kind enum (KindVersion 5), per-kind rows incl. the two intrinsic kinds |
| internal/ir/print.go | Canonical print forms (intrinsicgridmap(index, %N), keyedattribute(k=v), @x(init, step)) |
| internal/ir/verify.go | Intrinsic shape check (operand count, DoSoC constness, field vocabulary) |
| internal/opt/typeinfer/rules.go | u32 typing rule for both intrinsics; KindCall builtin signature check |
| internal/opt/tiling/record.go | PortionCTA/PortionWarp/AutoWarpPortion constants; Params (MaxCTA, MaxThreadsPerCTA, WarpPortion, MaxCache, MaxFragPerThread, ...) + Validate + DefaultParams |
| internal/opt/tiling/verify.go | tiling-verify: CTA LoC == MaxCTA, warp portion multiples, F2 coupled floor, kernel-wide budget |
| internal/opt/tiling/search.go | Searcher policy: ≤1 CTA dim + ≤1 independent WARP dim per LNT; coupled floor; per-thread reuse gate |
| internal/opt/tiling/lower_rewrite.go | DoSoC numbering; dimvar/bound/rank/offset/index generation (F1 tileMax, LO-05 u32 extent guard); ForLoop construction with soc/dosoc/loc |
| internal/opt/tiling/lower_cache.go | Budget resolution (R3/F4/F6); buffer/fill/access surgery (R1 placement, R2 window sizing + rank-in-index, loadtile variant) |
| internal/opt/tiling/lower.go | Pass skeleton; §8 structural self-checks incl. checkNoBarriers (F5) |
