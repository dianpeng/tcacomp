# fix4 — The factorization scheduling refactor (implementation spec)

- Status: READY FOR IMPLEMENTATION. This doc consolidates `doc/fix/fix-opt-tile-search.md` (final) and every ruling from `doc/review/fix-opt-tile-search.md` rounds 1–8 into ONE self-contained spec covering the full cascade: decide pass, tile-lower, runtime maps/iterators/launch, codegen, final-lower.
- **Amendment 2026-10-26 (post-T1/T2 verification)**: §2.4 reuse re-defined to ONE CTA-wide currency — collective per-window reuse for SMEM sites, holders-scaled per-thread reuse for FRAG sites. The pre-amendment form multiplied per-window bytes by per-thread trips, rewarding warp-splits that shrink blockDim without moving any wire bytes (the §3.3 pin divergence was this artifact). §3.3 worked example re-pinned; §9(2) signature updated; Wave-1 task **T1.1** (§8) lands the judge.go correction. Search/eviction/verify machinery unchanged.
- **Conflict rule: where this doc disagrees with `doc/optana/loop_tile_lower.md`, `doc/rt.md`, `doc/cg.md`, or `doc/optana/final_lower.md`, THIS DOC WINS.** Those docs get pointer-stub updates (task T8) — the smem_insertion.md precedent.
- Superseded by this refactor, explicitly: the disjoint-shard warp model; the portion grammar (PORTION_CTA/PORTION_WARP, −1 remainder, AutoWarpPortion, WarpPortion); the v0 searcher policy (≤1 CTA dim + ≤1 WARP dim per LNT); `MaxThreadsPerCTA`/`MaxCTA`-as-exhaustion; the runtime assert-and-trap pre-conditions; the F2 coupled-portion floor; the CTA `LoC == MaxCTA` verify check.
- Staging: **Wave 1** = everything in this doc, with split-k excluded from the feasible area via `Params.ReducedDimWarp = false` (a CAPABILITY gate on the feasible area, NOT the rejected permanent policy). **Wave 2** = split-k enablement + its goldens — **LANDED (T8, quick-261027-kzv)**: the default flipped to `true` with the dedicated split-k matmul e2e green on the A6000 (§3.5); explicit `false` remains as the Wave-1 policy opt-out.

## §1 The factorization model

### 1.1 Assignment

Per window iteration (the M=4 innermost consecutive spine iters; outside-window and non-spine iters default SEQ):

- **SoC** ∈ {SEQ, CTA, WARP}
- **LoC** — ABSOLUTE count: CTA dim → # CTAs for this dim; WARP dim → # warps for this dim; SEQ → 1 implicit.
- **TileSize** T_d ∈ presets {32, 64, 128, 256}; tile ≥ 2 always (2026-10-24 ruling).

### 1.2 Composition = factorization (both levels)

- **CTA space**: `gridDim.x = Π (CTA LoC_d)`. CTA coordinate tuple via reverse stride-shape decomposition of `cuda_linear_id`. In dim d, CTA c owns tiles ≡ c (mod LoC_d).
- **Warp space**: warp index factors into one coordinate per WARP dim; `Π (warp K_d) ≤ 32`. In dim d, warp c owns tiles ≡ c (mod K_d).
- Every tile combination has exactly ONE owning (CTA, warp). **Exhaustive launch**: `blockDim.x = Π K × 32` exactly — launch what you factorize; no idle or co-owning warps.
- WARP dim with K = 1 ≡ SEQ — pruned from the feasible area (dominated).
- The CTA dim's coupled inside walk strides over ALL blockDim threads (§1.3). `coupledThreads := blockDim`. **F2 is deleted** (no "remaining warps" concept exists).

### 1.3 Structural table (TBL / TIL)

| SoC | TBL (between loop) | TIL (inside walk) |
|---|---|---|
| CTA | rounds stride `LoC_d × T_d` elements; rank = `grid_map.index(d)` ∈ [0, LoC_d); offset = `rank·T_d + round` | **coupled walk**: thread iterator init = flat thread rank (`thread_map.index()` no-arg), **step = blockDim**, element = `offset + i`, bound = `d_tile_max` |
| WARP | rounds stride `K_d × T_d`; rank = warp coordinate `thread_map.index(d)` ∈ [0, K_d); offset = `rank·T_d + round` | **lane chunks**: the owning warp's 32 lanes split the tile into CONSECUTIVE chunks of ⌈T_d/32⌉; lane l walks `[l·⌈T_d/32⌉, (l+1)·⌈T_d/32⌉)` of the tile, each lane's loop is sequential, trip count 0 when out of range |
| SEQ | rounds stride `T_d` (LoC = 1) | each thread walks the full T_d interior sequentially (its own coordinates) |

- Expanded SoC instances (what constraints count): CTA dim → 1 CTA instance + 1 THR instance; WARP dim → 1 THR instance; SEQ → none.
- **ThreadMap dimensionality (adopted ruling — veto point):** ThreadMap carries ONE slot per WARP dim only (n slots, coordinate in warp units; `lane_id()` for the lane). The CTA dim's coupled walk uses the NO-ARG flat thread rank and needs NO map slot. GridMapping carries one slot per CTA dim (m slots). (Rationale: the coupled walk is flat-strided — a coordinate slot buys nothing; this also simplifies the old m+n counting invariant.)

### 1.4 Mask contract (survives from loop_tile_lower.md §4)

Lane-prefix masks hold under consecutive chunks: chunk lengths are non-increasing in lane id, so the active-lane set at step j is always a prefix. Full-warp fast path: the coupled walk's blockDim stride is warp-multiple ⇒ non-tail iterations mask 0xFFFFFFFF. thread_map stays the SINGLE mask supplier; exact / uniformly-derived / per-iteration properties unchanged.

## §2 Decide pass — `internal/opt/tiling`

### 2.1 Params (the hyperparameter surface)

| Field | Default | Meaning |
|---|---|---|
| `WindowK` | **4** (was 5) | innermost window depth |
| `TileSizes` | {32, 64, 128, 256} | per-dim tile presets |
| `CtaLoCPresets` | {2, 4, 8, 16, 32, 64} | absolute CTA counts per CTA dim |
| `WarpLoCPresets` | {2, 4, 8, 16, 32} | absolute warp counts per WARP dim |
| `MaxPartitionCta` / `MaxPartitionThr` | 3 / 3 | MAPM dim caps (expanded instances) |
| `MaxCTA` | 512 | gridDim policy cap (MALoC CTA bound) |
| `MaxWarpsPerCTA` | 32 | hardware bound (MALoC warp bound) |
| `MaxCache` | 65536 | SMEM window budget, kernel-wide |
| `MaxFragPerThread` | 256 | per-thread FRAG budget, kernel-wide |
| `CacheDims` | 2 | trailing cached dims |
| `MaxCS` | 16 | candidate-set cap |
| `ReducedDimWarp` | true (Wave 2 — T8 landed; explicit false = the Wave-1 policy) | allow WARP-SoC on reduced dims (split-k) — §3.5 |

Retired: `PortionCTA`, `PortionWarp`, `AutoWarpPortion`, `WarpPortion`, `MaxThreadsPerCTA`, old `MaxCTA`-as-exhaustion. `Validate` fail-fasts on the new fields.

### 2.2 Feasible area & enumeration

- Per window dim: SoC (3) × LoC preset × TileSize (4) — 12 SoC-LoC combos + SEQ ⇒ 48/dim; `48^4 ≈ 5.3M` upper bound per LNT.
- **Pruning is mandatory**: reject a PARTIAL vector the moment MAPM/MALoC/CuMAP is violated (incremental, never enumerate-then-filter); skip WARP dims with K=1; skip WARP on reduced dims when `ReducedDimWarp=false` (the reducer-consuming iters are visible via the analysis used by final-lower's anchor gate; if cheap detection is not available in the decide pass's position, gate on the pass's existing reachability of reducer consumers).
- **Enumeration order (determinism contract)**: outermost window dim = most significant digit; digits ordered by SoC (SEQ, CTA, WARP), then **LoC DESCENDING**, then tile ASCENDING.
- **Selection**: strictly-greater on the compare key `(traffic, partitionCount)` replaces the best; exact ties keep the FIRST-enumerated candidate. With LoC-descending enumeration this resolves full ties toward LARGER LoC vectors.
  - **Why the LoC order matters (consolidation finding, round 8): traffic is NEUTRAL on LoC magnitudes** — total fills = (extent/T) per dim regardless of how many CTAs/warps share the rounds, and cross-warp sharing multiplies reuse by warp-sharing counts, not magnitudes. Partition count is also magnitude-blind. Without a magnitude tiebreak, full ties resolve to the smallest preset (2 CTAs) — broken utilization. Descending enumeration gives "use the partitioned space fully" as the tie-only preference, consistent with the GoC philosophy and never overriding traffic.

### 2.3 Constraints (checked over expanded instances + factorizations)

1. **MAPM**: #CTA dims ≤ 3; #THR instances (WARP dims + CTA dims) ≤ 3.
2. **MALoC**: per LNT, `Π CTA LoC ≤ MaxCTA` and `Π warp K ≤ MaxWarpsPerCTA`.
3. **CuMAP**: ≥ 1 CTA-assigned dim. Violation = structural infeasibility = hard error.
4. **MASRAM** (two gates, kernel-wide running floors threaded through LNTs in `Roots()` order):
   - Σ SMEM window bytes ≤ MaxCache.
   - Σ per-thread FRAG bytes ≤ MaxFragPerThread.

### 2.4 Cost model (site judgments)

- **Discovery**: unchanged — `findINS`/`analyzeOverlap`/`newSiteFacts` (set cardinality ≥ 2; tie = innermost non-referenced iter; covered = trailing CacheDims dims; rank ≥ CacheDims; MaxCS cap; cross-LNT INS ⇒ explicit non-candidate, R4).
- **Tiers** (in order):
  1. Any covered dim CTA-assigned → **REFUSED** (ReasonCTACovered; kept this refactor).
  2. **C1b**: any non-covered referenced iter (INS) CTA- or WARP-assigned → **forced FRAG** (thread-varying source coordinate at the fill point).
  3. Any covered dim WARP-assigned → **SMEM window** class.
  4. All covered dims SEQ → **FRAG** class (single thread's registers, walked wholly).
- **Bytes**:
  - FRAG per thread = `Π(covered T_d) × sizeof(dtype)` — NO lane division (the thread walks all covered dims wholly). Min site = T₁·T₂·sizeof(dtype).
  - SMEM window buffer = `Π over covered dims of (factor_d × T_d) × sizeof(dtype)`, where a WARP-covered dim contributes `factor_d = K_d` (one window per warp coordinate) and a SEQ-covered dim contributes 1. **This is 32× smaller than the old shard-model `#threads_d × T_d`** — rhs-class sites now fit.
- **Reuse (2026-10-26 amendment — one-currency accounting)** = re-reads of a filled window before its next fill, counted at the window's OWN granularity so that `bytes × (reuse − 1)` is CTA-wide GMEM bytes avoided for EVERY site class. The pre-amendment form mixed per-window bytes with per-thread trips and systematically rewarded shrinking blockDim (each thread trips more, fewer threads read the same window — collective reads per fill are invariant); that phantom term is gone.
  - **SMEM class — collective, per window instance**: reuse = Π over the re-traversal dims (tie + iters strictly inner than the tie, minus covered) of the COLLECTIVE trips — CTA dim → `T_d` (⌈T/blockDim⌉ per-thread trips × blockDim readers = the tile covered exactly once; the blockDim factor cancels); WARP dim → `T_d` (the lanes' consecutive chunks sum to the tile); SEQ dim → `T_d × readers` (readers = 32 for warp-sliced windows, blockDim for CTA-shared) — **× the cross-warp sharing factor** (unchanged, EXACT: `sharing = Π over WARP-assigned dims that are NOT in the load's subscript set of K_d`; warps differing only on such coordinates read the SAME window instance — factorization makes windows genuinely shared).
  - **FRAG class — per-thread, scaled to the same currency**: reuse = Π over the re-traversal dims of the per-thread trips (CTA → ⌈T_d/blockDim⌉, WARP → ⌈T_d/32⌉, SEQ → T_d); the site's traffic avoided is multiplied by **holders** (threads that fill a window per epoch, ≈ the LNT's blockDim) so FRAG and SMEM sites compare in one currency. `sharing = 1` (registers are private). The shard model's intra-thread-only identity no longer holds.
- **Promotion gate**: traffic avoided (`bytes × (reuse − 1)`, all sites in the one CTA-wide currency — equivalently reads-served minus elements-refilled per fill epoch) > 0; FRAG site per-thread bytes ≤ MaxFragPerThread. Fill cost is therefore visible: a C1b window refilled every element-trip at reuse 2 spends half its reads on fills, and the gate and the eviction density both see it.

### 2.5 Objective

- **Primary (strict): maximize traffic avoided** = Σ over promoted sites of `bytes × (reuse − 1)`, computed on the POST-EVICTION set, every site in the one CTA-wide currency of the §2.4 reuse amendment.
- **Tiebreak 1: partition count** = (#CTA dims + #THR instances) over expanded instances (≤ 6). Never overrides a traffic difference.
- **Tiebreak 2 (implicit)**: enumeration order ⇒ larger LoC (§2.2).
- **Evolution note (author ruling, round 6)**: the parallelism bias of the tiebreak (e.g. split-k on reduced dims) is ACCEPTED; no special-case policy. The searcher is formula-agnostic — future factors (reduction-staging cost, shape-aware utilization via `NominalExtent`) drop into the score with zero algorithm change.

### 2.6 Eviction & error contract

- Both MASRAM gates run per candidate against the running floors. On violation: **greedy spill (Chaitin–Briggs-style)** — repeatedly drop the site with the least value density, `density = value/bytes = reuse − 1` (ties: smaller bytes first), until both gates pass. Evicted sites demote to GMEM; loops stay tiled.
- **Zero-cache outcome** (all sites evicted/declined): ALLOWED, emit a warning.
- **Structural infeasibility** (no CTA dim possible, no legal factorization, tile < 2): hard error. This consciously extends the 2026-10-24 "tiling never disabled" ruling (loops always tiled ≥ 2; caches may be zero).

### 2.7 Record + tiling-verify

- Record (unchanged shape, new semantics): `Iters: map[BIV]{SoC, LoC, TileSize}` with LoC ABSOLUTE (CTA → CTAs, WARP → warps, SEQ → 1); `Sites: map[MemLoad id]{Promoted, Tie, Class, Reuse, Bytes, Reason}` (`Reuse` is collective per window instance for SMEM sites, per-thread for FRAG sites — §2.4 amendment); `Verified` stamp (F7).
- tiling-verify re-checks become: LoC ∈ the SoC's preset list; per LNT `Π CTA LoC ≤ MaxCTA` and `Π warp K ≤ 32`; ≥ 1 CTA dim; TileSize ≥ 2; the two kernel-wide budget sums; tier-class agreement with a re-derived judge. DELETED: `CTA LoC == MaxCTA`, warp-portion multiples, the F2 floor, the v0-policy check.

## §3 Tile-lower — rewrite forms

### 3.1 Loop templates (replacing loop_tile_lower.md §6bis PHASE 3)

Per LNT, per dim in original order; ALL between loops nest in dim order, ALL inside loops nest inside them in dim order (unchanged layout):

- **SEQ dim d**: between `for (a = 0; a < A; a += T_d)`; inside sequential, bounded by `d_tile_max`.
- **CTA dim d**: between `for (a = 0; a < A; a += LoC_d × T_d)`; in-body `cta_rank = IntrinsicGridMap(DoSoC_d, "index")`; `a_offset = cta_rank·T_d + a`; inside = the COUPLED walk: `for (i = flat_index(); i < d_tile_max; i += blockDim)` with element `a_offset + i` (flat = `IntrinsicThreadMap` no-arg; the 2026-10-23 flat-index form survives; only the STEP changes from "remaining warp portions × quantum × 32" to **blockDim**).
- **WARP dim d**: between `for (b = 0; b < B; b += K_d × T_d)`; in-body `warp_rank = IntrinsicThreadMap(DoSoC_d, "index")` (warp units); `b_offset = warp_rank·T_d + b`; inside = LANE CHUNKS: `lane = lane_id()`, `c = ⌈T_d/32⌉`, `for (i = lane·c; i < min((lane+1)·c, b_tile_max); i++)` — each lane's sequential chunk walk, trip 0 out of range.
- **Saturating bounds (F1, unchanged)**: `off′ = umin(offset, extent); tile_max = umin(T, extent − off′)` before the inside loops.
- **Multi-dim**: the CTA form nests once per CTA-assigned dim (each with its own LoC_d stride and GridMap slot); the WARP form once per WARP-assigned dim (each with its own K_d stride and ThreadMap slot). Nothing in the template assumes a single dim.

### 3.2 DoSoC & map binding

- Two counters per LNT (outer→inner): CTA dims → GridMap slots; WARP dims → ThreadMap slots; SEQ no slot (unchanged §2 of loop_tile_lower.md; counters now go past 1).
- Map construction per LNT: `GridMapping(kernel_info, <LoC_1 …LoC_m>)`, `ThreadMapping(kernel_info, <K_1 …K_n>)` — shape args are the dim extents directly (absolute counts; NO portions, NO −1).
- Per-LNT scoping/remap unchanged (§7bis); surplus CTAs/warps under the max-over-LNT launch whole-body-skip via the existing §4 guards — ALL parallel loops stay guarded, `cta_loop` included.

### 3.3 Cache insertion

- Tier classification identical to §2.4 (the lower re-derives and must agree with the record, fail-closed — unchanged).
- **Window sizing**: buffer extents per §2.4 bytes (WARP-covered dim → K_d × T_d; SEQ → T_d). Access index on a WARP dim d = `thread_map.index(d)·T_d + i` (slot d in the LNT's ThreadMap); SEQ dim = its inside coordinate (R2 rank-in-index survives; only the factor changes).
- **C1b forced-FRAG sizing**: a WARP-covered dim under forced FRAG prices/sizes at `⌈T_d/32⌉` per lane (the lane's chunk), fill coordinate rides the AXES path as `warp_rank·T_d + round` — updated from the old per-thread T_d.
- **Fill placement (R1, unchanged)**: LoadTileThread fills INSIDE the element loop defining its per-thread coordinate; LoadTileCta fills at the innermost covered dim's between-round body with CTA-uniform round bases only. One structural splice; the §4 layering rule (cooperative ops outside bailing element loops) unchanged.
- Budget resolution (halve-then-drop) becomes an **unreachable fail-closed backstop** — the record is feasible by construction (decide-side eviction); keep the code as defense, mark unreachable in comments.
- **Worked example (2026-10-26 re-pin under the one-currency reuse amendment), matmul C[4096, 2048] += A[4096, 512]·B[2048, 512], fp32**: M=CTA LoC=16 T=256 (the R-LOCCAP cap: ceil(4096/256) = 16 — 16 tiles, 16 CTAs all busy), N=SEQ T=256, K=WARP K=2 T=32; gridDim=16, blockDim=2×32=64; rhs SMEM window = 2·32·256·4 B = 64 KB (at cap) → PROMOTED, collective reuse = T_M = 256; lhs FRAG = ⌈32/32⌉·4 B = 4 B → PROMOTED, per-thread reuse = T_N = 256. Why this beats the pre-amendment example vector (N=WARP K=4 T=64, K=SEQ T=64): the rhs sites of BOTH vectors score EXACTLY equal — 64 KB windows, collective reuse T_M = 256, sharing 1 (N and K are both in rhs's subscript set) — and that equality is the amendment's unit-consistency invariant, pinned as a regression guard; the lhs then decides honestly: the winner's 1-element windows refill ~64× fewer bytes per fill epoch (4 B × 4 fills/thread vs 256 B × 2), a genuine fill-cost difference invisible to the pre-amendment model. (Pre-amendment history: the original N=WARP/K=SEQ example was hand-computed with per-thread trips against per-window bytes; the 3.5× "traffic win" it implied for warp-split vectors was double-counting, not wire bytes. The old flagship's rhs-declined outcome remains gone — the K_d×T_d window is 32× smaller.)

### 3.4 What survives verbatim from loop_tile_lower.md

§1 record interface (F7 gating), §2 DoSoC walk, §4 loop materialization/guards/mask contract/reducer placement, §5 intrinsic node companions (six-place rule), §6 mechanics (MutContext choke points, RAUW, loads-only scope, runtime-owns-barriers), §6bis PHASE 0/1/2/4/5 (prerequisites, record fetch, numbering, cache insertion mechanics, epilogue), F1 bounds, F3 branched-LNT rejection, F7, M1–M4 defaults, §8 self-checks.

### 3.5 Split-k reducer wiring (Wave 2 — designed now, enabled by `ReducedDimWarp=true`)

A WARP-assigned REDUCED dim d (split-k over K_d warp coordinates):

- Walk: the dim's residue TBL (stride K_d·T_d, warp-coordinate rank) + lane-chunk TIL, exactly §3.1's WARP form. Each warp (…, c_d, …) accumulates a partial over its k-residue for the outputs its other coordinates own.
- Reducer protocol (per output element, unchanged shape): `Init` before the element walk → `Reduce(v)` per element → `Combine()` after. Under split-k, `Init/Reduce` run in each of the K_d warps; **`Combine` is called by ALL warps (uniform-call rule)** — warps outside this output's combine set call it too and stage the identity (zero `Reduce` calls ⇒ identity accumulates); the staging + `__syncthreads` + replicated fold over `warp_count()` then yields the full sum to every thread; the single owner (existing R3 single-owner store gate) stores.
- Ownership masks: lane-prefix over the lane-chunk walk (§1.4); the k-coordinate warp set = warps differing only in slot d.
- **Implementation risk note**: this is the most intricate emission in the refactor — implement behind the flag, land with a dedicated e2e numerical test (split-k matmul variant) before flipping the default.
- **Wave-1 defense in depth**: while `ReducedDimWarp=false`, tile-lower FAILS CLOSED (hard error) on any WARP-assigned dim whose walk feeds a reducer — if the decide-side feasible-area gate ever misses one, the pipeline must stop loudly, never emit an undesigned lowering.

## §4 Runtime — `runtime/`

### 4.1 Map constructors (replacing rt.md §4.2 portion decomposition)

- `GridMapping(KernelInfo, unsigned ndim, /*unsigned loC…*/)` and `ThreadMapping(KernelInfo, unsigned ndim, /*unsigned warpsPerDim…*/)` — each arg is the dim's ABSOLUTE extent (LoC_d / K_d), all ≥ 2 by construction (K=1 pruned; presets start at 2). No portions, no −1, no `Q = total/kPortion`.
- Decomposition: `index()` (no-arg) = the flat rank (blockIdx.x / threadIdx.x analogs); `index(d)` = the tuple coordinate via reverse stride-shape over the constructor shape. For ThreadMapping, `index(d)` is in WARP units; `lane_id()` gives the lane within the warp; `warp_index()`, `is_first_in_warp()`, `warp_count()` survive as pure functions of the flat rank.
- `total_cta()/total_thread_per_cta()` (KernelInfo) survive — they now return the DERIVED launch values.

### 4.2 Iterators (rt.md §4.3 — the design survives, the arithmetic changes)

- `SeqIterator(init, step, bound)` — unchanged.
- `CtaIterator(GridMapping, dosoc, loc, init, step, bound)` — `loc` semantic = the dim's absolute LoC; round step = LoC×T supplied by the caller (the emitted init/step/bound from §3.1); rank = the map's tuple coordinate.
- `ThrIterator(ThreadMapping, dosoc, loc, init, step, bound)` — for a WARP dim: `loc` = K_d, rank in warp units; for the coupled walk: the flat-rank form with step = blockDim (dosoc = the no-arg flat query).
- `Has()/Next()/Index()` faces unchanged; CTA-uniform round walk unchanged.

### 4.3 ValidateLaunch (rt.md §5)

New pre-conditions (replacing multiples-of-10/8, ≥256): `blockDim.x % 32 == 0 && blockDim.x/32 ≤ 32`; `gridDim.x ≤ MaxCTA` (the policy constant emitted by the compiler); `smemBytes + reserved ≤ device opt-in max` (queried — never hardcoded). The launch VALUES themselves are compiler-derived constants (§5.1) — validation remains defense-in-depth.

### 4.4 Reducers — unchanged

Seq/Warp/Global protocol survives verbatim (R1/R2 2026-10-23): staging row `#warps × sizeof(T)` + replicated fold is already a cross-warp combine ⇒ split-k capable (§3.5 uses exactly this).

### 4.5 Tests

CPU gtest of the factorization decomposition (reverse stride-shape over shape args; warp-unit coordinates; flat-rank identities) replaces the portion-decomposition tests. GPU e2e unchanged in shape.

## §5 Codegen — `internal/cg`

### 5.1 Launch derivation (replacing cg.md LR-15 / §9)

- Still NOT an IR node. **Derived from the loop annotations**: walk the emitted ForLoops' `soc/dosoc/loc/tile` KeyedAttributes, group per LNT (contiguous loop-nest region), compute per LNT `Π CTA LoC` and `Π warp K`, then `gridDim.x = max over LNTs`, `blockDim.x = (max over LNTs of Π warp K) × 32`. LoCs are compile-time constants ⇒ the launcher bakes literal `dim3(...)` values.
- `__launch_bounds__(<derived blockDim>)` replaces `MaxThreadsPerCTA` (D9 update).
- `ValidateLaunch(dim3(<derived grid>), dim3(<derived block>), <smemBytes>)` in the generated launcher (§9 update).
- Mapping emission: `tca_gm<N>(kinfo, <m, LoC…>)` / `tca_tm<N>(kinfo, <n, K…>)` — shape args replace portion args (line ~300 update).

### 5.2 Unchanged

D5 dynamic SMEM pool (offsets disjoint in allocation order; extents shrink under the new windows), D10 comments (LoC annotation now absolute), symbol conventions, Tensor contract + per-dim validation, CMake emission.

## §6 final-lower — `internal/opt/final_lower` + doc

- Mapping emission args (final_lower.md 237-238): `<dims + portions …>` → `<dims + shapes …>`.
- Worked example (355-395): rewrite for multi-dim + absolute shapes (the `<1, −1>` / `<2, 4, −1>` / AutoWarpPortion text is stale). The Reducer anchor gate + bracket law survive unchanged.

## §7 schedule / LICM — verified unaffected

The new loop shapes are ordinary ForLoops; effect-chain placement, derived fill placement, SchedSN/PrototypeBlock semantics all form-agnostic. Regression-test only (the existing placement pins must stay green).

## §8 Implementation plan

**Wave 1** (each task = atomic commits; build/vet/test green at every step):

- **T1 Decide pass** — `internal/opt/tiling`: Params (§2.1 + retirements), feasible-area iterator with incremental pruning (§2.2), constraints (§2.3), cost model rewrite (§2.4: tier rules, K_d×T_d windows, cross-warp reuse, FRAG = Π(T)×dtype), objective + tiebreaks (§2.5), eviction (§2.6), record (§2.7). Files: `record.go`, `search.go` (likely split into `search.go`/`judge.go`/`constraints.go` per the module split), `overlap.go` (site-facts additions for cross-warp sharing), tests. Verify: unit tests per module; the matmul worked example (§3.3) as a pinned expectation.
- **T1.1 Objective currency patch** (2026-10-26 amendment; T1 follow-up) — `internal/opt/tiling/judge.go`: `siteReuseCount` branches on class per §2.4 (SMEM → collective trips: CTA `T_d`, WARP `T_d`, SEQ `T_d × readers`; FRAG → per-thread trips unchanged) and `judgeSite` scales FRAG traffic by holders (≈ the LNT blockDim); promotion gate/eviction now see fill cost per §2.4. Search, enumeration, eviction order, verify cross-check, and record shape are UNCHANGED — only scores (and therefore possibly some selections) move. Verify: `TestFix33WorkedExample` re-pinned to the §3.3 re-pin (decisions expected unchanged, Reuse values change: rhs 4 → 256); NEW pin — the §3.3-pre-amendment vector (N=WARP/4/64, K=SEQ/64) and the winner score EXACTLY equal on the rhs site (unit-consistency regression guard); hermetic suite green.
- **T2 tiling-verify** — `verify.go`: new re-checks (§2.7), deletions. Verify: negative tests for each check.
- **T3 Runtime** — `runtime/`: map constructors (§4.1), iterator arithmetic (§4.2), ValidateLaunch (§4.3), decomposition tests (§4.5). Verify: CPU gtest + existing runtime tests ported.
- **T4 Tile-lower** — `internal/opt/tiling/lower*.go`: loop templates (§3.1), map binding (§3.2), cache insertion updates (§3.3), fallback → backstop. Verify: unit tests; goldens regen + re-pin (sanctioned; expected flip per §3.3).
- **T5 Codegen + final-lower** — launch derivation (§5.1), launch bounds, emission args (§5, §6). Verify: golden `.cu` fixtures regen + re-pin.
- **T6 E2E** — frontend matmul fixtures numerical acceptance vs CPU reference on the A6000 + compute-sanitizer (memcheck/racecheck) zero errors.
- **T7 Docs** — pointer-stub updates to loop_tile_lower.md / rt.md / cg.md / final_lower.md ("superseded by doc/fix/fix4.md §X" headers on the stale sections), runtime_objects.md row updates.

**Wave 2**: **T8 Split-k** — §3.5 behind `ReducedDimWarp=true`; dedicated split-k matmul e2e; flip default; goldens.

Gates: stdlib-only (module count 1); determinism pins (fixed enumeration, first-strict-max); fail-closed record gating (F7) intact; no barriers emitted by the compiler; goldens byte-re-pinned per regen with numerical e2e backing.

## §9 Acceptance criteria & risks

- **Accept**: (1) matmul_scale e2e numerically correct on the A6000 under the new schedule; (2) the §3.3 worked-example signature (both sites promoted, M=CTA/N=SEQ/K=WARP, rhs at the MaxCache cap; AND the rhs score of this vector is EXACTLY equal to the pre-amendment N=WARP/K=SEQ vector's — the one-currency regression guard); (3) sanitizer clean; (4) search deterministic across runs (same input ⇒ same record); (5) verify gates fire on corrupted records (negative tests); (6) runtime decomposition tests green; (7) zero-cache path exercised by a tight-budget fixture with a warning, not an error.
- **Risks**: (a) split-k wiring (§3.5) — mitigated behind the flag; (b) golden churn is large (expected — the cost model changed); (c) the LoC-magnitude tiebreak (§2.2) is a heuristic — revisit with the future utilization terms; (d) idle-CTA artifacts when extent/T < LoC (guarded, harmless, accepted — future formula work).
