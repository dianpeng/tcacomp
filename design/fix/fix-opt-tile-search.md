# The loop-tiling schedule search algorithm (finalized design)

- Status: FINAL — all rulings folded in (design review doc/review/fix-opt-tile-search.md, rounds 1–6, 2026-10-26).
- Purpose: replace the naive PSched search in `internal/opt/tiling` (uniform tile, no LoC search, binary budget gate, ≤1+≤1 policy) with a formalized, general-purpose optimizer over per-dim (SoC, LoC, TileSize) assignments with cache eviction.
- Supersedes (explicitly):
  - The **disjoint-shard warp model** (2026-10-20/21 composition rulings: "threads = sum of disjoint whole-warp shards", coupled walk = remaining warps, the F2 floor) → replaced by **factorization at both levels**.
  - The **portion grammar** (PORTION_CTA/PORTION_WARP percentages, −1 remainder, AutoWarpPortion) → replaced by **absolute counts**.
  - The **v0 searcher policy** (≤1 CTA dim + ≤1 independent WARP dim per LNT) → **repealed**; multi-dim same-category assignment is the point of MAPM.
  - The runtime **assert-and-trap pre-conditions** (CTAs multiple of 10; ≥256 threads, multiple of 8 warps) → replaced by the launch-derivation contract below.
  - The **weighted two-term objective** of earlier drafts → **traffic-avoided primary, partition-count tiebreak only**.

## 1. Input

- A loop nest of N iterations. The search window is the **M innermost consecutive** iterations of the nest's root→deepest-leaf spine (loops outside the window default SEQ; non-spine loops default SEQ).
- **M = 4** (default; `Params.WindowK` changes 5 → 4).

## 2. Schedule Assignment

Each window iteration gets one **Schedule Assignment**:

- **(1) SoC** — space of concurrency: `Sequential | CTA | WARP`.
- **(2) LoC** — level of concurrency, an ABSOLUTE count (portions/percentages are invalid and deleted):
  - CTA: **# of CTAs allocated to this dimension** (a preset factor).
  - WARP: **# of warps allocated to this dimension** (a preset factor; 1 warp = 32 threads).
  - Sequential: always 1 (implicit, not searched).
- **(3) TileSize** — the per-dimension tile extent owned per round; presets {32, 64, 128, 256}. Tile 1 is not a legal schedule (2026-10-24 ruling); every entry ≥ 2.

### 2.1 Composition = FACTORIZATION (both levels)

- **CTA space**: `gridDim.x = Π (CTA LoC over CTA-assigned dims)`. Each CTA has a coordinate (via reverse stride-shape decomposition of `cuda_linear_id`); in dim d, **CTA c owns tiles ≡ c (mod LoC_d)**.
- **Warp space**: the warp index factors uniquely into one coordinate per WARP-assigned dim, `Π (warp K) ≤ 32` warps total; in dim d, **warp c owns tiles ≡ c (mod K_d)**.
- Every tile combination is owned by **exactly one (CTA, warp)** — no redundant execution, no idle warps. `blockDim.x = Π K × 32` **exactly**: we launch precisely what the schedule factorizes (exhaustive launch).
- A WARP dim with K = 1 is vacuous distribution ≡ SEQ — pruned from the feasible area (dominated).

### 2.2 Structural hierarchy (TBL / TIL, in factorization terms)

Each original iteration becomes a Tile-Between-Loop and a Tile-Inside-Loop:

| SoC | TBL (between) | TIL (inside) |
|---|---|---|
| CTA | CTA residue-class tiles (coordinate mod LoC) | **coupled walk**: distributed over ALL `blockDim` threads, element-strided (thread t owns every blockDim-th element) |
| WARP | warp residue-class tiles (coordinate mod K) | owning warp's **32 lanes split the tile into consecutive ⌈T/32⌉ chunks** |
| SEQ | sequential rounds (every thread) | each thread walks the full T interior sequentially (its own coordinates) |

- Expanded SoC instances (what the constraints count): a CTA dim contributes **1 CTA instance + 1 THR instance** (its coupled TIL); a WARP dim contributes 1 THR instance; SEQ contributes none. "A CTA assignment implicitly uses 2 SoC."
- F2 (coupled-portion floor) is **deleted** — there is no "remaining warps" concept; `coupledThreads := blockDim`.

## 3. Constraints

Checked over the **expanded instance list** (per §2.2) and the **factorizations** (per §2.1):

- **(1) MAPM** — max parallelization mapping: `#CTA dims ≤ MaxPartition.cta = 3` and `#THR instances ≤ MaxPartition.thr = 3` (WARP dims + CTA dims).
- **(2) MALoC** — factorization bounds: `Π CTA LoC ≤ MaxCTA` (Params; default 512 — a real policy bound, not the vacuous 2^31−1) and `Π warp K ≤ 32` (hardware: 1024 threads).
- **(3) MASRAM** — TWO budget gates (units differ; a single pool alone is unsound — round-5 R1 ruling):
  - SMEM: kernel-wide sum of promoted SMEM windows ≤ `MaxCache` = 64 KB.
  - FRAG: kernel-wide per-thread FRAG sum ≤ `MaxFragPerThread` = 256 B. A FRAG site is **all covered dims SEQ-assigned** — the window lives in a single thread's registers and is walked wholly by it; per-thread bytes = `Π(covered T_d) × sizeof(dtype)`. (At 256 B, 2-D all-SEQ fragments (32×32 fp16 = 2 KB) decline — accepted for v0; the knob is the hyperparameter surface.)
- **(4) CuMAP** — at least one CTA-assigned dim (CUDA thread-model requirement). Violation = structural infeasibility = hard error.

### 3.1 Kernel-launch derivation (fully dynamic — no static one-size-fits-all)

- `gridDim.x = max over LNTs of (Π CTA LoC)`; `blockDim.x = max over LNTs of (Π warp K) × 32`.
- Each LNT **remaps its own Grid/Thread mappings** from its own factorization at its nest head (per-LNT scoped maps — the existing F5 mechanism). CTAs/warps beyond a smaller LNT's product **whole-body-skip** via the existing rank/extent guards, so the max-over-LNTs launch is safe.
- Runtime pre-conditions (replacing assert-and-trap): blockDim is a multiple of 32 with warp count ≤ 32; gridDim ≤ MaxCTA.

## 4. Objective function

- **Primary (strict): maximize GMEM traffic avoided** = `Σ over promoted sites of bytes × (reuse − 1)`, computed on the **post-eviction** site set (§6).
- **Tiebreak (exact ties only): partition count** GoC = `total_cta + total_thr` over expanded instances (≤ 6); i.e. `goc_score = (total_cta + total_thr) / 6`. GoC NEVER overrides a traffic difference — hypothetical utilization never beats measurable bytes (round-5 R2 ruling).
- **Determinism**: fixed enumeration order (outermost window dim = most significant digit; digits ordered SoC, then LoC ascending, then tile ascending), first strict maximum wins.

> **KNOWN BIAS — objective evolution note (author ruling, 2026-10-26).** The tiebreak prefers parallelism over sequential (e.g. warp split-k on a reduced dim: 4/6 vs 3/6 partitions) whose reduction-staging cost (SMEM staging + barriers per output round) is **not modeled**. This bias is **accepted for v0**; there is deliberately **no special-case policy** (e.g. "reduced dims stay SEQ") — we are not certain sequential is better for reduction either. The search algorithm is formula-agnostic: enriching the score with more factors (reduction cost, shape-aware utilization via the dormant NominalExtent knob) improves the optimum **without any algorithm change**. Polish the formula later, not the searcher.

## 5. Cost model (site judgments)

- **Discovery** (existing, unchanged): `findINS` + `analyzeOverlap`/`newSiteFacts` in overlap.go — INS sets (cardinality ≥ 2 filter: a singleton set has no reuse), tie = innermost non-referenced iter, covered = the trailing `CacheDims` (=2) cached dims, reuse dims = tie + iters strictly inner than the tie, minus covered. Tensor rank ≥ CacheDims; `MaxCS` caps the candidate set.
- **Tier rules** (in order):
  1. Any covered dim **CTA-assigned → site REFUSED** (no coherent fill variant; kept for this refactor, revisit later).
  2. **C1b forced-FRAG**: any non-covered referenced iter (INS) assigned CTA or WARP — a thread-varying source coordinate — forces ClassFRAG regardless of the covered dims' SoC.
  3. Any covered dim **WARP-assigned → SMEM window** class (cooperative fill, buffer holds one window per warp coordinate of the WARP-covered dims).
  4. All covered dims SEQ → **FRAG** (per-thread registers).
- **Bytes**:
  - FRAG (per thread): `Π(covered T_d) × sizeof(dtype)` — no lane division.
  - SMEM window (buffer): `Π over covered dims of (factor_d × T_d) × sizeof(dtype)`, where a WARP-covered dim contributes `factor_d = K_d` (its per-round residue tiles, one per warp coordinate) and a SEQ-covered dim contributes `factor_d = 1`.
- **Reuse** = the number of times a filled window is re-read before its next fill: the per-thread/per-lane interior walks of the site's re-traversal dims (tie + inner non-covered), **times the cross-warp sharing factor** — a load whose subscripts MISS a warp dim d is re-read by every warp sharing the other coordinates (factorization makes windows genuinely shared; the shard model's "all reuse is intra-thread" identity no longer holds).
- **Promotion gate**: promote iff traffic avoided (`bytes × (reuse − 1)`) > 0 (total view, cross-warp included); FRAG bytes ≤ MaxFragPerThread per site.
- **Cross-LNT**: SMEM and FRAG sums are KERNEL-WIDE — running floors threaded through LNTs in deterministic `Roots()` order; site dedupe by MemLoad id (D11).

## 6. Search algorithm

Exhaustive search with greedy cache eviction:

1. **Enumerate** the feasible area (§7) in fixed order, with **incremental constraint pruning** (reject a partial vector as soon as MAPM/MALoC/CuMAP is violated — never enumerate-then-filter).
2. **Judge sites** for each candidate — memoized: a site's judgment depends only on the (SoC, LoC, TileSize) of the dims it touches, so cache judgments across candidates sharing those.
3. **Budget check** (both MASRAM gates against the kernel-wide floors). If violated: **greedy eviction (Chaitin–Briggs-style spill, not backtracking)** — repeatedly drop the site with the least value density, where `density = value/bytes = reuse − 1` (ties: smaller bytes first), until both gates pass. Evicted sites are demoted to GMEM; their tile shapes remain (loops stay tiled).
4. **Score** the post-eviction set: traffic avoided (primary), partition count (tiebreak). Update on strict improvement; on exact tie keep the earlier-enumerated candidate.
5. **Return** the best schedule found.

- **Structural infeasibility** (no CTA dim possible, no legal LoC factorization, tile < 2) → **hard error** (extends the 2026-10-24 "tiling never disabled" ruling: loops are always tiled ≥ 2).
- **Zero-cache outcome** (budget forces evicting ALL sites) → **allowed, emit a warning** (round-5 R3 ruling — graceful degradation is the point of eviction; erroring at zero would make eviction useless exactly when budgets are tight).

## 7. Feasible area

- Per window dim: SoC (3) × LoC preset × TileSize (4), where LoC presets are
  - CTA: {2, 4, 8, 16, 32, 64}
  - WARP: {2, 4, 8, 16, 32}
  - SEQ: {1} (implicit)
- K = 1 WARP dims pruned (≡ SEQ, dominated); presets are the parallelism ceiling (there is no remainder rule) — {…, 32} lets a single warp dim reach a full 1024-thread block.
- Upper bound per LNT: `((1 + 6 + 5) SoC-LoC combos) × 4 tiles = 48` per dim → `48^4 ≈ 5.3M` candidates — tractable ONLY with the incremental pruning and memoization of §6.

## 8. Implementation

### 8.1 Module separation (the current code mixes all of these)

1. **Iterator** — feasible-area enumeration with incremental constraint pruning.
2. **Constraints + eviction** — MAPM/MALoC/CuMAP checks over expanded instances; the two MASRAM gates; the spill loop.
3. **Score** — the site judge (tier rules, bytes, reuse) + traffic objective + partition tiebreak, on the post-eviction set.
4. **Update** — strict-max with fixed-order determinism.
5. **Return** — best schedule + budget consumption for the caller's running floors.

### 8.2 Parameters (all as pass input, sane defaults; the hyperparameter-optimizer surface)

`WindowK=4`, `TileSizes={32,64,128,256}`, `CtaLoCPresets={2,4,8,16,32,64}`, `WarpLoCPresets={2,4,8,16,32}`, `MaxPartitionCta=3`, `MaxPartitionThr=3`, `MaxCTA=512` (gridDim policy cap), `MaxWarpsPerCTA=32` (hardware), `MaxCache=65536`, `MaxFragPerThread=256`, `CacheDims=2`, `MaxCS=16`. **Retired**: PortionCTA/PortionWarp/AutoWarpPortion/WarpPortion/MaxThreadsPerCTA/MaxCTA-as-exhaustion.

### 8.3 Contract changes (the refactor's real size)

- **Record**: per-iter (SoC, LoC absolute, TileSize) keyed by BIV; per-site (promoted/tie/class/reuse/bytes/reason) keyed by MemLoad id. Launch derivation recomputed by lower/codegen (gridDim = max_LNT Π, blockDim = max_LNT Π × 32) — no new record field.
- **tiling-verify**: `CTA LoC == MaxCTA` → `LoC ∈ presets ∧ Π == gridDim`; warp `Π K ≤ 32`; F2 check deleted; the two budget gates re-checked kernel-wide.
- **lower (loop_tile_lower)**: residue-class loop rewrite forms (replacing chunk-stride shard forms); coupled walk = blockDim-strided; fills per the re-derived tier rules; runtime contract updated (runtime_objects.md).
- **Pipeline position unchanged**: tiling@114 → tiling-verify@115 → loop-tile-lower.
- **Goldens WILL change** (new cost model → new chosen schedules; e.g. the flagship is expected to flip from M=CTA/N=SEQ/K=WARP to the canonical M=CTA/N=WARP/K=SEQ) — sanctioned regen + re-pin + e2e numerical verification on the A6000, per the fix3 C11 precedent.

### 8.4 Sequencing

0. **Downstream spec revisions** (audit: doc/review/fix-opt-tile-search.md round 7) — loop_tile_lower.md (§6bis PHASE 3 forms, §7 v0-policy inheritance, §7bis portion grammar + launch validation + counting invariant, R2 window sizing, C1b sizing, worked example), rt.md (§4.2 map constructors, §4.3 iterator arithmetic, §5 ValidateLaunch, §12 tests), cg.md (LR-15 + §9 launcher derivation, D9 launch bounds, mapping emission args), final_lower.md (mapping emission + worked example). Three new design items must be written: ThreadMap dimensionality under factorization (ruling needed), launch-derivation ownership in codegen, split-k reducer wiring in tile-lower (WARP-assigned reduced dims are reachable — no v0 policy). Runs in PARALLEL with step 1.
1. Decide-half rewrite (cost model §5 + searcher §6 + verify §8.3) with unit tests.
2. Lower-half rewrite (rewrite forms, fills, runtime mapping contract) — consumes the step-0-revised specs; goldens regen + re-pin.
3. E2E: numerical acceptance + compute-sanitizer (memcheck/racecheck) on the A6000.
