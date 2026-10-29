# Review: Tile scheduling design docs

- **Targets:** `doc/tile_scheduling.md` (v1 prose), `doc/tile_scheduling2.md` (v2 layout-algebra formalization), `doc/tile_scheduling_cg.md` (codegen worked sample)
- **Method:** full read of all three documents; every arithmetic claim independently re-derived; cross-referenced against `doc/tca.md` (§2.4 tiling, §3.3 tiling rule, §5.5 iteration factoring, §5.7 automatic parallelization, §5.10 reduce lowering) and the project's whole-warp thread-group ruling.
- **Severity:** CRITICAL (math or notation wrong; a mechanical implementer would build incorrect code) / HIGH (ambiguity blocking a unique mechanical implementation) / MEDIUM (terminology or cross-document inconsistency).
- **Finding tags:** (F) factual bug — fix in the canonical doc; (Q) ambiguity requiring an author ruling; (I) cross-document inconsistency — unify.
- **Citation convention:** quoted phrases and section names come from the source doc named on each finding.
- **Verdict:** the three documents describe one consistent high-level model (factor the iteration polytope, assign SoC per layer, mechanical codegen + memory mapping + reduce lowering) and the core insight is sound; the documents disagree with each other on definitions, contain several arithmetic errors that propagate into the worked examples, and leave several design questions unstated or ambiguous. A single canonical doc can be assembled by fixing the bugs, unifying terminology, and flagging the open questions.

---

## 1. CRITICAL findings (factually wrong; implementer would produce incorrect code)

### F-1 [F, CRITICAL] RSR example arithmetic: partition product is 96, not 64

- **Source:** `tile_scheduling.md`, Reduce operation example.
- **Text:** "I = [1][3][2][2][8]" as a 5-way partition of size-64 1-D iteration space, followed by "it already reduced 8 * 2 * 2 * 2 = 64 == 64 elements".
- **Problem:** the stated partition has product 1 × 3 × 2 × 2 × 8 = 96, which violates the factorization invariant (Π_m F[m][i] = S[i]). The sentence "8 * 2 * 2 * 2 = 64" introduces a factor of 2 that is not present in the first position of the partition (which is 1, not 2).
- **Correction:** a consistent 5-way partition of [64] is `[1][2][2][2][8]` (product 64). RSR scanning inner-to-outer (layers 4→3→2→1→0) accumulates 8 (layer 4) × 2 (layer 3) × 2 (layer 2) × 2 (layer 1) = 64, stopping at layer 1 before covering layer 0 (the CTA layer, extent 1).
- **Note:** the example as written also has the RSR direction ambiguous ("backwards from 5'th tiling back to 1'th tiling" vs. inner-to-outer). The canonical doc must define RSR direction precisely (inner-to-outer = most-sequential first).

### F-2 [F, CRITICAL] tile() quotient vs block-size swapped in v2 coordinate mapping

- **Source:** `tile_scheduling2.md`, Tile Extension → Coordinates and Offset.
- **Text:** `tile(S, I) = O` with `O[i] = S[i] / I[i]`. Then "Offset[i]: index[i] * O[i]" and "subspace coordinate: parental_coordinate[i] % O[i] / index[i]: floor(parental_coordinate[i] / O[i])".
- **Problem:** with I as the per-subspace block extent and O as the number of subspaces, the offset of subspace `index[i]` must be `index[i] * I[i]` (each subspace is I wide), not `index[i] * O[i]`. The worked numbers prove it: S=[100,200,50], I=[2,2,10], so O=[50,100,5]. Subspace index 1 in dimension 0 has offset 2 (1 × 2), not 50 (1 × 50). The modular mapping is similarly inverted: the intra-subspace coordinate is `parental % I[i]` (range 0..I[i]−1) and the subspace index is `floor(parental / I[i])` (range 0..O[i]−1).
- **Correction:**
  - `Offset[i] = index[i] * I[i]` (subspace index × per-subspace block extent).
  - Intra-subspace coordinate domain: 0..I[i]−1.
  - `parental → subspace`: `intra[i] = parental[i] % I[i]`, `index[i] = floor(parental[i] / I[i])`.
  - `subspace → parental`: `parental[i] = index[i] * I[i] + intra[i]`.
  - Bijection: since O[i] × I[i] = S[i] and I divides S, this is a bijection between [0, S[i]) and [0, O[i]) × [0, I[i]).
- **Impact:** this single swap propagates through every downstream claim that uses "TSOffset" or coordinate transformation. Every numerical example that relies on Offset is wrong under the stated formulas (the prose may be describing the correct thing, but the formulas are inverted).

### F-3 [F, CRITICAL] cg sample table: stride row 1 is wrong; middle column is inconsistent

- **Source:** `tile_scheduling_cg.md`, header table with SHAPE / middle column / STRIDE.
- **Rule (from v2 and v1's SM(m)):** stride row m, column i = product of factor extents of all layers strictly below m in column i. This equals the per-unit region shape at layer m.
- **SHAPE matrix (given):**
  ```
  (0/CTA):  [4][4][1]
  (1/WARP): [2][2][2]
  (2/THR):  [4][4][2]
  (3/SEQ):  [4][4][4]
  ```
- **Re-derivation of per-unit region shape (= stride row):**
  - Row 3 (SEQ): product of nothing below = [1][1][1] → table shows [1][1][1] ✓
  - Row 2 (THR): product of row 3 = [4][4][4] → table shows [4][4][4] ✓
  - Row 1 (WARP): product of rows 2+3 = [4×4][4×4][2×4] = [16][16][8] → table shows [16][16][16] ✗
  - Row 0 (CTA): product of rows 1+2+3 = [2×4×4][2×4×4][2×2×4] = [32][32][16] → table STRIDE col shows [32][32][16] ✓
- **Middle column inconsistency:** the middle column is unnamed and disagrees with the stride column in row 0. Row 0 middle = [32][32][1], row 0 stride = [32][32][16]. They cannot both be "the region owned by one unit" because there is only one such quantity per layer.
- **Correction:**
  - Row 1 stride must be [16][16][8] (not [16][16][16]).
  - The middle column in row 0 must be [32][32][16] to match the stride column (or the two columns must be given distinct definitions — see Q-5).
  - The canonical doc uses one consistent "per-unit region shape" quantity and one stride representation; they coincide (stride row = per-unit region extent for each dim).

---

## 2. HIGH findings (ambiguity blocks unique implementation)

### F-4 [F, HIGH] MaxLoC constraint mis-indexed in v1

- **Source:** `tile_scheduling.md`, Tile Based Scheduling → Shape assignment → Constraints.
- **Text:** "For a specific tile, with assigned SoC to be X — Multiplication of all Sm[i] must be <= MaxLoC(CTA)".
- **Problem:** the bound is stated as `MaxLoC(CTA)` regardless of the tile's SoC. v2 correctly uses `MaxLoC(TileSoC[m])`. If a warp-level tile were bounded by MaxLoC(CTA) (2^31−1), there would be no effective bound.
- **Correction:** "Multiplication of all Sm[i] must be ≤ MaxLoC(X)", where X is the SoC assigned to that layer.

### Q-1 [Q, HIGH] Is F[m][i] = LoC[m][i] required for all non-SEQ layers?

- **Source:** cross-doc; v2 worked example has factor rows == LoC rows for every parallel layer; v2's quantity constraint only says `0 < PLoC[i] <= TS[m-1][i]` (i.e. LoC ≤ parent extent — parent being the input to that layer's tiling step).
- **Ambiguity:** the constraint allows LoC < factor extent on a parallel layer, meaning each SoC unit would sequentially iterate over the remaining factor. But none of the worked examples do this — all examples push all per-unit residual extents into the SEQ layer. Is F = L on all parallel layers a *required canonicalization*, or merely the common case? This determines whether codegen ever emits sequential loops at a CTA/WARP/THREAD level (i.e. a thread walks multiple sub-tiles of its assigned tile).
- **What each reading changes:**
  - **Required (F=L on parallel layers):** the schedule search picks LoC (parallel) extents, and the factor is exactly LoC; all remaining per-unit sequential work lives in the SEQ layer. Simpler codegen (only one layer has sequential loops).
  - **Optional (F > L allowed):** a CTA's tile can be larger than the CTA LoC, meaning each CTA walks multiple sub-tiles of its work sequentially. Adds sequential loops at any layer.
- **Provisional (used in canonical doc §5 and §9):** F = L for all non-SEQ layers; all per-unit sequential slack is pushed into the single SEQ layer. Marked "(provisional, see Q-1)" wherever it matters.

### Q-2 [Q, HIGH] Cross-thread collection: shuffle vs SMEM(thread) region

- **Source:** v1 §5 table (WARP → SMEM, THREAD → SMEM) vs cg sample (thread combine uses `__shfl_down_sync`).
- **Ambiguity:** the storage-class table lists both WARP-owned and THREAD-owned regions as "SMEM", implying the cross-thread collection uses a thread-owned SMEM buffer. But the cg sample's `reducer.combine<thread>` uses `__shfl_down_sync`, which is a register-file (warp shuffle) path, not a load/store path through SMEM(thread). These are distinct mechanisms with different performance characteristics and different hardware scope (shuffle is intra-warp only; SMEM(thread) could in principle span warps).
- **What each reading changes:**
  - **Shuffle for thread-level combine:** uses `__shfl_sync` family; limited to within-warp; no SMEM allocation for thread-owned buffers in the reduce path.
  - **SMEM(thread) for thread-level combine:** each thread owns a slot in a SMEM region; inter-thread combine goes through shared memory; works across warps (with appropriate sync).
- **Note:** the storage-class model (storage class + owning SoC) is compatible with both — the question is which mechanism is selected and when. Flagged as Q-2.

### Q-3 [Q, HIGH] Role of reducer.Init at layers above the RSR stop layer

- **Source:** cg sample: RSR for k (extent 16 = 2·2·4 in WARP×THR×SEQ) is exhausted at WARP, yet a `<cta,...>` reducer is initialized and the final step is a write-out, not a combine.
- **Ambiguity:** what does `reducer<cta>` represent when the reduced dimension's RSR stops at WARP? Is it: (a) a no-op scaffolding object that just owns the output buffer shape, (b) an actual accumulator that the CTA-level combine would feed into (but there is no CTA-level combine because RSR is done), or (c) the output-element allocation at the CTA-owned SMEM level? The cg sample initializes it but never combines into it from below — the warp combine writes into `acc_warp` at warp-owned SMEM, and then there is a `mem_write(gout, ...)` at the end with no cta-level combine shown.
- **What each reading changes:**
  - **Output buffer at CTA-owned SMEM:** Init allocates the CTA-owned output region; each warp writes its result into its slice; no CTA combine needed because each warp owns distinct output elements (the surviving non-reduced dimensions are partitioned across warps).
  - **Accumulator scaffolding:** Init at a layer above RSR is a no-op / structural placeholder and is always paired with no Combine at that layer.
- **Flagged as Q-3.** The canonical doc §8 states the RSR sequence explicitly and notes this question.

### I-1 [I, HIGH] Stride representation differs across docs

- **Source:** v1 defines SM(m) as an N×N diagonal matrix; v2 prints an M×N stride matrix; the cg sample uses a 3-column table.
- **Unification:** define both — per-layer diagonal stride matrix SM(m) for algebraic manipulation, and the M×N stride matrix whose row m is diag(SM(m)) — for tabulation and codegen. Use the M×N stride matrix consistently in prose and examples.

### I-2 [I, HIGH] Factor matrix vs LoC matrix relationship (see also Q-1)

- **Source:** v1's Sm[i] is the factor extent per dimension at layer m, and LoC is a scalar per SoC; v2 separates factor matrix (TS) from LoC matrix (PLoC).
- **Unification:** canonical doc uses two distinct matrices: F (factor matrix, M×N, product-over-rows = S) and L (LoC matrix, M×N, with L[SEQ] = 1 everywhere). The quantity constraint is 1 ≤ L[m][i] ≤ F[m][i]. With the provisional Q-1 reading, L[m][i] = F[m][i] for all non-SEQ m.

---

## 3. MEDIUM findings (terminology, consistency, minor bugs)

### F-5 [F, MEDIUM] Reducer level typo: CTA/CROSS-WARP saves SMEM(warp)

- **Source:** `tile_scheduling.md`, §6 Reduce translation → item (2).
- **Text:** "reduce in CTA/CROSS WARP, saving SMEM(warp)".
- **Problem:** in the hierarchy being described (sequential → warp → cta → gmem), the CTA/cross-warp level must save to SMEM owned by the CTA, not by the warp.
- **Correction:** SMEM(cta).

### F-6 [F, MEDIUM] Warp-group MaxLoC = 32 × 7 is wrong

- **Source:** `tile_scheduling2.md`, MaxLoC table: "MaxLoc(WARP GROUP), 32 * 7".
- **Problem:** 32 × 7 = 224, which is not a multiple of a warpgroup's size. A warpgroup is 4 warps = 128 threads. Max warpgroups per CTA = total warps per CTA / 4 = 32 / 4 = 8. So MaxLoC(WARP GROUP) should be 8 (warpgroups per CTA) or 1024 threads (8 × 128). The value 224 does not match any hardware limit.
- **Correction:** 8 warpgroups per CTA (32 warps ÷ 4 warps/warpgroup). Cite provenance: CUDA maximum 32 warps per CTA / 4 warps per warpgroup = 8.

### I-3 [I, MEDIUM] Running examples differ across docs

- **Source:** v1 uses `[128][128][16]` (matmul), v2 uses `[128][128][128]`, cg sample uses `[128][128][16]`.
- **Unification:** canonical doc uses one running example end-to-end. Recommend `[128][128][16]` (matmul, consistent with the project core value and the cg sample).

### F-7 [F, MEDIUM] Vocabulary typos that change parseability

- **Source:** both v1 and v2.
- **Issues:**
  - "Space of Currency" (SoC) → correct term is **Space of Concurrency** (the project's standard term, used in tca.md and code).
  - "Level of Concurrency" (LoC) — this one is correct but must not be confused with "lines of code"; the canonical doc uses LoC with a clear definition.
  - "PI(PLoC[i]) <= TS[m-1][i]" with TS[m-1] undefined for m=0 (no row -1). Define S as virtual row -1: TS[-1] = S (the input shape).
  - "Tripcount is 1, normalized; init is 0" — unintelligible. State that each loop in an LNC has trip count equal to the factor extent at that layer, and that "normalized" means the induction variable runs from 0 to extent−1 with step 1.
  - Spelling: "heirachy" → hierarchy; "parallisation" → parallelization; "conguruence" → congruence; "polytop" → polytope; "coorindate" → coordinate; "dentoed" → denoted; "iteraiton" → iteration; "irrlevant" → irrelevant; "paritition" → partition; "diagnal" → diagonal; "aixs" → axis; "parallisation" → parallelization; "paritioni" → partition; "concurency" → concurrency; "totall" → total; "heirachy" → hierarchy.
- **Correction:** fix all in the canonical doc.

### Q-4 [Q, MEDIUM] Reduce axis ordering and generalization

- **Source:** all three docs say reduce collapses the last dimension; v1 ties it to polytope dimension N-1; v2 doesn't specify which axis is reduced.
- **Ambiguity:** does the scheduler require the reduced axis to be last in the factor/stride matrices? Who permutes it there if not — the frontend (tca.md §2.5 and §2.7 already pin last-dim semantics via `as`), or the scheduler? How does RSR generalize when there are multiple reduced dimensions or a non-last reduced axis?
- **Clarification (consistent with tca.md):** the reduce operates on the last dimension of the producer map's `as` order. The scheduler sees the iteration dimensions in `as` order and the reduced axis is the last one. For v0 with exactly one reduce per map chain, there is exactly one reduced axis and it is last. Multiple reduced axes = chained reduces, each last-in-its-`as`-order at the time of reduction.
- **Open question:** how does RSR generalize to multiple reduced dimensions? Flagged as Q-4.

### I-4 [I, MEDIUM] Positioning vs existing pipeline docs

- **Source:** all three docs vs `doc/tca.md` §5.5.
- **Note:** tca.md §5.5 ("Iteration Factoring") currently says "Covered by the canonicalization step in Section 5.2" — i.e. it's a one-liner deferral. The tile scheduling docs describe the full factoring + scheduling + mechanical-codegen expansion. The canonical doc should state its relationship to tca.md §5.5: this document is the detailed expansion of iteration factoring plus SoC assignment plus codegen, and is consistent with tca.md's vocabulary (per-LNT absolute LoC extents, whole-warp thread groups).

### F-8 [F, MEDIUM] Codegen sketch has multiple variable/bound defects

- **Source:** `tile_scheduling.md` loop-nest sketch and `tile_scheduling_cg.md` pseudo-code.
- **Issues (all in illustrative pseudo-code, not in algorithm definitions):**
  1. v1 CTA loop nest: third loop `cta_2` has bound `4` but LoC row for CTA is [4][4][1] (third dim = 1, loop should trip once).
  2. cg sample: `auto cta_2_offset = cta_1 * 32 + blockIdx.y;` — reuses `cta_1` variable (copy-paste error; should be `cta_2 * 32 + blockIdx.y` or similar).
  3. cg sample: `warp_*_offset` and `thr_*_offset` are used in outer scopes before the loops that define them (e.g. `load_frag` at CTA level uses `warp_0_offset` which is defined inside the warp loop).
  4. cg sample: `fragment acc_thr[4][4]` vs `reducer.init(acc_frag)` — name mismatch (`acc_thr` vs `acc_frag`).
  5. cg sample: reducer shapes `reducer<cta, shape<2,2>, ...>` / `<warp, shape<4,4>, ...>` / `<thread, shape<4,4>, ...>` drop the reduced third LoC component (WARP [2,2,2], THR [4,4,2]) with no stated rule for why the reducer's shape omits dims.
  6. cg sample: `if (...)` body guard around the compute is unexplained — is it a tail predicate (for non-divisible tiles) or a reduce-axis membership check?
- **Correction:** the canonical doc labels the pseudo-code as illustrative, and makes variable names, bounds, and scoping self-consistent. The reducer shape question is part of Q-3.

---

## 4. Confirmed sound — preserve as-is

1. **Core model:** iteration-space tiling as factorization into M layers, SoC assigned per layer, mechanical codegen from factor/LoC/stride matrices — sound, consistent with tca.md's tile rule (§2.4, §3.3).
2. **Map invariance under tiling:** Map(I, λ) = Map(I ~ M, λ) — correct; tiling is a bijection on the iteration space and map is unordered element-wise application.
3. **Reduce invariance under tiling (with associative+commutative combiner):** Reduce(I, λ) = Reduce(I ~ M, λ) when the cover is a partition — correct; the RSR defines which layers participate and in what order.
4. **Memory hierarchy mapping by SoC owner:** FRAG (SEQ), SMEM with owner (thread/warp/cta), GMEM (CTA) — consistent with the project's 3-verb memory IR (MemNew/MemLoad/MemStore + storage class).
5. **Hoisting rule (concatenation shape):** buffer hoisted to level m has shape = (owner-coordinate extents at m) ++ (inner buffer shape) — conceptually sound; see §7 of the canonical doc for the formal statement.
6. **SoC priority order CTA → WARP GROUP → WARP → THREAD → SEQ (outer to inner):** consistent with the project's whole-warp thread-group model and CUDA's execution hierarchy.
7. **RSR intuition:** reduce stops at the outermost layer whose covered product equals the reduced-axis extent — correct in concept; the exact arithmetic and direction need fixing (F-1).
8. **Storage class + owning SoC model:** consistent with project memory model (FRAG owned by thread/SEQ, SMEM regions parameterized by owner, GMEM).

---

## 5. Minimal revision set (numbered)

The canonical doc should apply the following changes, in this order:

1. **Fix tile() semantics** (F-2): O = S/I is number of subspaces; I is per-subspace block extent; offset = index × I; inverse mapping uses mod I and div I. Update all downstream formulas.
2. **Fix stride/region definitions** (F-3, I-1): define stride row m = per-unit region shape at layer m = product of factor extents of layers below m. Use M×N stride matrix consistently. Correct the cg sample table row 1 stride to [16][16][8].
3. **Fix RSR example** (F-1): use consistent partition [1][2][2][2][8] for size-64 example; define RSR direction (inner-to-outer); state RSR formally.
4. **Fix MaxLoC constraint** (F-4): bound is MaxLoC(X) where X is the layer's SoC, not always MaxLoC(CTA).
5. **Fix reducer level typo** (F-5): CTA/cross-warp saves SMEM(cta).
6. **Fix warp-group MaxLoC** (F-6): 8 warpgroups per CTA; add provenance table for all MaxLoC values.
7. **Unify terminology** (I-2, I-3, F-7): "Space of Concurrency" (SoC), LoC matrix vs factor matrix, single running example [128][128][16], fix spelling throughout.
8. **Clean up pseudo-code** (F-8): label all codegen sketches as pseudo-code; make variable names/bounds/scoping self-consistent.
9. **Flag open questions** (Q-1 through Q-5): each in a dedicated section of the canonical doc, with both possible readings and mechanical impact stated.
10. **Relate to tca.md** (I-4): position the canonical doc as the detailed expansion of tca.md §5.5 (iteration factoring) + §5.7 (automatic parallelization) + §5.10 (reduce lowering).

---

## 6. Open questions for the author — RESOLVED (rulings recorded 2026-10-28)

| ID | Question | Possible readings | Mechanical impact |
|----|----------|-------------------|-------------------|
| Q-1 | Must factor extent equal LoC on every non-SEQ layer? | (a) Required — all per-unit sequential slack pushed to SEQ; (b) optional — parallel layers can carry per-unit sequential extent. | (a) simpler codegen (only SEQ has sequential loops); (b) larger schedule space, sequential loops at any layer. |
| Q-2 | When is cross-thread collection via shuffle vs via SMEM(thread)? | (a) shuffle for intra-warp thread combine; (b) SMEM(thread) region for all thread-level combine; (c) both with selection rule. | Affects reducer lowering path, SMEM budget accounting, warp vs cross-warp applicability. |
| Q-3 | What is the normative role of reducer.Init at layers above the RSR stop layer? | (a) output buffer allocation at that layer's owned SMEM; (b) no-op scaffolding; (c) CTA-level accumulator for cross-warp combine (but RSR says no combine). | Affects codegen shape of multi-level reducer, SMEM allocation per layer. |
| Q-4 | How does RSR generalize to multiple reduced dimensions? | (a) each reduce axis has its own RSR, independent; (b) reduces share the layer sequence. | Affects scheduling of multi-reduce chains, reducer stack shape. |
| Q-5 | Does the cg sample's middle column represent a different quantity than stride, or is it just wrong? | (a) same quantity (stride = per-unit region shape) — middle col has a typo; (b) different quantities (e.g. "output region shape" vs "stride"). | Determines whether the canonical doc needs one or two region-shape quantities. |

### Rulings (author, 2026-10-28)

- **Q-1 → reading (a), enforced by construction.** LoC = 1 on a dimension means it does not participate in that layer's SoC assignment; the full-factorization constraint guarantees any dimension untouched by all parallel layers is picked up by the SEQ layer automatically. Canonical doc §5.4 now derives the SEQ row as the residual.
- **Q-2 → per-thread combine is grouped shuffle.** Fine-grained axis sharding within a warp rules out the traditional single-axis full-warp reduction; the runtime groups lanes by layout modular arithmetic and reduces per group with `__shfl_xor`, then writes each group's combined value into the per-warp accumulator (wmem = SMEM replicated per warp). Canonical doc §7.1.1, §8.3 step 2.
- **Q-3 → Init = allocate + identity-initialize that tier's intermediate memory.** Each SoC tier's intermediate storage receives the tier below's combined values for the next tier; the ladder stops at the RSR layer, above which the initialized memory stages the output for write-out. Canonical doc §8.3.1.
- **Q-4 → reduce is always single-dimension; RSR is per-axis.** Unvisited (outer) layers must carry LoC = 1 on the reduced dimension. Canonical doc §8.2.
- **Q-5 → one quantity (stride row = per-unit region shape); the cg middle column is erroneous.** The LoC unit may be coarse-grained (thread by default; warp or warpgroup units expressible via the alignment/modularity constraint), which does not split the region/stride quantity. Canonical doc §4.2–§4.3, §10.
