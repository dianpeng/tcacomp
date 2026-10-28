# Tensorization by Scheduling Constraints (ldmatrix + mma)

This document specifies how tensor-core instructions (`ldmatrix`, `mma.sync`) are introduced into the tile-scheduling pipeline. It extends the canonical scheduling model in [tile_scheduling.md](./tile_scheduling.md) — read §4 (factorization), §5 (SoC/LoC), §7 (memory hierarchy), §8 (reduce scheduling / RSR) there first.

## 1. Thesis and Scope

**Thesis.** Hardware-instruction support is a *property of the scheduling constraint system*, not a pass. Every requirement an instruction imposes is expressed as one of:

1. an **arithmetic constraint on the tile matrix** (factor extents, LoC, lane grids), resolved by the scheduling solve;
2. a **remediation recipe** — a small, enumerated IR insertion (conversion, transpose, padding) materialized as ordinary code and lowered normally.

Given a schedule that satisfies bundle (1), introducing the instruction is a **mechanical rewrite**: a pass that *verifies* the scheduled shape against the instruction's equations and *substitutes* the instruction. The rewrite makes no decisions.

**Hardware targets (sm_86, fixed by project constraints).**

| Instruction | Shape / variants | Note |
|---|---|---|
| `ldmatrix.sync.aligned.m8n8.x{1,2,4}.shared.b16` | 8×8 b16 tiles; x1/x2/x4 = 1/2/4 tiles per warp collective | sm_75+; `.trans` qualifier for operand transposition |
| `mma.sync.aligned.m16n8k8.row.col` | f16·f16→f32; bf16→f32; tf32 (f32-reinterpret); s8 imma | sm_80+ m16n8k8 family |

Excluded (sm_90+): TMA / `cp.async.bulk(.tensor)`, `wgmma`, TMEM, cluster features. The sm_86 async-copy ceiling remains `cp.async` (fills).

**Relationship to prior rulings.**

- Memory IR stays exactly three verbs (`MemNew/MemLoad/MemStore` + storage class); an instruction is **codegen-level fusion over the verbs**, never a fused IR node (2026-10-25 ruling).
- Lowering stays mechanical: no sweep beyond the verb's rule (standing ruling) — the rewrite pass verifies and substitutes, nothing more.
- The reducer is not an opaque object (Q-2/Q-3 rulings): the ladder and the mma interact only through RSR accounting (§8).
- Parallelization is compiler-owned; the instruction *nomination* is a scheduling output.

**Staged adoption.** An `ldmatrix`-only stage (fragment loads feeding scalar FMAs) is legal and shares every constraint of the full stage, but the fragment layout shards the reduced axis across lanes, so the compute side then needs the Q-2 grouped shuffle to combine k-quads. The target stage is the full cluster (§4).

---

## 2. Model Extension: Stride-Decorated Factor Matrix

The canonical model factors each dimension into unit-stride layers: `g[i] = Σ_ℓ c_ℓ[i]·stride[ℓ][i]` with strides *derived* as products of factors below. Tensorization requires pinning strides that the hardware chose, so the model gains a **stride decoration**:

```
layout = (extents F[ℓ][i], strides S[ℓ][i])
global coordinate:  g[i] = Σ_ℓ S[ℓ][i] · c_ℓ[i]
```

- In the plain (non-tensorized) model, `S[ℓ][i] = Π_{r>ℓ} F[r][i]` — derived, unchanged.
- In a tensorized region, the `(THREAD, REG)` strides are the **hardware fragment map** — pinned, not derived.

**Digit-stack validity (mixed-radix condition).** A dim's layer digits `(c_k)` with `(S_k, F_k)` reconstruct that dim bijectively iff, ordered by stride, each digit's stride equals the product of the lower digits' extent·stride:

```
S_{k+1} = S_k · F_k        (per dim, digits sorted by stride)
```

Example (warp-interior m, per §9): `m = g + 8e` with `g∈[0,8)` stride 1, `e∈{0,1}` stride 8: `S_e = 1·8` ✓.

**The REG layer.** A tensorized warp's coordinate stack, outermost to innermost:

```
(CTA) → (WARP) → (THREAD: lane digits) → (REG: per-lane register digits) → (SEQ: warp-uniform rounds)
```

- **THREAD** holds the lane digits — the fragment's lane grid, with non-unit strides allowed.
- **REG** holds the per-lane register digits — the FRAG buffer's element shape. REG is not a loop; it is the register index of the per-lane buffer (§7.2 of the canonical doc).
- **SEQ** holds the warp-uniform instruction rounds: every lane executes each round together (this satisfies the uniform-collective-call invariant by construction).

**Per-node layout table.** The lane/reg digits are *shared per cluster*, but every operand node has its own map from the digit vector to its operand tile. The cluster's warp-interior factorization is written once, with a per-role table of maps (§4.2). Each map must be a bijection onto its tile — checked by coefficient matching (§5, C3).

---

## 3. Access Maps and the Two-Level Bijection

**Candidate filter (per memory node, pre-scheduling).** A node is an `ldmatrix` candidate iff:

1. its tier path is **SMEM → FRAG** (`ldmatrix` reads shared memory only; a GMEM→FRAG load is never a candidate — it sits behind a wmem cache by construction);
2. its dtype is in the instruction's dtype set (b16 family);
3. its subscript `σ(x) = A·x + b` is affine on the node's **own domain** D (the dims that index it), and **A is injective on the covered tile**.

Note on (3): the subscript map from the *whole* iteration space to a tensor is generally a projection (e.g. `(m,n,k) ↦ (m,k)` for the A operand) and never square. Invertibility is required of the map restricted to D — equivalently, the accessed image tile has the same shape as the iteration tile. This is exactly the project's one-owner-per-address race-freedom invariant restated: broadcasts and stride collapses are excluded by the same clause that guarantees correctness.

**The address chain.** Instruction conditions live at byte level, so the full chain is:

```
I --(σ: affine bijection on the tile)--> O --(L: buffer layout)--> byte offset
```

`L` is a **compiler-chosen degree of freedom**: a layout family (which dim is unit-stride) plus a swizzle tag (§6). The bijection of σ is what lets shape constraints be stated in I space and carried into O; L then supplies the byte-level arrangement.

**Constraint pullback.** A condition C stated in O space (tile shape, row contiguity, anchor alignment) pulls back through σ into arithmetic clauses on the factor matrix (divisibility, taboo, stride equations); conditions that are about bytes (16B alignment, bank behavior) become clauses on L (family validity, tag selection).

---

## 4. Instruction Clusters and the Spec Table

### 4.1 The cluster

The unit of constraint batching is the **instruction cluster**: one conjunction covering the loads and the compute that feeds them, because `ldmatrix`'s destination layout has no meaning apart from the mma operand it produces. For the f16 GEMM:

```
cluster(mma.m16n8k8.f32.f16.f16) = { ldA (ldmatrix), ldB (ldmatrix), mma, acc }
shared lane digits (g, t):  l = 4g + t,   g = l/4 (slow), t = l%4 (fast)
shared register digits (e, j): e, j ∈ {0, 1}
```

Scheduling loads without the compute in the same conjunction can satisfy every load clause and still be mma-incompatible; the cluster prevents that class of failure.

### 4.2 Fragment maps (m16n8k8, f16/f32) — the per-node layout table

With `g = l/4`, `t = l%4`:

| Role | Tile | Map (cell owned per lane) |
|---|---|---|
| A operand (16×8) | (m, k) | `A[g + 8e][2t + j]` |
| B operand (8×8) | (k, n) | `B[2t + j][g]` |
| C accumulator (16×8) | (m, n) | `C[g + 8e][2t + j]` |

Each map is an invertible affine map from the digit vector `(g, t, e, j)` to the operand tile; each is a bijection (verifiable by the mixed-radix condition per operand). **These tables must be pinned against the PTX ISA "Matrix Fragments" tables at implementation time** — the schema, not the constants, is the design (§10).

**ldmatrix address semantics** (the *address-supply* map, distinct from the destination map above — the instruction routes internally):

| Variant | Tiles per collective | Address lanes |
|---|---|---|
| `.x1` (8×8) | 1 | lanes 0–7 → rows 0–7 |
| `.x2` (16×8) | 2 (stacked in M) | lanes 0–15 → rows 0–15 |
| `.x4` (16×16 / 8×32 / 32×8) | 4 | lanes 0–31 → 32 rows, groups of 8 per tile |

### 4.3 The instruction spec table (data, not code)

Each instruction family is one row; adding imma (s8) later is a table row, not a pass change:

```
spec(instr) = { shapes, k-step, dtype-sets, per-role fragment maps,
                ldmatrix variants, layout families, trans-legality }
```

Bundle derivation (§5) is a *generator* over this table; the recognizer (§7) and the rewrite (§7) consult it.

---

## 5. The Two Constraint Bundles

For each recognized site, constraints split into exactly two bundles.

### 5.1 Bundle SAT — must be resolved by the scheduling solve

| # | Clause | Formal statement |
|---|---|---|
| C1 | instruction-tile divisibility | warp-region extents on the cluster's dims are multiples of the instruction tile; k innermost factor = the instruction k-step (8 for m16n8k8; a double-slab uses 16 with two mma per round) |
| C2 | warp-uniform rounds | the load/round loops have `LoC[THREAD] = 1` on the load's subscript dims (taboo clause); rounds are SEQ-class — the collective is warp-level, like today's CTA-collective `loadtile` |
| C3 | fragment equations | per role, coefficient match: the node's derived map == the spec table's fragment map |
| C4 | RSR integration | the mma's internal cross-lane K reduction covers the fragment k-pairing: no THREAD-tier combine inside the warp; the ladder starts at the WARP tier (§8) |
| C5 | anchor alignment | 16B rows, 16B-aligned anchors — automatic under exact factorization (an anchor is a multiple of the factors above it) |
| C6 | budgets | wmem sizing (canonical §7.2 hoisting rule) within the SMEM budget; FRAG within the per-thread budget |
| C7 | base scheduling | ΠF = S, MALoC, MAPM, CuMAP, whole-warp, SEQ residual derivation (canonical §5) |

### 5.2 Bundle PROACTIVE — the recipe table

Constraints that scheduling cannot resolve are matched against an **enumerated recipe table** — never an open-ended fixing step. Every recipe is a composition of the three memory verbs plus ordinary compute ops, inserted as ordinary IR and lowered by the normal pipeline:

| Recipe | Trigger | Inserted IR | Cost |
|---|---|---|---|
| `convert` | operand dtype ∉ the instruction's dtype set | cast at fill: gmem → convert → wmem | ALU + bandwidth |
| `trans` | source layout vs. operand layout mismatch | **(a)** `ldmatrix.trans` instruction variant — cost 0; **(b)** fill-time transpose (SIMT) — ALU + smem pass | 0 or ALU+smem |
| `pad` | extent not divisible by the instruction tile | zero-fill predicate at fill (the runtime loader already zero-fills out-of-range) | masked lanes + wasted instruction work |
| `convert_out` | acc dtype ≠ output dtype | cast at the epilogue store | 1 ALU op per stored element |
| `replicate` | σ not injective on the tile (broadcast access) | materialize an expanded copy into wmem | bandwidth + smem |

Two consequences of the table shape:

- Some "unmet" constraints resolve by **instruction-variant selection** at zero cost (`trans`(a)); the objective evaluator picks it automatically.
- Recipes **compose** (fp32 operands *and* col-major source → `convert` + `trans`, costs add). Recipe costs are evaluated in the objective *after* the solve — they are never SAT variables.

---

## 6. Swizzle: Derived Layout Tags (Not Searched)

Bank-conflict avoidance is **derived, not solved**. The standard swizzle is a XOR permutation between row bits and 16B-chunk bits:

```
// span-s swizzle for 16B chunks (the CUTLASS Swizzle<3,3,3> family):
chunk' = chunk ^ (row & ((1 << s) - 1))
byte_offset = row · pitch + chunk' · 16
```

**Derived span rule.** With `chunks_per_row = pitch / 16`, choose `s = min(log2(chunks_per_row), 3)`. Without the swizzle, an 8-lane group reads 8 rows at a fixed chunk with addresses `pitch` apart, colliding in banks when `pitch/16 mod 8` is small; the XOR spreads the rows across `2^s` distinct chunks. The bank-validity predicate is proven **once per (pitch, span) pair** and kept as a lookup table — it is never a per-kernel SAT variable.

**Runtime ABI.** The swizzle lives in the runtime, not in emitted code, with the compiler passing a single **layout tag**:

- `SmemView` gains a layout tag parameter (plain | `SW32` | `SW64` | `SW128`, plus the row-contiguity family).
- The **fill** (cp.async loader — which already owns copy, sync, and zero-fill) and the **read** (the `ldmatrix` wrapper) both consult the tag, applying the same XOR.
- The compiler-side clause set reduces to: *tag consistent at fill and read; tag valid for the pitch*. No XOR arithmetic is ever emitted or lowered.

Zero-fill padding composes with the swizzle for free (padding rows are written through the same permuted address function).

---

## 7. Pipeline Phases

```
P1 recognize ──▶ P2 derive bundles ──▶ P3 SAT solve ──▶ P4 per-site fork ──▶ P5 mechanical rewrite
```

**P1 — Recognition (pre-scheduling, conservative).**

- *Cache/fill sites*: the existing cache-site discovery (the tiling decide pass's `findINS`/overlap analysis) already recognizes hoistable loads before scheduling; the *same nodes* are the `ldmatrix` candidates, because every frag-load sits behind a wmem cache by construction.
- *GEMM clusters*: a `reduce<@r>` whose piped producer is a two-operand product in which `@r` appears in both operand subscripts in inner-product position (`A[m,k]·B[k,n]`, reduce over k). Batch dims commute. The recognizer is a small structural match over the effect/dataflow graph.
- **Conservatism rule**: anything unrecognized simply never nominates — zero risk. Recognition may not rewrite anything.

**P2 — Bundle derivation.** Mechanical generation of the C1–C7 clauses and the recipe-table checks from the spec table, per recognized site.

**P3 — SAT solve.** Feasibility over the preset-universe factor digits, with all sites' clauses conjoined with the base scheduling constraints. The joint instance is where the solver earns its keep over greedy enumeration (shared factors across sites: one k-innermost factor serves every cluster; shared budgets). Brute-force enumeration with incremental pruning — the existing searcher's machinery — is sufficient at these sizes. Recipes are **not** variables.

**P4 — Per-site fork (the objective).** For each site:

1. All SAT clauses met → tensorize (the mma throughput term, ~8× on this silicon, dominates; nearly always wins).
2. Clauses unmet (post-solve, recipe-checkable) → compare in the one-currency traffic/throughput model:
   - (a) keep the **SIMT** version of the site;
   - (b) **tensorize + Σ recipe costs** (recipes chosen by the table; `trans`(a) at cost 0 when legal).
   The comparison is a two-way evaluation, not a search. A **precision-policy flag** gates recipes that change numerics (`convert` f32→f16, tf32 reinterpretation): such recipes may be disallowed by policy even when profitable.

**P5 — Mechanical rewrite.** Per site, against the *solved* schedule: re-derive every equation (divisibility, taboo, fragment coefficient match, tag validity, RSR accounting), then **fail-closed on any mismatch** (same discipline as the existing tile-lower backstops) and substitute:

- the wmem fills become tagged runtime loader calls (unchanged verbs, tag added);
- the frag-loads become `ldmatrix` wrapper calls;
- the SEQ round body becomes the `mma.sync` sequence;
- all address and index expressions are already closed forms — the rewrite substitutes text, it computes nothing.

---

## 8. RSR Interaction: the Ladder Starts at the Warp Tier

`mma.sync` performs the cross-lane reduction along K **inside the instruction** (each lane holds partial K elements; the instruction combines across lanes). Consequences for the reduce model:

1. **No THREAD-tier combine exists inside a tensorized warp.** The fragment k-pairing is consumed by the instruction; the Q-2 grouped shuffle is not emitted for the reduced axis when the fragment equation holds.
2. **The ladder starts at the WARP tier**: if the warp layer shards the reduced axis (warp k-factor > 1 — split-k over warps), the cross-warp combine into the CTA-tier staging buffer is emitted exactly per the canonical §8.3, and is the top combine (RSR stops at the WARP layer).
3. **Split-k is a plain RSR consequence** — no special-cased flag: the reduced dim sharded at an outer layer is just a column of the factor matrix whose RSR coverage includes that layer.
4. Layers above the RSR stop carry `LoC = 1` on the reduced axis (Q-4 ruling), unchanged.

Clause C4 is the scheduling-side statement of points 1–2: the RSR derivation must *count* the mma's internal reduction as covering the fragment k digits, or the solver would emit a redundant (and wrong) shuffle tier.

---

## 9. Worked Example: ldmatrix + mma, End to End

### 9.1 Program

```
C[M,N] += A[M,K] · B[K,N]        M = 128, N = 128, K = 64
map((m,n,k): A[m,k] · B[k,n]) as (@m, @n, @k) | reduce<@k>(+, 0)
```
A, B: f16 row-major. C: f16. Accumulate f32. Epilogue: multiply by α.

Iteration space `S = [128, 128, 64]`.

### 9.2 Recognition (P1)

| Site | Recognition | Nomination |
|---|---|---|
| S1 | reduce<@k> over two-operand product, k in both subscripts | `cluster(mma.m16n8k8.f32.f16.f16)` = {ldA, ldB, mma, acc} |
| S2/S3 | A/B loads hoistable (cache discovery) | cp.async fill sites |
| S4 | wmem_A→FRAG, b16, σ = identity on (m,k) ✓ | ldmatrix candidate |
| S5 | wmem_B→FRAG, b16, σ = identity on (k,n) ✓ | ldmatrix candidate |

### 9.3 Bundles (P2)

**SAT bundle** — C1: warp region (m,n,k) multiples of (16, 8, ·) with k-innermost 8; C2: S4 dims (m,k) and S5 dims (k,n) taboo at THREAD; C3: fragment equations per §4.2; C4: RSR counts mma-internal lane-K; C5: automatic (exact factorization); C6: wmem_A 16×32×2B + wmem_B 32×16×2B = 2KB/warp, 8 warps → 16KB ≤ 64KB ✓; C7: base constraints.

**Proactive bundle** — dtype(A,B) ∈ {f16} ✓ met; B rows contiguous along n ✓ met (row-major); extents divisible ✓ met; acc f32 → C f16 **unmet** → recipe `convert_out` (epilogue cast, 1 ALU/elem).

### 9.4 SAT solution (P3) — one satisfying assignment

```
                 m              n              k            S = [128, 128, 64]
(0/CTA)        4              4              1            16 CTAs
(1/WARP)       2              2              2            8 warps/CTA, blockDim 256
(2/THREAD)     8   (g = l/4)  4   (t = l%4)  4   (stride 2)     ← shared lane digits
(3/REG)        2   (stride 8) 2   (pair)     2   (pair)         ← per-role stride decoration
(4/SEQ)        1              2   (n-blocks)  4   (k-slabs)      ← warp-uniform rounds
```

Column checks: m 4·2·8·2·1 = 128 ✓ · n 4·2·4·2·2 = 128 ✓ · k 1·2·4·2·4 = 64 ✓. Warp tile = 16×16×32; CTA tile = 32×32×64.

Per-role layout table (the stride decorations of the THREAD/REG block):

```
S4 (A-frag):  (m, k) = (g + 8e, 2t + j)       per-lane frag shape [2][2]
S5 (B-frag):  (k, n) = (2t + j, g)
acc (C):      (m, n) = (g + 8e, 2t + j)
```

Layout tags (derived per §6): wmem_A pitch = 64B → SW64; wmem_B pitch = 32B → SW32.

Validity spot-checks (C3, mixed-radix per dim within the warp): m: digits g (S=1, F=8), e (S=8, F=2) → 8·1 = 8 ✓; k: digits j (S=1, F=2), t (S=2, F=4) → 2·1 = 2 ✓, round stride 8 = 2·4 ✓.

### 9.5 Fork (P4) and rewrite (P5)

All SAT clauses met; recipes = {`convert_out`}. Objective: mma throughput (~8×) ≫ 1 ALU/elem conversion → **tensorize**. One warp's tile after the rewrite:

```cpp
// warp (wi, wj, wq):  m0 = 16·wi, n0 = 16·wj, k-half q
tca::rt::LoadTileWarp(wmem_A[wi], srcA.window(m0, 0, 32, 64), SW64);  // cp.async fill, tag
tca::rt::LoadTileWarp(wmem_B[wj], srcB.window(0, n0, 64, 32), SW32);

for (int r = 0; r < 4; ++r) {                 // C2: warp-uniform SEQ k-slabs
  frag_a = tca::rt::LdmatrixX2(wmem_A[wi], m0, 8*r, SW64);  // S4: 16×8, lanes 0..15 address rows
  frag_b = tca::rt::LdmatrixX1(wmem_B[wj], 8*r, n0, SW32);  // S5: 8×8
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32           // C3: maps match by construction
      {c0,c1,c2,c3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
  // second n-block (n0+8): the SEQ n-round, same slab
}
// C4: no intra-warp combine — the mma absorbed lane-K.
combine_warp(acc, staging[wi][wj]);           // cross-warp: warps q∈{0,1} own k-halves →
                                             // one combine into smem(cta); top of the ladder (RSR @ WARP)
if (owner_gate())                            // single-owner store (existing epilogue gate)
  out_v.Set(f16(acc * α), (m, n));           // recipe convert_out + epilogue scale
```

Every address is a closed form fixed by the solve; the rewrite substituted text.

### 9.6 Counterfactuals — the fork in action

1. **B stored column-major `[n][k]`.** SAT bundle unchanged; proactive: row-contiguity unmet → recipe `trans` has two instances: `ldmatrix.trans` (cost 0, legal for this operand) vs. fill-time transpose (ALU + smem pass). Objective picks **`.trans`** — an "unmet constraint" resolved at zero cost by variant selection.
2. **A, B are f32.** No f32×f32→f32 m16n8k8 exists. Fork: (a) SIMT; (b) `convert` → f16 (precision loss); (c) `convert` → tf32 reinterpretation (10-bit mantissa). The objective compares all remediated variants against SIMT; the **precision-policy flag** decides whether (b)/(c) are even admissible.
3. **M = 100.** Divisibility unmet → `pad`: the fill predicate zero-fills A-tile rows 100–111 (the loader already zero-fills out-of-range — the recipe is nearly free; cost = masked lanes + ~12% wasted mma work). Tensorized+pad wins for any reasonable shape.

---

## 10. Build Inventory and Open Items

**Already exists (carries over):** the factor-matrix/LoC constraint machinery and searcher; cache-site discovery; the RSR ladder and Q-2/Q-3/Q-4 rulings; the runtime loader (copy + sync + zero-fill); the one-currency traffic objective; fail-closed backstop discipline; per-node effect chains.

**To build:**

1. **GEMM recognizer** (small, conservative, read-only);
2. **instruction spec table** (shapes, fragment maps, dtype sets, layout families, trans-legality — as data);
3. **recipe table** with cost entries + the precision-policy flag;
4. **layout-tag ABI** (`SmemView` tag; fill/read consult; the two runtime XOR sites);
5. **SAT clause generator** over the spec table (C1–C7);
6. **the rewrite pass** (verify equations, substitute; fail-closed) + goldens per dtype/variant.

**Open items (pin before implementation):**

- Verify every fragment map constant against the PTX ISA "Matrix Fragments" tables (§4.2 is the schema; the B-fragment map in particular must be confirmed).
- Pin the swizzle constants and the (pitch, span) bank-validity table; prove each entry once.
- `.trans` legality matrix per operand role and dtype (when variant selection is legal vs. when a fill-time transpose is required).
- The `ldmatrix`-only intermediate stage: decide whether to ship it (it exercises the Q-2 grouped shuffle on k-quads) or go straight to the full cluster.
- Recognition conservatism rules: what disqualifies a site outright (non-affine σ, multi-writer tiles) vs. what routes to the recipe table.
