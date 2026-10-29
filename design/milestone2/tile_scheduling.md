# Tile Scheduling

## 1. Motivation and Goals

Tile scheduling is the mechanical procedure that transforms an iteration space into a layered, parallelized form suitable for GPU code generation. It is the bridge between the high-level TCA iteration algebra and the concrete CUDA execution model.

This document defines the canonical tile scheduling model for the TCA compiler. It covers:

- **Iteration-space factorization.** An N-dimensional iteration polytope is factored into M layers of tiles, each with a well-defined extent per dimension.
- **Compiler-owned parallelization.** Each layer is assigned a space of concurrency (CTA, warp group, warp, thread, sequential). Users never bind iterations to CUDA threads or blocks.
- **Mechanical loop-nest code generation.** The factor matrix, LoC matrix, and stride matrix fully determine the emitted loop nest structure and coordinate reconstruction.
- **Mechanical memory hierarchy mapping.** Storage class (FRAG, SMEM, TMEM, GMEM) and ownership (which SoC level owns a buffer region) are derived from the layer assignment and the hoisting rule.

**Scope.** This document describes iteration-space scheduling only. Memory/data is opaque to the scheduler; data dependencies enter as constraints (e.g. a particular dimension must be sequential within a thread). The semantics of map and reduce operations under tiling are preserved exactly, given the standard TCA invariants (immutable semantics, associative+commutative reduce combiners).

**Relationship to tca.md.** This document is the detailed expansion of tca.md §5.5 (Iteration Factoring), §5.7 (Automatic Parallelization), and §5.10 (Reduce Lowering). It is consistent with the implemented vocabulary: per-LNT absolute LoC extents, whole-warp thread groups, the three-verb memory IR (MemNew/MemLoad/MemStore + storage class).

**Relationship to tensorization.** Tensor-core instruction support (`ldmatrix` + `mma.sync`) is specified as a property of this scheduling constraint system in the companion [tile_scheduling_tensorization.md](./tile_scheduling_tensorization.md), which extends the factor matrix with stride-decorated layers and per-node layout equations.

---

## 2. Definitions

### 2.1 Iteration Space and Shape

An **iteration space** (or **polytope**) of rank N is the Cartesian product of N integer intervals:

```
S = [0, S[0]) × [0, S[1]) × ... × [0, S[N-1])
```

- `S[i]` is the **extent** of dimension i.
- A **coordinate** (or **point**) in S is an N-tuple `(c_0, c_1, ..., c_{N-1})` where `0 ≤ c_i < S[i]`.
- Total number of points: `Π_i S[i]`.

We use the word **shape** interchangeably with "extent vector" when the dimensionality is clear from context.

### 2.2 Layout Vocabulary

We borrow minimal vocabulary from layout algebra:

| Term | Meaning |
|------|---------|
| **Tuple** | An ordered, flat list. `Tuple(X)` has elements of type X. |
| **HTuple** | A hierarchical (recursive) tuple. May contain nested tuples. |
| **Congruence** (`~`) | Two HTuples are congruent when their nested structure matches element-by-element. |
| **Weak congruence** | A sub-structure match (prefix or subset of axes). |
| **Flattening** | An HTuple can be flattened to a Tuple by concatenating leaf elements in depth-first order. |

**Convention.** Throughout this document, all shapes and coordinate tuples are **flattened** Tuples, not HTuples. We work with flat rank-N vectors. Hierarchical structure arises from the layering (M layers, each with an N-vector), not from nested tuples.

### 2.3 Coordinates, Strides, and Offsets

- A **coordinate set** for a shape S is the set of all N-tuples `(c_0, ..., c_{N-1})` with `0 ≤ c_i < S[i]`.
- A **stride** is an N-tuple `(s_0, ..., s_{N-1})` of integers that maps a coordinate to a scalar **offset** via dot product: `offset = Σ_i c_i · s_i`.
  - In row-major layout, `s_i = Π_{j>i} S[j]` (the stride of dimension i is the product of extents of all dimensions inner to i).
  - The stride is the per-unit increment of the linear offset along that dimension.

---

## 3. The Tiling Operation

### 3.1 tile(S, I) → (C, B)

Given an input shape S (rank N) and a tile shape I (congruent to S, i.e. also rank N), the tiling operation produces:

- **C = number of subspaces** (subspace-index extent), where `C[i] = S[i] / I[i]` for each dimension i.
- **B = per-subspace block extent**, where `B[i] = I[i]` (this is just I under a different name, to clarify its role).

We write:

```
tile(S, I) = (C, B), where C[i] = S[i] / I[i], B[i] = I[i]
```

**Divisibility requirement.** For exact tiling, `I[i]` must divide `S[i]` for all i. When `I[i]` does not divide `S[i]`, the last subspace has a smaller extent and the iteration is guarded by a tail predicate — this is consistent with tca.md §2.4 ("When T does not divide M, the inner iteration is implicitly guarded by `@x_o + @x_i < M`").

### 3.2 Coordinate Mapping

The tiling operation defines a bijection between coordinates in S and pairs `(subspace_index, intra_subspace_coordinate)`:

- **Subspace index**: an N-tuple `idx` with `0 ≤ idx[i] < C[i]`.
- **Intra-subspace coordinate**: an N-tuple `intra` with `0 ≤ intra[i] < B[i]`.

**Forward mapping (subspace → parental):**

```
parental[i] = idx[i] * B[i] + intra[i]
```

The **offset** of a subspace is:

```
Offset[i] = idx[i] * B[i]
```

This is the starting coordinate of the subspace along dimension i.

**Inverse mapping (parental → subspace):**

```
idx[i]   = floor(parental[i] / B[i])   (subspace index)
intra[i] = parental[i] % B[i]          (intra-subspace coordinate)
```

**Bijection proof (one line).** Since `C[i] * B[i] = S[i]` for exact tiling, each parental coordinate `0 ≤ p < S[i]` has a unique `idx = p // B[i]` in `[0, C[i])` and a unique `intra = p % B[i]` in `[0, B[i])`. The forward map inverts the inverse map exactly.

### 3.3 Worked Example

Let `S = [100, 200, 50]` and `I = [2, 2, 10]`.

- `C = [100/2, 200/2, 50/10] = [50, 100, 5]` (50 subspaces along dim 0, 100 along dim 1, 5 along dim 2).
- `B = [2, 2, 10]` (each subspace is 2×2×10).

For subspace index `(1, 0, 2)`:

- Offset = `(1*2, 0*2, 2*10) = (2, 0, 20)`.
- Parental coordinate of point `(1, 1, 3)` inside this subspace = `(2+1, 0+1, 20+3) = (3, 1, 23)`.

Inverse: parental coordinate `(3, 1, 23)`:

- idx = `(3//2, 1//2, 23//10) = (1, 0, 2)`.
- intra = `(3%2, 1%2, 23%10) = (1, 1, 3)`.

---

## 4. Multi-Layer Factorization

### 4.1 Factor Matrix

A **M-layer factorization** of shape S is an M×N matrix F (M rows, N columns) such that for every dimension i:

```
Π_{m=0}^{M-1} F[m][i] = S[i]
```

- `F[m][i]` is the **factor extent** of layer m along dimension i.
- Layer 0 is the outermost layer; layer M−1 is the innermost (sequential) layer.
- The product of all factor extents in a column equals the full extent of that dimension.

This is the column-factorization invariant: each column of F is a multiplicative partition of the corresponding dimension of S.

### 4.2 Per-Unit Region Shape

At each layer m, one **unit** of that layer (e.g. one CTA, one warp, one thread) owns a region of the iteration space whose shape is the product of all factor extents in layers strictly below m:

```
R[m][i] = Π_{r=m+1}^{M-1} F[r][i]
```

`R[m]` is the **per-unit region shape** at layer m. It answers: "how big is the piece of the iteration space owned by one unit at layer m?"

- `R[M-1][i] = 1` for all i (the innermost unit is a single point along every dimension — it is one iteration point).
- `R[-1][i] = S[i]` (virtual row: the "whole space" before any layering).

### 4.3 Stride Matrix and Stride Per Layer

The **stride matrix** is an M×N matrix whose m-th row equals the per-unit region shape at layer m:

```
stride[m][i] = R[m][i] = Π_{r>m} F[r][i]
```

This is the stride (offset per unit) along dimension i for the layer-m index.

**Per-layer diagonal stride matrix.** For algebraic convenience, we also define SM(m) as the N×N diagonal matrix whose i-th diagonal entry is `stride[m][i]`:

```
SM(m) = diag(stride[m][0], stride[m][1], ..., stride[m][N-1])
```

The M×N stride matrix is just the stack of diagonals of SM(0) through SM(M−1).

### 4.4 Global Coordinate Reconstruction

Given per-layer coordinate vectors `c_m` (each an N-tuple with `0 ≤ c_m[i] < F[m][i]`), the **global coordinate** in S is:

```
g[i] = Σ_{m=0}^{M-1} c_m[i] * stride[m][i]
```

In matrix form: `g = Σ_m SM(m) · c_m`.

**Bijection.** The mapping from the M×N tiled coordinate matrix `(c_0, c_1, ..., c_{M-1})` to the N global coordinates is a bijection. Proof: by induction on M using the single-layer bijection (§3.2). Each layer's coordinate selects a sub-region; the innermost layer selects a single point.

### 4.5 The SEQ Layer

The innermost layer (layer M−1) is always the **SEQ** (sequential) layer. It represents iterations that execute sequentially within a single thread. The factor extents `F[M-1][i]` are the per-thread sequential trip counts along each dimension.

---

## 5. Scheduling: SoC Assignment and LoC

### 5.1 Space of Concurrency (SoC)

Each layer m is assigned a **Space of Concurrency** (SoC) from the list:

```
CTA → WARP GROUP (optional) → WARP → THREAD → SEQ
```

- Outer-to-inner order: layer 0 (outermost) gets the highest-level SoC, layer M−1 (innermost) gets SEQ.
- Assignment is in priority order: we assign the highest-priority available SoC to the outermost layer and work inward.
- **WARP GROUP** is optional hardware support; when absent, it is skipped.

**Why this order?** It matches the CUDA execution hierarchy and ensures that parallelism is maximized at the highest available level before moving inward.

### 5.2 LoC Matrix

The **Level of Concurrency (LoC) matrix** L is an M×N matrix where `L[m][i]` is the number of parallel units along dimension i at layer m.

Constraints:

1. **Quantity constraint:** `1 ≤ L[m][i] ≤ F[m][i]` for all m, i.
   - LoC cannot exceed the factor extent at that layer.
   - For m = 0, the "parent extent" is `S[i]`; equivalently the constraint is `L[0][i] ≤ F[0][i]` where F[0] is the first factor layer.

2. **Hardware max constraint:** For layer m with SoC X:
   ```
   Π_i L[m][i] ≤ MaxLoC(X)
   ```
   The total number of parallel units at layer m must not exceed the hardware maximum for that SoC level.

3. **SEQ constraint:** `L[M-1][i] = 1` for all i. The sequential layer has no parallelism (all work is serial).

4. **Alignment constraints:** For warp-level and thread-level layers, the total number of threads must be a multiple of 32 (whole-warp thread groups — project ruling). Specifically:
   - For THREAD layer: `Π_i L[THREAD][i]` must be a multiple of 32 (so threads form complete warps).
   - For WARP layer: `Π_i L[WARP][i]` is in warp units (each "unit" is one warp), so the total thread count is `Π_i L[WARP][i] * 32`.
   - For WARP GROUP layer: `Π_i L[WARP_GROUP][i]` is in warpgroup units; total warps = `Π_i L[WARP_GROUP][i] * 4`.

5. **Taboo dimensions:** A dimension i is "taboo" at layer m if `L[m][i] = 1` (no parallelism along that dimension at that layer). Taboo dimensions are always allowed — they mean the dimension is not parallelized at this layer.

### 5.3 MaxLoC Values and Provenance

| SoC | MaxLoC | Provenance |
|-----|--------|------------|
| CTA | 2^31 − 1 | CUDA grid size limit (effectively unbounded for practical purposes) |
| WARP GROUP | 8 warpgroups / CTA | 32 warps per CTA ÷ 4 warps per warpgroup = 8 |
| WARP | 32 warps / CTA | CUDA maximum: 1024 threads per CTA ÷ 32 threads per warp = 32 |
| THREAD | 32 threads / warp | CUDA warp size = 32 |
| SEQ | 1 | By definition |

**Notes:**
- Warp group (WARP GROUP) is optional hardware; MaxLoC of 8 is the architectural maximum on sm_86.
- The whole-warp ruling: every thread group is a whole number of physical warps. This means LoC at the THREAD level is always a multiple of 32, and the THREAD layer's factor extent covers the same number of threads as the WARP layer's factor extent × 32.
- The CTA-level maximum (2^31−1) is the CUDA grid dimension limit; in practice the per-CTA resource limits (registers, shared memory) are the binding constraint.

### 5.4 Factor = LoC on Parallel Layers (Ruled)

On all non-SEQ layers, the canonical form is:

```
F[m][i] = L[m][i]  for all m < M-1 (non-SEQ layers)
F[M-1][i] = S[i] / ( Π_{m=0}^{M-2} L[m][i] )   (SEQ layer, derived)
```

**Ruling (author, 2026-10-28).** A parallel layer's factor extent is exactly its LoC — there is no per-unit sequential slack at CTA/WARP/THREAD layers. If a dimension's LoC is 1 at some layer, that dimension simply **does not participate** in that layer's SoC assignment. The SEQ layer's factor extent is then *derived*, not chosen: it is the residual left over after all parallel layers have taken their shares,

```
F[M-1][i] = S[i] / ( Π_{m<M-1} L[m][i] )
```

so a dimension that no parallel layer picks up (all LoCs are 1) is absorbed entirely by the SEQ layer, and the column-product invariant `Π_m F[m][i] = S[i]` is enforced automatically by the constraint system. The quantity constraint `1 ≤ L[m][i] ≤ F[m][i]` guarantees the residual is ≥ 1.

Consequences:

- The scheduler searches over LoC extents on parallel layers only; the factor matrix is identical to the LoC matrix there, and the SEQ row is computed.
- Only the SEQ layer emits sequential loops; CTA/WARP/THREAD layers are fully parallel.
- Non-participation is expressed as LoC = 1 ("taboo" dimension), never as factor > LoC.

---

## 6. Mechanical Loop-Nest Code Generation

### 6.1 Loop Nest Cluster (LNC)

Each layer m generates a **loop nest cluster (LNC)**: N nested loops, one per dimension. The outer-to-inner order of dimensions follows the iteration-space dimension order (dimension 0 outermost, dimension N−1 innermost), consistent with tca.md §2.7 (as-clause defines loop order).

Layer 0 (outermost) → outermost LNC.
Layer M−1 (innermost) → innermost LNC.

### 6.2 Loop Trip Counts

The loop at layer m, dimension i has trip count `F[m][i]`.

- **SEQ layer:** emits a standard `for` loop with induction variable from 0 to F[M−1][i]−1, step 1.
- **CTA layer:** loops are materialized as `blockIdx` acquisitions. The CTA LNC does not emit literal `for` loops; instead, the block index directly provides the coordinate along each CTA-parallel dimension.
- **WARP layer:** warp index is acquired from runtime maps (e.g. `thread_map.warp_index()`).
- **THREAD layer:** thread/lane index is acquired from runtime maps (e.g. `thread_map.lane_id()`).

**Normalization.** Every loop has init = 0 and step = 1. "Normalized trip count" means the trip count equals the extent (counting from 0 to extent−1 inclusive is extent iterations). There is no separate "normalization" step — the coordinate is just the raw loop index or hardware id.

### 6.3 Per-Layer Offset Accumulation

At layer m, the per-unit offset contribution along dimension i is:

```
offset_m[i] = c_m[i] * stride[m][i]
```

where `c_m[i]` is the coordinate at layer m, dimension i (from the loop index or hardware id).

The **global coordinate** at any point in the nest is the sum of offset contributions from all layers:

```
global[i] = Σ_{m=0}^{M-1} offset_m[i] = Σ_{m=0}^{M-1} c_m[i] * stride[m][i]
```

### 6.4 Relation to tca.md §5.5 (Iteration Factoring)

The LNC structure is the codegen realization of tca §5.5 iteration factoring. The factorization described here produces the per-LNT loop structure that the final lowering pass expands into concrete CUDA code. The whole-warp thread-group model (tca §5.7) is reflected in the alignment constraint on THREAD and WARP layers.

---

## 7. Memory Hierarchy Mapping

### 7.1 Storage Classes and Owners

Memory buffers have a **storage class** and an **owning SoC**:

| Storage Class | Owner SoC | Meaning |
|---------------|-----------|---------|
| FRAG | SEQ / thread | Register file / per-thread fragment; private to each thread. |
| SMEM | thread | Shared memory region with per-thread ownership (one slot per thread); thread writes its own slot. |
| SMEM | warp | Shared memory region with per-warp ownership (one slot per warp); warp writes its own slot. |
| SMEM | CTA | Shared memory region with CTA-level ownership; collectively accessed by all threads in a CTA. |
| TMEM | CTA | Tensor memory (not used in v0; reserved for future hardware). |
| GMEM | CTA | Global memory; accessible to all CTAs but owned per-CTA for output buffers. |

**Key principle:** storage class + owner together define where data lives and who can write it. This is consistent with the project's memory IR model (three verbs + storage class).

### 7.1.1 The wmem Convention (Ruled)

A warp-owned SMEM region is denoted **wmem** as a conceptual convenience. Physically, wmem is ordinary shared memory **replicated once per runtime warp** and each replication is assigned to exactly one warp of the CTA.

Example: a CTA with warp allocation `[2][2]` (2×2 = 4 warps per CTA in the two sharded dimensions) replicates each wmem buffer 4 times; runtime warp w accesses only its own replication. The convention exists because the *per-warp-private* view makes reducer reasoning easier (a warp reads and writes "its" wmem with no cross-warp interference, exactly like a thread and its fragment); it does not introduce a new storage class. When the warp tier is the destination of a cross-thread combine (§8.3), one replication is written per warp and the cross-warp combine (§8.3 step 3) reads across replications.

### 7.2 Hoisting Rule

When a per-unit buffer at layer m+1 is hoisted (promoted) to layer m (i.e., allocated at the higher level and reused across iterations of the lower level), its shape becomes:

```
shape_hoisted[m] = (owner-coordinate extents at m) ++ (inner_buffer_shape)
```

- `owner-coordinate extents at m` = the LoC extents of all dimensions parallel at level m that index the buffer.
- `inner_buffer_shape` = the buffer's shape before hoisting.
- `++` denotes concatenation of shape dimensions.

Intuitively: the buffer has one copy per owner at the higher level, and each copy has the same shape as the original inner buffer.

### 7.3 Axis Elision Under Reduce

When a buffer is associated with a reduce operation, reduced dimensions are elided from the buffer shape at layers above the RSR stop layer. The buffer only carries the surviving (non-reduced) dimensions plus the ownership dimensions from the hoisting rule. This is formalized in §8 (Reduce Scheduling).

### 7.4 Subscript Rewriting

When a memory access is hoisted across a loop cluster (i.e., the buffer moves up a layer), its subscripts must be rewritten. The §3 parental↔subspace coordinate map formalizes this:

- **Before hoisting** (at inner layer): access uses intra-subspace coordinates plus the subspace offset.
- **After hoisting** (at outer layer): the buffer is indexed by the outer-layer coordinate (which selects the subspace), and the inner loop reads from the subspace-local slice.

This is the formal statement of cache-insertion subscript rewriting: hoisting a load to level m means the buffer at level m is indexed by the level-m coordinate, and the inner access becomes a slice of that buffer.

---

## 8. Reduce Scheduling

### 8.1 Reduce Semantics (Tie to tca.md)

A `reduce` operation collapses the last dimension of the producing map's `as` order (tca.md §2.5). The combiner must be associative and commutative. For v0, each map has at most one reduce chained after it, and it reduces exactly one axis (the last in `as` order). Multiple reductions appear as chained reduce operations.

### 8.2 Reduce Stop Rule (RSR)

**Single reduced axis (ruled).** A reduce always applies to a **single dimension** of the input polytope. Conceptually, that dimension is partitioned across SoC spaces — a portion runs sequentially inside one thread, groups of the portion across the fine-grained thread shards of a warp, further groups across warps, and so on — a layered, hierarchical partition of the one axis. Multiple reduced dimensions appear only as chained reduce operations, each with its own RSR.

Given the reduced axis k (the last dimension), the **Reduce Stop Rule (RSR)** determines which layers participate in the reduction hierarchy.

**Formal statement.** Scan layers inner-to-outer (from M−1 down to 0). Maintain an accumulated product `acc = 1`. For each layer m:

1. Multiply: `acc = acc * F[m][k]`
2. If `acc == S[k]`, stop. The set R = {M−1, M−2, ..., m} is the set of layers covered by the reducer.
3. If `acc > S[k]`, the factorization is invalid for this reduce (should not happen if the column product is exactly S[k]).

**Direction:** inner-to-outer. The innermost layer (SEQ) is always included. The outermost included layer is the first one (going outward) whose factor completes coverage of the reduced axis: at that layer the axis is fully exhausted/visited and no further tier of reduction is needed.

**Unvisited-layer constraint (ruled).** Any layer *not* in R (i.e., layers outer than the RSR stop layer) must have `L[m][k] = 1` for the reduced dimension k — the reduced axis is not parallelized above the stop layer. Combined with the F = LoC canonicalization (§5.4), this also forces `F[m][k] = 1` there, so the reduced axis is entirely covered by the R layers.

**R is ordered outer-to-inner** for the purpose of describing the reducer hierarchy: R = {r_0, r_1, ..., r_{|R|−1}} with r_0 the outermost covered layer and r_{|R|−1} = M−1 the innermost (SEQ).

### 8.3 Reducer Lowering Sequence

A reducer over reduced axis k with coverage R lowers to a ladder of operations moving up the SoC hierarchy:

1. **Fragment accumulator (SEQ layer):** each thread accumulates into a register (FRAG) during its sequential loop. This is the innermost Reduce operation.
2. **Per-thread combine (THREAD → WARP):** partial results of the threads inside one warp are combined.
   - **Mechanism (ruled).** Because tile scheduling allows *fine-grained sharding on different axes* for the threads of a single warp, the traditional single-axis full-warp shuffle reduction is not applicable. Instead, the runtime groups the lanes of a warp by layout modular arithmetic on the reduced axis and performs a **per-group** reduction with `__shfl_xor` — this is the *per-thread combine*. The combined value of each group is then written into the tier above: the per-warp accumulated storage (wmem, §7.1.1).
   - **Grouping rule.** With the warp's thread layout `L[THREAD]` and reduced axis k last in the thread sharding order, a group is the set of lanes sharing all thread coordinates except the k-shard: group member id = `lane_id % L[THREAD][k]`, group id = `lane_id / L[THREAD][k]`. Example: warp thread sharding `[2][4][4]`, reduce along the last dimension → 8 groups of 4 threads; the group members are lanes with equal `lane_id / 4`, member within group = `lane_id % 4`; the warp-level accumulated storage has shape `[2][4]` (the surviving thread-shard axes), one slot per group.
   - This step is register-file-resident (shuffle path); the write into wmem is the first SMEM traffic of the ladder.
3. **Cross-warp combine (WARP → CTA):** warps within a CTA combine their partial results.
   - The destination is a CTA-owned SMEM region.
4. **Cross-CTA combine (CTA → GMEM):** (optional, only if R includes CTA and multi-CTA reduction is needed).
   - The destination is a GMEM buffer.

**General rule (frag → wmem/smem(warp) → smem(cta) → gmem):** each Combine step moves the accumulator from the inner tier's storage to the outer tier's storage. The per-thread combine uses the grouped shuffle path (register file); all other combines are SMEM load/store steps. Conceptually, reduction is a *tiered reduction over the SoC/memory hierarchy layers*, not an opaque runtime object: the reducer is just a list of reduction operations moving the reduced result from FRAG to SMEM at successively coarser SoC levels.

### 8.3.1 Reducer.Init Semantics (Ruled)

`reducer.Init` at tier X **allocates that tier's intermediate reduction memory and initializes it with the reducer's identity value** (e.g. 0 for sum, −∞ for max). Each SoC category in the ladder has its own intermediate storage whose purpose is to receive the combined values of the tier below, readying them for the next tier's reduction — the ladder is climbed tier by tier.

The ladder terminates at the RSR stop layer (§8.2): no combine is emitted above it. A tier above the RSR stop layer (such as the `reducer<cta>` in the legacy cg sample, where RSR stops at WARP) does not perform a reduction — its initialized memory is the **output staging** that the top combine's final values are deposited into, from which the epilogue writes out. Initializing it with the identity value is what makes the deposit well-defined when some upper-tier slices receive no contribution (e.g. tail tiles).

**Associativity/commutativity requirement.** The combiner must be associative and commutative for the parallel reduction to be correct. This is restated from tca.md §2.5.

### 8.4 Map/Reduce Invariance Under Tiling

- **Map:** `Map(I, λ) = Map(I ~ M, λ)` — tiling is a bijection on the iteration space, and map is unordered element-wise application.
- **Reduce:** `Reduce(I, λ) = Reduce(I ~ M, λ)` — the result is identical when the tiling forms an exact partition of the reduced axis and the RSR defines the correct layer sequence. The associative+commutative combiner ensures that grouping by tiles does not change the result (up to floating-point rounding).

---

## 9. Worked Example: Matmul [128] × [128] × [16]

This section carries the running example through every concept defined above.

### 9.1 Setup

**Problem.** Matrix multiplication `C[M][N] += A[M][K] * B[K][N]`, with M=128, N=128, K=16.

- Iteration space shape: `S = [128, 128, 16]` (dim 0 = M, dim 1 = N, dim 2 = K).
- K is the reduced axis (last dimension in the map's `as` order for the inner product).
- 4 layers: CTA (layer 0), WARP (layer 1), THREAD (layer 2), SEQ (layer 3).

### 9.2 Factor Matrix F

Per §5.4, F = L on the non-SEQ layers and the SEQ row is the derived residual:

```
          M     N     K
(0/CTA)  [ 4 ][ 4 ][ 1 ]    F[0] = [4, 4, 1]
(1/WARP) [ 2 ][ 2 ][ 2 ]    F[1] = [2, 2, 2]
(2/THR)  [ 4 ][ 4 ][ 2 ]    F[2] = [4, 4, 2]
(3/SEQ)  [ 4 ][ 2 ][ 4 ]    F[3] = [4, 2, 4]   (derived: S[i] / Π_{m<3} L[m][i])
```

**Verification of column-product invariant:**
- M: 4 × 2 × 4 × 4 = 128 ✓
- N: 4 × 2 × 4 × 2 = 128 ✓
- K: 1 × 2 × 2 × 4 = 16 ✓

**Verification of the derived SEQ row:**
- M: 128 / (4 × 2 × 4) = 128 / 32 = 4 ✓
- N: 128 / (4 × 2 × 4) = 128 / 32 = 2 ✓
- K: 16 / (1 × 2 × 2) = 16 / 4 = 4 ✓

### 9.3 LoC Matrix L

Per §5.4, L = F for layers 0–2; L[3] = 1 by the SEQ constraint.

```
          M     N     K
(0/CTA)  [ 4 ][ 4 ][ 1 ]    total: 4*4*1 = 16 CTAs
(1/WARP) [ 2 ][ 2 ][ 2 ]    total: 2*2*2 = 8 warps / CTA
(2/THR)  [ 4 ][ 4 ][ 2 ]    total: 4*4*2 = 32 threads / warp
(3/SEQ)  [ 4 ][ 2 ][ 4 ]    sequential: 4*2*4 = 32 iters / thread
```

**Hardware constraint check:**
- CTA: 16 ≤ 2^31−1 ✓
- WARP: 8 ≤ 32 ✓
- THREAD: 32 ≤ 32 (whole warp) ✓

### 9.4 Per-Unit Region Shape / Stride Matrix

`R[m][i] = product of F[r][i] for r > m`

```
          M          N          K
(0/CTA)  [2*4*4=32][2*4*2=16][2*2*4=16]   stride[0] = [32, 16, 16]
(1/WARP) [  4*4=16][  4*2=8 ][  2*4=8 ]   stride[1] = [16, 8,  8 ]
(2/THR)  [    4   ][    2   ][    4   ]   stride[2] = [4,  2,  4 ]
(3/SEQ)  [    1   ][    1   ][    1   ]   stride[3] = [1,  1,  1 ]
```

**Note:** this corrects the cg sample's row 1 stride (which printed 16/16/16 — see F-3 in doc/review/tile_scheduling.md) to the correct values shown here. The source cg sample's column 1 (N) used different factor values; the numbers above are consistent with F = [4,4,1] / [2,2,2] / [4,4,2] / [4,2,4].

### 9.5 Global Coordinate Reconstruction

For a concrete point with per-layer coordinates:
- CTA: `(1, 2, 0)`
- WARP: `(0, 1, 1)`
- THR: `(2, 0, 0)`
- SEQ: `(3, 1, 2)`

Global coordinate = Σ c_m[i] * stride[m][i]:

- M: 1·32 + 0·16 + 2·4 + 3·1 = 32 + 0 + 8 + 3 = 43
- N: 2·16 + 1·8 + 0·2 + 1·1 = 32 + 8 + 0 + 1 = 41
- K: 0·16 + 1·8 + 0·4 + 2·1 = 0 + 8 + 0 + 2 = 10

Verification: global (43, 41, 10) should be within bounds:
- 0 ≤ 43 < 128 ✓
- 0 ≤ 41 < 128 ✓
- 0 ≤ 10 < 16 ✓

### 9.6 Loop Nest Sketch (Pseudo-Code)

The following is illustrative pseudo-code showing the structure. Variable names and bounds are consistent with the matrices above.

```cuda
// Layer 0: CTA — indices from blockIdx
int cta_0 = blockIdx.x;  // [0, 4)
int cta_1 = blockIdx.y;  // [0, 4)
// (cta_2 = 0 always, since F[0][2] = 1 — no loop)

// Layer 1: WARP — index from runtime map
int warp_0 = warp_id / 4;   // [0, 2)  (warps arranged in 2×2×2, first dim)
int warp_1 = (warp_id / 2) % 2;  // [0, 2)
int warp_2 = warp_id % 2;   // [0, 2)

// Layer 2: THREAD — index from runtime map
int thr_0 = lane_id / 8;    // [0, 4)  (32 threads in 4×4×2 layout)
int thr_1 = (lane_id / 2) % 4;  // [0, 4)
int thr_2 = lane_id % 2;    // [0, 2)

// SMEM buffers (CTA-owned)
__shared__ float lhs_smem[32][16];  // per-CTA: R[0] on M×N dims
__shared__ float rhs_smem[16][16];  // per-CTA

// Reducer: CTA-tier output staging (§8.3.1) — RSR stops at WARP (§9.8),
// so the CTA-tier memory is initialized with the identity value and receives
// the top (cross-warp) combine's final values; the epilogue writes it out.
// Per-CTA output region = R[0]_M × R[0]_N = 32 × 16.

// Per-thread fragment accumulator: surviving SEQ dims = F[3][M] × F[3][N] = [4][2]
float acc_frag[4][2];

// Layer 3: SEQ — actual sequential loops
for (int seq_0 = 0; seq_0 < 4; seq_0++) {
  for (int seq_1 = 0; seq_1 < 2; seq_1++) {
    float sum = 0.0f;
    for (int seq_2 = 0; seq_2 < 4; seq_2++) {
      // global coordinate reconstruction
      int m = cta_0 * 32 + warp_0 * 16 + thr_0 * 4 + seq_0;
      int n = cta_1 * 16 + warp_1 * 8 + thr_1 * 2 + seq_1;
      int k = cta_2 * 16 + warp_2 * 8 + thr_2 * 4 + seq_2;

      float a = lhs_smem[(warp_0*16 + thr_0*4 + seq_0)][(warp_2*8 + thr_2*4 + seq_2)];
      float b = rhs_smem[(warp_2*8 + thr_2*4 + seq_2)][(warp_1*8 + thr_1*2 + seq_1)];
      sum += a * b;
    }
    acc_frag[seq_0][seq_1] = sum;
  }
}

// Per-thread combine (§8.3 step 2): grouped shuffle reduction.
// Thread sharding is [4][4][2]; K is the reduced axis and is last in the
// thread shard order, so each group is the pair of lanes sharing thr_0/thr_1:
//   group member id = lane_id % 2,  group id = lane_id / 2
// __shfl_xor(mask=0xffffffff, value, laneMask=1) reduces within each pair.
// Each group then writes its combined value into its warp's wmem slot.
reducer_combine_thread(acc_frag, thr_0, thr_1, thr_2);

// Cross-warp combine (§8.3 step 3): the top combine — warps along the K
// shard (warp_2) combine their wmem partials into the CTA-tier staging
// buffer. RSR stops here (§9.8): no combine is emitted above this tier.
reducer_combine_warp(acc_warp, warp_0, warp_1, warp_2);

// Write out CTA-owned output region
// (tail predicate for non-divisible extents — not shown)
```

### 9.7 Hoisted Cache Shapes

**Hoisting rule restated for this example.** A buffer used by the units of layer m+1, when hoisted to layer m, gains the layer-m owner coordinates as leading dimensions: one slot per layer-m unit, each slot keeping the inner buffer shape (§7.2). Concretely, hoisting from layer m+1 to m prepends the *surviving* layer-m LoC extents along the dimensions that index the buffer (reduced-axis coordinates are elided above the RSR stop layer — §7.3, §9.8).

For the LHS operand cache (indexed by dims M, K), hoisted tier by tier:

- **FRAG (per thread, innermost):** the tile each thread touches in its SEQ loop = `F[3][M] × F[3][K]` = `[4][4]`. (K is not yet fully reduced inside the thread — the thread iterates seq_2 over K — so K still indexes the operand tile.)
- **Hoisted to WARP (wmem, one copy per warp):** one copy per warp covering its threads' tiles = warp's region along (M, K) = `R[1][M] × R[1][K]` = `[16][8]`. Note the hoist skips a *slot-per-thread* intermediate: the operand is shared read-only, so one warp-level copy serves all 32 lanes.
- **Hoisted to CTA (SMEM, CTA-owned):** the CTA's region along (M, K) = `R[0][M] × R[0][K]` = `[32][16]` — this is the `lhs_smem[32][16]` of the loop-nest sketch (§9.6).

Symmetrically, the RHS operand cache (indexed by K, N) has CTA shape `R[0][K] × R[0][N]` = `[16][16]`.

The accumulator ladder is a different composition — it follows §8.3 and is worked out in §9.9.

### 9.8 RSR and Reduce Sequence

Reduced axis: K (dim 2), extent S[2] = 16.

RSR scan (inner to outer, layers 3→2→1→0):
- Layer 3 (SEQ): acc = 1 × 4 = 4 < 16 → continue
- Layer 2 (THR): acc = 4 × 2 = 8 < 16 → continue
- Layer 1 (WARP): acc = 8 × 2 = 16 == 16 → STOP

Coverage set R = {1, 2, 3} (WARP, THREAD, SEQ). RSR stops at layer 1 (WARP). Layer 0 (CTA) is NOT covered.

**Reducer sequence (inner to outer):**
1. **SEQ (FRAG):** per-thread fragment accumulation — Reduce per element (§8.3 step 1).
2. **THREAD tier (per-thread combine):** grouped `__shfl_xor` reduction inside the warp (§8.3 step 2), each group writing its combined value into its slot of the per-warp accumulator (wmem).
3. **WARP tier (cross-warp combine — the top combine):** the two warps along the K shard (warp_2 ∈ {0, 1}) combine their wmem partials into the CTA-tier staging buffer. RSR stops here; no combine is emitted above this tier.

**Unvisited-layer constraint check:** layer 0 (CTA) is outside R; its reduced-axis entries are L[0][K] = F[0][K] = 1 ✓ (§8.2) — K is not parallelized above the stop layer.

**The `reducer<cta>` of the legacy cg sample** is the CTA-tier staging buffer of §8.3.1: initialized with the reducer's identity value, receives the top combine's final values, and is written out by the epilogue. It is not a combine tier — no cross-CTA reduction exists when RSR stops at WARP.

### 9.9 Accumulator Ladder Shapes with RSR Axis Elision

With K as the reduced axis and RSR stopping at WARP (layer 1), the accumulator at each tier carries only surviving (non-reduced) dimensions:

- **FRAG (per thread):** `F[3][M] × F[3][N] = [4][2]`. K does not appear — it is consumed by the sequential accumulation over seq_2.
- **wmem (per warp):** indexed by (group coordinates) ++ (per-thread accumulator shape), where a group is the set of lanes sharing all thread shards except the reduced K shard (thr_2): `[F[2][M], F[2][N]] ++ [4, 2] = [4][4][4][2]`. Each of the 4×4 = 16 groups deposits one `[4][2]` combined partial. This buffer is replicated per warp (§7.1.1) — 8 replications for the 2×2×2 = 8 warps of this CTA.
- **CTA-tier staging (SMEM, CTA-owned):** after the cross-warp combine across warp_2, indexed by `(warp_0, warp_1, thr_0, thr_1, seq_M, seq_N)` = `[2][2][4][4][4][2]`, i.e. exactly the per-CTA output region `[32][16]` on the surviving M×N dims (2·4·4 = 32 on M, 2·4·2 = 16 on N ✓, matching R[0] restricted to (M, N)). The warp_2 coordinate is elided: combining across it is what finishes the reduced axis.

---

## 10. Resolved Rulings (Author Decisions, 2026-10-28)

The five open questions raised by the design review (doc/review/tile_scheduling.md) were ruled by the author. The rulings are normative; the provisional markers previously used in this document have been removed. Each entry records the question, the ruling, and where it is formalized.

### Q-1: Must factor extent equal LoC on every non-SEQ layer? — RESOLVED

**Ruling.** Yes, by construction. A LoC of 1 on a dimension at some layer means the dimension does not participate in that layer's SoC assignment; the constraint system requires each dimension to be a full factorization of the input shape's extent, so any dimension no parallel layer picks up is absorbed by the SEQ layer automatically. Formalized in §5.4 (F = L on non-SEQ layers; SEQ row derived as the residual `S[i] / Π_{m<M-1} L[m][i]`).

### Q-2: Cross-thread collection: shuffle vs SMEM(thread) region — RESOLVED

**Ruling.** The per-thread (intra-warp) combine is shuffle-based: because threads within a warp are fine-grain sharded across different axes, the runtime forms lane groups by layout modular arithmetic and performs a per-group `__shfl_xor` reduction; each group then writes its combined value into the per-warp accumulated storage (wmem) one tier above. The wmem tier is physically SMEM replicated per warp (§7.1.1). No SMEM is consumed by the intra-warp combine itself. Formalized in §8.3 step 2 and §9.8–§9.9.

### Q-3: Role of reducer.Init at layers above the RSR stop layer — RESOLVED

**Ruling.** `reducer.Init` at tier X allocates that tier's intermediate reduction memory and initializes it with the reducer's identity value. Each SoC tier in the ladder owns an intermediate storage that receives the tier below's combined values for the next tier's reduction; the ladder climbs until the RSR stop layer, above which no combine is emitted — the next tier's initialized memory stages the output that the epilogue writes out. Formalized in §8.3.1.

### Q-4: RSR generalization to multiple reduced dimensions — RESOLVED

**Ruling.** Reduce always applies to a single dimension of the input polytope; multiple reductions are chained operations. RSR is per-axis: scan the tile-matrix column inner-to-outer, stop at the layer where the product of visited factors equals the input extent of that dimension; every *unvisited* (outer) layer must have LoC = 1 on the reduced dimension. Formalized in §8.2.

### Q-5: Per-unit region shape vs stride — one quantity or two? — RESOLVED

**Ruling.** One quantity: the stride row equals the per-unit region shape (§4.2–§4.3); the legacy cg sample's inconsistent middle column is erroneous (finding F-3 in the review doc). The "unit" itself may be coarse-grained: the atomic unit of a layer's LoC is normally a thread, but a layer may instead use a warp unit or warpgroup unit — expressed through the alignment/modularity constraint (whole-warp/whole-warpgroup multiples), so different tile layers may carry different unit granularities while keeping the single region/stride quantity.
