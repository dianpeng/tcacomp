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

### 5.4 Provisional: Factor = LoC on Parallel Layers

**(provisional, see Q-1)** On all non-SEQ layers, we use the canonical form:

```
L[m][i] = F[m][i]  for all m < M-1 (non-SEQ layers)
```

This means all per-unit sequential slack is pushed into the single SEQ layer. Parallel layers are "full": every factor extent is also the LoC extent. Each unit at layer m processes exactly `R[m]` worth of iterations, all sequentially inside the unit.

Under this reading:
- The schedule search picks LoC extents on parallel layers; the factor matrix is identical to the LoC matrix on those layers.
- Only the SEQ layer has sequential loops; CTA/WARP/THREAD layers are fully parallel.
- This is the simpler model, and all worked examples in the source docs follow it.

The open question Q-1 asks whether this is a required canonicalization or just the common case.

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

Given a reduced axis k (the last dimension), the **Reduce Stop Rule (RSR)** determines which layers participate in the reduction hierarchy.

**Formal statement.** Scan layers inner-to-outer (from M−1 down to 0). Maintain an accumulated product `acc = 1`. For each layer m:

1. Multiply: `acc = acc * F[m][k]`
2. If `acc == S[k]`, stop. The set R = {M−1, M−2, ..., m} is the set of layers covered by the reducer.
3. If `acc > S[k]`, the factorization is invalid for this reduce (should not happen if the column product is exactly S[k]).

**Direction:** inner-to-outer. The innermost layer (SEQ) is always included. The outermost included layer is the first one (going outward) whose factor completes coverage of the reduced axis.

**R is ordered outer-to-inner** for the purpose of describing the reducer hierarchy: R = {r_0, r_1, ..., r_{|R|−1}} with r_0 the outermost covered layer and r_{|R|−1} = M−1 the innermost (SEQ).

### 8.3 Reducer Lowering Sequence

A reducer over reduced axis k with coverage R lowers to a sequence of operations moving up the SoC hierarchy:

1. **Fragment accumulator (SEQ layer):** each thread accumulates into a register (FRAG) during its sequential loop. This is the innermost Reduce operation.
2. **Cross-thread combine (THREAD → WARP):** threads within a warp combine their partial results.
   - The destination is a warp-owned SMEM region (or a shuffle-based combiner — see Q-2).
3. **Cross-warp combine (WARP → CTA):** warps within a CTA combine their partial results.
   - The destination is a CTA-owned SMEM region.
4. **Cross-CTA combine (CTA → GMEM):** (optional, only if R includes CTA and multi-CTA reduction is needed).
   - The destination is a GMEM buffer.

**General rule (frag → smem(thread) → smem(warp) → smem(cta) → gmem):** each Combine step moves the accumulator from the inner layer's storage to the outer layer's storage. The specific mechanism (shuffle vs SMEM load/store) is a codegen detail — see Q-2.

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

(provisional: F = L on non-SEQ layers, per Q-1)

```
          M     N     K
(0/CTA)  [ 4 ][ 4 ][ 1 ]    F[0] = [4, 4, 1]
(1/WARP) [ 2 ][ 2 ][ 2 ]    F[1] = [2, 2, 2]
(2/THR)  [ 4 ][ 4 ][ 2 ]    F[2] = [4, 4, 2]
(3/SEQ)  [ 4 ][ 2 ][ 4 ]    F[3] = [4, 2, 4]
```

**Verification of column-product invariant:**
- M: 4 × 2 × 4 × 4 = 128 ✓
- N: 4 × 2 × 4 × 2 = 128 ✓
- K: 1 × 2 × 2 × 4 = 16 ✓

### 9.3 LoC Matrix L

(provisional: F = L on non-SEQ layers, so L = F for layers 0–2)

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

// Reducer: CTA-level (output buffer at CTA-owned SMEM)
// shape: surviving M×N dims at CTA level = [4,4] units * [32,16] per-unit
// = full [128, 128]? No — per-CTA output region = R[0]_M × R[0]_N = 32 × 16
// (provisional interpretation, see Q-3)

// Warp-level reducer init
float acc_frag[4][2];  // per-thread fragment: SEQ output shape = F[3][0..1] = [4, 2]

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

// Thread-level combine → warp-level (intra-warp)
// (shuffle or smem(thread); see Q-2)
reducer_combine_thread(acc_frag, thr_0, thr_1, thr_2);

// Warp-level combine → CTA-level (cross-warp, via smem(warp) → smem(cta))
reducer_combine_warp(acc_warp, warp_0, warp_1, warp_2);

// Write out CTA-owned output region
// (tail predicate for non-divisible extents — not shown)
```

### 9.7 Hoisted Cache Shapes

For the LHS buffer (indexed by [M, K]):

- **SEQ/FRAG (per thread):** `[4, 4]` — the SEQ-level tile access shape (seq_0 × seq_2 = 4 × 4). Wait — let's be precise. The per-thread sequential region is R[3] = [1,1,1] (single points). The fragment buffer for LHS in the SEQ layer would be the tile loaded at the thread level: shape = [F[3][0], F[3][2]] = [4, 4] (since N is not an index of LHS). Actually, the FRAG buffer is the per-thread tile of LHS accessed during the SEQ loop. Its shape is the per-thread region along the dimensions that index LHS.

More systematically, applying the hoisting rule:

- At SEQ (layer 3): each thread's LHS access shape = SEQ-region along indexed dims = F[3][M] × F[3][K] = 4 × 4. Storage: FRAG.
- Hoisted to THREAD (layer 2): shape = THREAD-LoC along indexed dims × inner shape = F[2][M] × F[2][K] × ... no — wait, the hoisting rule says: owner-coordinate extents at the new level ++ inner buffer shape.

Actually let me be precise. When hoisting from layer m+1 to m:
- The buffer at m+1 has shape `B_{m+1}`.
- After hoisting to layer m, shape = `LoC(m)[indexed_dims] ++ B_{m+1}`.

Wait, that's right. The number of buffer "copies" equals the number of owners at the higher level. Each copy has the inner shape.

For LHS (indexed by M, K — two dims):

- Hoisted from SEQ → THREAD: shape = `F[2][M] × F[2][K] ++ F[3][M] × F[3][K]`
  No — that double-counts. The correct shape at THREAD level is: one per thread, each holding its own tile. The tile per thread is F[3][M] × F[3][K]. So total shape = `F[2][M] × F[2][K]`... no.

Let me re-state the hoisting rule correctly:

> A buffer hoisted to level m has shape = (owner-coordinate extents at m) ++ (inner buffer shape)

The "owner-coordinate extents at m" are how many parallel owners there are at level m along the dimensions that index the buffer. The "inner buffer shape" is the per-owner buffer shape.

So for LHS at SEQ level (one copy per thread, shape = per-thread access = F[3][M] × F[3][K] = 4 × 4):

- FRAG: `[4][4]` (per-thread, register file)

Hoisted to THREAD-owned SMEM (one copy per thread, but now in SMEM — same shape, different storage class):

- SMEM(thread): `[4][4]` per thread, total = 32 threads × 4×4 floats = 32×16×4 B = 2048 B

Hoisted to WARP-owned SMEM:
- One copy per warp, each copy covers all threads in the warp. Per-warp shape = F[2][M] × F[3][M] × ... no.

I think I'm overcomplicating this. Let me use the v1 doc's formulation, which is clear: "concatenation of inner tile's cache when hoist out". The shape at layer m is the concatenation of the layer-m LoC shape and the inner buffer's shape.

For LHS (indexed by M, K):
- At SEQ (layer 3): `[F[3][M], F[3][K]] = [4, 4]`
- At THREAD (layer 2): `[F[2][M], F[2][K], F[3][M], F[3][K]]`... no, that doesn't make sense physically.

The correct interpretation (consistent with v1's example): when you hoist a buffer up a layer, you add the outer layer's coordinate extents as new leading dimensions. The buffer has one "slot" per unit of the outer layer, and each slot has the inner shape.

So LHS at:
- SEQ: `[4][4]` (4×4 floats = per-thread tile, indexed by seq_M and seq_K)
- THREAD: `[4][2][4][4]` (thread dims: thr_M=4, thr_K=2; each thread has 4×4 frag — total 4×2×4×4? No, thr_M and thr_K are the thread index dimensions, not spatial dimensions of the buffer. Each thread owns one 4×4 tile. The SMEM(thread) buffer is indexed by thread coordinates AND intra-thread coordinates.)

Actually, yes — that's exactly it. An SMEM buffer owned by "thread" is indexed by (thread_id, local_index). The thread_id part has shape LoC(thread), and the local part has shape per-thread.

So: SMEM(thread) LHS shape = `[F[2][M], F[2][K], F[3][M], F[3][K]]` = `[4, 2, 4, 4]`

Wait, but F[2][K] = 2, and the K dimension is the reduced one. At layers above RSR stop, the reduced dimension is elided. Let me check RSR first.

### 9.8 RSR and Reduce Sequence

Reduced axis: K (dim 2), extent S[2] = 16.

RSR scan (inner to outer, layers 3→2→1→0):
- Layer 3 (SEQ): acc = 1 × 4 = 4 < 16 → continue
- Layer 2 (THR): acc = 4 × 2 = 8 < 16 → continue
- Layer 1 (WARP): acc = 8 × 2 = 16 == 16 → STOP

Coverage set R = {1, 2, 3} (WARP, THREAD, SEQ). RSR stops at layer 1 (WARP). Layer 0 (CTA) is NOT covered.

**Reducer sequence (inner to outer):**
1. SEQ: per-thread fragment accumulation (FRAG) — Reduce per element
2. THREAD → WARP: intra-warp thread combine — Combine to warp-owned SMEM
3. WARP: done — the reduce output lives at warp-owned SMEM (or whatever the WARP-level destination is)

The CTA layer has no reduce combine step — the surviving M×N dimensions are partitioned across warps, and each warp writes its own slice of the output.

**(provisional, see Q-3)** What about the `reducer<cta>` in the cg sample? It likely represents the CTA-level output buffer allocation, not a CTA-level combine step (since there is no CTA-level combine when RSR stops at WARP).

### 9.9 Cache Shapes with RSR Axis Elision

With K as the reduced axis and RSR stopping at WARP (layer 1):

The accumulator/reducer buffer at each level only carries surviving (non-reduced) dimensions:

- **SEQ/FRAG:** per-thread accumulator shape = surviving SEQ dims = F[3][M] × F[3][N] = [4][2]
  - The K dimension is consumed by the reduce; it doesn't appear in the accumulator shape.
- **THREAD → WARP combine destination (warp-owned SMEM):**
  - shape = warp_coords_surviving ++ per-thread_acc_shape
  - = [F[2][M], F[2][N]] ++ [F[3][M], F[3][N]] = [4, 4, 4, 2]
  - Wait — each warp has F[2][M] × F[2][N] = 4 × 4 = 16 threads, and each contributes a [4][2] accumulator.
  - The warp-owned SMEM buffer shape = [16, 4, 2] = [F[2][M]*F[2][N], F[3][M], F[3][N]]
  - Or equivalently, indexed by thread coordinates: [F[2][M], F[2][N], F[3][M], F[3][N]] = [4, 4, 4, 2]

---

## 10. Open Questions / Required Author Rulings

### Q-1: Must factor extent equal LoC on every non-SEQ layer?

**Question:** Is `F[m][i] = L[m][i]` for all non-SEQ layers m a required canonicalization, or just the common case?

**Reading A (required):** All per-unit sequential slack is pushed into the single SEQ layer. Parallel layers have no sequential interior beyond what the inner layers cover. This is the simpler model and all worked examples follow it.

**Reading B (optional):** A parallel layer's factor extent can be larger than its LoC, meaning each unit sequentially iterates over the remaining factor. For example, a CTA with F[0][M] = 8 and L[0][M] = 4 would mean each CTA processes 2 sub-tiles of dimension M sequentially.

**Mechanical impact:**
- A: simpler codegen (only SEQ layer has sequential loops); search space = LoC space.
- B: richer schedule space; sequential loops at any layer; more complex codegen.

**Provisional reading used in this doc:** A (F = L on parallel layers), marked "(provisional, see Q-1)" at each relevant point.

### Q-2: Cross-thread collection: shuffle vs SMEM(thread) region

**Question:** When combining partial results across threads within a warp, is the mechanism: (a) warp shuffle instructions (`__shfl_sync` family), (b) thread-owned SMEM regions with loads/stores and a barrier, or (c) both with a selection rule?

**Reading A (shuffle):** Thread-level combine uses warp shuffle. This is register-file-only, intra-warp only, no SMEM allocation.

**Reading B (SMEM(thread)):** Each thread writes its accumulator to a thread-owned SMEM slot; combine reads from SMEM; barrier synchronization required. Works across warps (with appropriate synchronization).

**Reading C (both):** Intra-warp combine uses shuffle; cross-warp combine uses SMEM(warp) or SMEM(cta) depending on level.

**Mechanical impact:**
- Affects SMEM budget accounting (does thread-level combine consume SMEM?).
- Affects the reducer lowering path (different codegen for shuffle vs SMEM).
- Affects applicability: shuffle is intra-warp only; SMEM works at any level.

**Note:** The storage class + owner model is compatible with all three readings — the question is about the collection mechanism, not the data model.

### Q-3: Role of reducer.Init at layers above the RSR stop layer

**Question:** What does a `reducer<X>` represent when X is a layer above the RSR stop layer? For example, in our matmul example RSR stops at WARP (layer 1), but the cg sample initializes a `reducer<cta>`.

**Reading A (output buffer):** Init at the outermost R layer allocates the output buffer at that layer's owned storage. In our example, WARP is the outermost covered layer, so the output is at warp-owned SMEM. The "cta" reducer in the sample is a misnomer / pre-allocated CTA-level output buffer that warps write into directly.

**Reading B (scaffolding):** Init at layers above RSR is a no-op structural placeholder. No allocation; no combine at that level.

**Reading C (accumulator):** The reducer is an actual accumulator at the CTA level that receives cross-warp combines — but this would mean RSR *should* include CTA, which contradicts the math (K=16, 2×2×4=16 → stop at WARP).

**Mechanical impact:**
- Affects codegen shape of multi-level reducers.
- Affects SMEM allocation: does the CTA layer allocate a reducer buffer?
- Clarifies what "RSR stop" means for the layer above.

### Q-4: RSR generalization to multiple reduced dimensions

**Question:** How does RSR generalize when there are multiple reduced dimensions (chained reduces, or a single reduce over multiple axes)?

**Reading A (independent):** Each reduce axis has its own RSR, computed independently. The reducer stack for axis k stops at its own layer.

**Reading B (shared):** All reduces share a single layer sequence; the RSR is determined by the outermost stop layer across all reduced axes.

**Mechanical impact:**
- Affects scheduling of multi-reduce chains.
- Affects reducer stack shape (one unified stack vs per-axis sub-stacks).
- Likely deferred beyond v0 (v0 has exactly one reduce per map chain).

### Q-5: Per-unit region shape vs stride — one quantity or two?

**Question:** Is "per-unit region shape" the same quantity as "stride row", or are they distinct?

**Reading A (same):** stride[m][i] = R[m][i] = product of factors below m in column i. Both names refer to the same number. The cg sample's middle column has a typo.

**Reading B (different):** There are two distinct quantities — e.g. "output region shape" (only surviving non-reduced dims) and "full stride" (all dims including reduced ones).

**Mechanical impact:**
- Determines whether the canonical doc needs one or two N-vector quantities per layer.
- Reading A is simpler and consistent with the definitions in §4.2–§4.3.
- Reading B would be needed if the cg sample's middle column intentionally shows a different quantity.

**Provisional reading used in this doc:** A (one quantity), with the cg sample's middle column treated as erroneous (F-3).
