# CUDA Memory Coalescing Optimization Pass

## 1. Overview

This compiler pass analyzes memory access patterns in CUDA kernels and automatically derives optimal tensor view transformations to improve Global Memory (GMEM) coalescing. The pass operates on parallelized tensor dimensions to reorder how multi-dimensional indices map to linear memory addresses, ensuring consecutive threads access consecutive memory locations.

**Scope:** Optimization applies to affine memory access patterns within the K parallelized dimensions of N-dimensional tensors.

## 2. Problem Statement

### Input
- Multi-dimensional tensor with shape `(D0, D1, ..., D_{N-1})`
- Subset of K dimensions involved in GMEM coalescing optimization (K ≤ 5)
- Current row-major strides `S = (s0, s1, ..., s_{N-1})`
- Data type with known byte size

### Objective
Find a dimension permutation π such that the recomputed strides `S' = (s'0, s'1, ..., s'_{N-1})` satisfy:
- **Coalescing criterion:** `S'[K-1] = sizeof(dtype)` (last optimized dimension has unit stride)
- **Bijection property:** New stride configuration maps the K-dimension iteration space identically to original (no gaps, overlaps, or incomplete coverage)
- **Quality metric:** Minimize warp fragmentation and memory access inefficiency

### Constraints
- Total element count unchanged (product of all dimensions constant)
- K ≤ 5 dimensions (brute-force search over K! permutations)
- Solver produces deterministic, reproducible results

## 3. Pass Architecture

```
Input: Kernel IR (analyzed memory access patterns)
  ↓
[Stage 1] Memory Access Pattern Analysis
  → Identify access sites, extract K parallelized dimensions
  ↓
[Stage 2] Optimal Stride Solver
  → Brute-force permutation search (K ≤ 5)
  → Compute new row-major strides
  ↓
[Stage 3] Code Generation
  → Emit optimized view instantiation (template params)
  → Insert transformation metadata
  ↓
[Stage 4] Integration
  → Patch kernel code with optimized view
  → Link solver output to runtime
  ↓
Output: Coalescing-optimized kernel
```

## 4. Stage Definitions

### 4.1 Memory Access Pattern Analysis

**Input:** Kernel function definition, tensor access sites (loads/stores)

**Responsibilities:**
- Identify all GMEM tensor access patterns
- Extract affine index expressions for each access
- Determine which dimensions are parallelized (thread/warp/block level)
- Partition dimensions into:
  - **Parallelized (K):** Candidate for coalescing optimization
  - **Non-parallelized (N-K):** Use original strides

**Output:** Structured representation of each access site with:
```
{
  access_id: unique identifier,
  tensor_shape: (D0, D1, ..., D_{N-1}),
  dtype_size: sizeof(dtype),
  parallelized_dims: [indices of K dimensions],
  original_strides: (s0, s1, ..., s_{N-1})
}
```

**Decision points:**
- Is access affine? (Yes → optimize; No → skip)
- Can K dimensions be isolated? (Yes → proceed; No → skip)

### 4.2 Optimal Stride Solver

**Input:** Access site metadata (from Stage 1)

**Algorithm:**
1. Generate all K! permutations of the K parallelized dimensions
2. For each permutation π:
   - Reorder dimensions: dims' = [dims[π[0]], dims[π[1]], ..., dims[π[K-1]]]
   - Compute row-major strides for reordered dimensions
   - Assign stride for last dimension: `S'[K-1] = sizeof(dtype)`
   - Score the configuration (see 4.2.1)
3. Select permutation with lowest score (optimal quality)

**Output:** 
```
{
  access_id: identifier,
  permutation: [π[0], π[1], ..., π[K-1]],
  original_strides: (s0, ..., s_{N-1}),
  optimized_strides: (s'0, ..., s'_{N-1}),
  score: quality metric,
  dtype_size: sizeof(dtype)
}
```

#### 4.2.1 Scoring Function

**Primary metric:** Minimize last dimension stride deviation
- Prefer `S'[K-1] = sizeof(dtype)` (perfect coalescing)

**Tiebreaker:** Minimize stride gaps (variance in strides)
- Minimizes control flow complexity in warp scheduling

**Formula (example):**
```
score = |S'[K-1] - sizeof(dtype)| + 0.1 * variance(S'[0:K])
```

#### 4.2.2 Bijection Guarantee

Row-major stride computation always produces bijective mappings over the K-dimension iteration space. Proof sketch:
- Original stride configuration partitions `∏ Di` into K-dimensional iteration space
- Permuting dimensions + recomputing strides preserves this partition (same element count)
- No reordering or remapping of elements—only relabeling of how iteration indices map to offsets
- Result: identical iteration space coverage, different memory order

### 4.3 Code Generation

**Input:** Solver output metadata

**Responsibilities:**
- Generate C++ tensor view instantiation call with solver results
- Embed permutation and optimized stride parameters
- Optionally generate verification/profiling hooks

**Output Format (example for TensorViewCoalesced):**
```cpp
auto view = TensorViewCoalesced<float, N, K>(
  tensor_ptr,
  original_shape,
  permutation,          // from solver
  optimized_strides     // from solver
);
```

**Metadata embedding:**
- Store solver output in kernel binary (symbol table, section, or comment)
- Allow runtime inspection of applied optimizations

### 4.4 Integration

**Responsibilities:**
- Locate memory access sites in kernel code
- Replace original tensor access with optimized view
- Update index expressions (if needed)
- Emit integration hooks for downstream compilation

**Assumptions:**
- Kernel uses structured tensor access (not pointer chasing)
- Access indices are available at compile time or symbolic

## 5. Data Flow

### Per Access Site:
```
1. Analysis extracts: shape, dtype, parallelized dims, original strides
2. Solver brute-forces K! permutations → finds optimal stride configuration
3. Codegen emits: TensorViewCoalesced instantiation with solver output
4. Runtime: View transparently maps user indices → optimized affine arithmetic
5. Result: GMEM addresses coalesce naturally, no user code changes
```

### Compilation Pipeline Integration:
```
Kernel IR
  ↓
[This Pass] CUDA Coalescing Optimizer
  ↓
  ├─→ Identify access sites
  ├─→ Run solver for each site
  ├─→ Generate view code
  └─→ Patch kernel
  ↓
Lower-level IR (device code)
  ↓
[Downstream] NVCC/LLVM → PTXAS → cubin
```

## 6. Input/Output Specification

### Pass Input
- Kernel function (IR representation)
- Tensor metadata (shape, dtype, layout)
- Parallelism constraints (thread/warp/block mapping)

### Pass Output
- Modified kernel code with optimized views
- Solver metadata (permutations, strides per access site)
- Quality metrics (score per access, overall coalescing improvement estimate)

## 7. Constraints & Assumptions

| Constraint | Rationale |
|---|---|
| K ≤ 5 dimensions | Brute-force solver guarantees optimality; K! ≤ 120 permutations |
| Affine access patterns only | Solver operates on stride-based indexing |
| Parallelized dims isolated | Must clearly identify which dimensions map to GMEM coalescence |
| sizeof(dtype) known | Required for stride coalescing criterion |
| Row-major layout assumed | Pass designed for C-contiguous tensors |

## 8. Extension Points

- **Custom scoring functions:** Allow application-specific metrics (L2 cache, bank conflicts)
- **Dynamic dimension sets:** Support runtime determination of K (requires fallback)
- **Multi-access optimization:** Jointly optimize multiple access sites (future)
- **Vectorization:** Exploit shuffle/warp-shuffle for sub-dimensional coalescing

## 9. Example

### Input Kernel:
```cuda
void kernel(float* A, int N, int M) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // 2D tensor access: row-major, shape (N, M)
  // Parallelized over both dimensions
  A[tid / M * M + tid % M] = compute();
}
```

### Solver Analysis:
- Shape: (N, M), dtype: float (4 bytes)
- Original strides: (M*4, 4)
- Parallelized dims: [0, 1] (both)
- Permutation search: [0,1] vs [1,0]
  - [0,1]: strides (M*4, 4) → last stride = 4 ✓
  - [1,0]: strides (4, 1) → last stride = 1 ✗

### Solver Output:
- Permutation: [0, 1] (no reordering needed)
- Optimized strides: (M*4, 4)
- Score: 0

### Generated Code:
```cpp
auto view = TensorViewCoalesced<float, 2, 2>(
  A, 
  shape={N, M},
  permutation={0, 1},
  optimized_strides={M*4, 4}
);
// User code now uses view for transparent coalescing
```

## 10. Testing Strategy

- **Correctness:** Verify bijection (all elements accessed exactly once)
- **Performance:** Measure GMEM throughput before/after optimization
- **Regression:** Ensure pass doesn't break non-optimizable access patterns
- **Determinism:** Same input → same solver output across runs
