# TCA: Tensor Computation Algebra

## 1. Motivation and Goals

TCA is an algebra designed as the core IR of a GPU kernel optimizer. It targets NVIDIA CUDA C++ as its code generation backend.

**Design goals**

- **Compact, expressive, and sound.** A small set of primitives with well-defined algebraic rewrite rules.
- **No explicit control flow.** Loops are expressed as iteration spaces, not as branches and back edges.
- **Automatic parallelization and tiling.** Every loop is tiled by default, and parallelism is assigned by the compiler rather than bound to CUDA threads by the user.
- **A single IR for all optimizations.** Fusion, tiling, caching, and parallelization all operate on the same IR, with no conversions between different IR families.

---

## 2. Syntax

The algebra is made up of five kinds of elements: scalars, sets, induction variables, iterations, and functions, which are combined with operators.

### 2.1 Scalar

A single typed value (see [Section 4, Types](#4-types)).

### 2.2 Sets

| Kind | Notation | Empty | Ordering |
|---|---|---|---|
| Unordered set (*uset*) | `< >` | `<>` | None |
| Ordered set (*tuple*) | `( )` | `()` | Total order by position |
| Partially ordered set (*poset*) | `{ }` | `{}` | Induced by data dependencies (def-use / use-def) between elements |

**Derived sets**

- **Iteration space.** A uset of scalars. Iteration spaces are **unbounded** by default. For example, the most common induction variable enumerates `<0, 1, 2, 3, ...>`.
- **Sequence.** A set that contains only scalars, plus two special marker elements, `<start>` and `<end>`. The markers are placeholders, not numeric values. Sequences are the input of `reduce`.

### 2.3 Induction Variables

An induction variable is written `@name` and refers to an iteration space.

- **Definition.** An induction variable is defined by `(init, step)`.
  - `@x(init=0, step=1)` is the **standard induction variable**.
  - `@x(init=0, step=T)` is a **tiled induction variable** with tile size `T`. The standard induction variable is the special case `T = 1`.
  - The fully qualified form is `@x(init=..., step=...)`.
- **Bounds.**
  - An induction variable is unbounded by default.
  - When a bound is needed, for example inside an iteration context, the variable is attached to a bound and becomes a **bounded induction variable**, written `@x/M`, meaning `@x < M`.
  - A bare `@x` means the variable is unbounded, or that its bound is irrelevant in the current context.

### 2.4 Iterations

An iteration enumerates the elements defined by an induction variable in an **unspecified order**.

```
iter(<induction_variable>, [offset=<expr>]) -> iteration space
```

- `iter` maps an induction variable to its iteration space. The mapping is **unordered**.
- If `offset` is given, every value is shifted by it. The default is `offset = 0`.
- Inside `iter`, induction variables are always assumed to be bounded.
- **Forms**
  - Standard form: `iter(@a)`
  - Compound form: `iter(@a, @b, @c)`, which is defined as `iter(@a/A) * iter(@b/B) * iter(@c/C)`
- Iterations combined with `*` form an **iteration cluster (IC)**, which corresponds to a loop nest tree.

**Tiling**

```
tile(@x/M, T) = iter(@x_o(step=T)/M) * iter(@x_i/T, offset=@x_o)
```

Here `@x_o` is the *tile-between* (outer) variable and `@x_i` is the *tile-inside* (inner) variable. When `T` does not divide `M`, the inner iteration is implicitly guarded by `@x_o + @x_i < M`.

### 2.5 Functions

Functions map sets to sets.

| Function | Signature | Semantics |
|---|---|---|
| `map` | `map(tuple, lambda) as (@v0, ..., @vn) -> tuple` | Applies `lambda` element-wise. **Pure** (side-effect free) and **unordered**. The `as` clause is **mandatory** and fixes the output order (see [Section 2.7](#27-output-order-as)). |
| `reduce` | `reduce(sequence) -> scalar` | Reduces the **last dimension in the producing map's `as` order**. For example, `as (@a, @b, @c)` becomes `(@a, @b)`. **Unordered.** |
| `load` / `store` | | Memory load/store of a scalar or tensor. `store` returns the stored value, which allows further fusion. |
| `none` | `none() -> ()` | Returns the empty set, meaning no return value. |

**Notes on `map`**

- `iter() -> map(...)` is a map over a zero-arity iteration: it applies its lambda at the single point of the empty iteration space. In a fused chain this is exactly a lambda applied to the piped value. It appears at statement level only as the fused tail of a chain (for example after a reduction); lowering re-materializes it over its live axes ([Section 5.10](#510-reduce-lowering)). A zero-arity map is never a standalone statement in a constructed graph, and it is not a no-op.

**Notes on `reduce`**

- `reduce` is **stateful**, because it carries an accumulator, but it has **no observable side effects**.
- Because `reduce` is unordered, its combiner **must be associative and commutative**. For floating-point types this holds only up to rounding, so results may differ bitwise between parallelization strategies.
- Every value chain **starts with a `map`**. A `reduce` can only appear chained after a map or another reduce.

For brevity, *lambda* in this document refers to either a `map` or a `reduce` operator.

### 2.6 Operators

**Cartesian product: `*`**

- Produces the Cartesian product of two usets.
- Commutative and associative.

**Concatenation: `->`**

- A 1:1 mapping. Read it as "loop body definition".
- Grouping: `a -> b -> c == -> {a b c}`, which defines a **poset** whose order comes from the data dependencies of its elements.
- Return value:
  - `-> a` returns the value of `a`.
  - `-> { ... last }` returns the value of `last`.
- Any list of functions can be attached to an `iter`:
  - `iter(@x) -> { ... }`
  - Nesting is allowed: `iter(@x) -> { iter(@y) -> { ... } } == iter(@x) * iter(@y) -> { ... }`
- `->` works for both map and reduce: `iter(@x) -> <identity map> | reduce == iter(@x) -> reduce`.

**Pipe: `|`**

- Passes the value returned by its left-hand side to its right-hand side as the bound variable `%0`.
- `|` is a construction-time shorthand for data dependencies (and therefore order). It is used before explicit SSA dependencies are formed and is removed during [Effect Linearization](#56-effect-linearization).

### 2.7 Output Order: `as`

Iteration spaces are unordered, and `*` is commutative, so the algebra itself carries no dimension order. The order is stated explicitly by the user on every map:

```
iter(@m) * iter(@k) * iter(@n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n)
```

**Rules**

1. **Mandatory.** Every `map` carries an `as` clause, including fused maps and zero-arity maps (`iter() -> map(...)`). A map without `as` is a verifier error.
2. **Permutation.** `as` lists every induction variable live at the map exactly once: for a root map, all variables of its iteration cluster; for a fused map, the variables still live at that point of the chain.
3. **Dimension order.** `as` defines the order of the map's result dimensions. A downstream `reduce` reduces the **last** variable of `as`; this is the source of the `reduce<@r>` annotation used in [Section 5.9](#59-iteration-lowering).
4. **Loop order.** `as` is the only source of loop nest order. The lowered loop nest follows it exactly ([Section 5.9.1](#591-loop-nest-order)).
5. **Fused maps (v0).** A map fused after a `reduce` must use the producer's `as` with the reduced variable removed. A map fused after a map must use the same `as`. Reordering inside a fused chain is not supported in v0.
6. **Relation to `store`.** `as` fixes the loop order; the `store` index fixes the memory layout. They may differ (for example a transposed write), but when the last variable of `as` does not index the store's last (contiguous) axis, global writes are uncoalesced, and the verifier emits a performance warning.

---

## 3. Algebra Rules

### 3.1 Fundamentals

- `iter(@x)` defines the iteration space of `@x`.
- `iter(@x)` combines with other components through `*` or `->`:
  - `iter(@x) * iter(@y)` produces every `(x, y)` pair, which is a nested loop.
  - `iter(@x) -> ...` attaches one or more functions to each iteration, forming the loop body.
- `a -> b` imposes **no order** on the calls beyond data dependencies.
- `a | b` imposes a **data-dependent order**.

### 3.2 Rewrite Rules

**Factor rule**

```
iter(@x, @y) = iter(@x) * iter(@y)
```

**Commute rule**

```
iter(@x) * iter(@y) = iter(@y) * iter(@x)
```

> **Order:** commuting never changes the iteration space, and dimension order (which dimension `reduce` acts on, and the loop nest order) comes only from the map's `as` clause ([Section 2.7](#27-output-order-as)). The commute rule is therefore always valid, including for bodies that contain a `reduce`.

**Lambda rule**

Applying a lambda to an iteration space applies it to every point in the space:

```
lambda(iter(@x) * ... * iter(@n)) = iter(@x) * ... * iter(@n) -> lambda
```

**Lambda combine rule** (`{ }` is a poset)

```
(1) a -> b -> ... -> k = -> { a b ... k }
(2) a -> { b c d }     = -> { a b c d }
(3) { a b c } -> d     = { a b c d }
```

**Iteration merge rule**

Given two statements that share the iteration prefix `@x0 ... @xk`:

```
A: iter(@x0) * ... * iter(@xk) * iter(@a_{k+1}) * ... -> { A }
B: iter(@x0) * ... * iter(@xk) * iter(@b_{k+1}) * ... -> { B }
```

merge the **longest common prefix**, where prefixes are compared in each statement's `as` order, and keep the remaining parts separate:

```
iter(@x0) * ... * iter(@xk) -> {
  iter(@a_{k+1}) * ... -> { A }
  iter(@b_{k+1}) * ... -> { B }
}
```

**Iteration clean rule**

1. *Base case.* In `iter(@a) * ... * iter(@z) -> map`, if the map's result is unused and the chain contains no `store`, the whole statement is removed.
2. *Recursive case.* For

   ```
   iter(@a) * ... * iter(@z) -> map | reduce(...) | reduce(...) | map ...
   ```

   removing the leading (identity) map moves the fused chain up to the iteration:

   ```
   iter(@a) * ... * iter(@z) -> reduce(...) | reduce(...) | map ...
   ```

**Load/store hoisting rule**

This rule is the basis of automatic shared-memory (SMEM) allocation. It defines how loads can be hoisted along the iteration chain.

```
iter(@x) * iter(@y) * iter(@z) -> {
  %l = load(%lhs[@x][@z])
  %r = load(%rhs[@y][@z])
  ...
}

=

iter(@x) * iter(@y) -> {
  // Load the whole @z-bounded slice into a local cache.
  %cache_l = load(%lhs[@x])   // indexed only by @x: can be hoisted further, above iter(@y)
  %cache_r = load(%rhs[@y])   // indexed by @y: cannot be hoisted above iter(@y)

  iter(@z) -> {
    %l = load(%cache_l[@z])
    %r = load(%cache_r[@z])
    ...
  }
}
```

**General form.** For each innermost element load

```
%l0 = load(%src0[@i0][@i1]...)
%l1 = load(%src1[@j0][@j1]...)
```

a load can be promoted to the deepest enclosing `iter` whose induction variable still appears in its index list. In other words, it can be hoisted above every loop whose variable it does not use. This is the basic SMEM caching rule, and it must be combined with the final parallelization of each iteration (see [Section 5.8](#58-cta-cache-allocation)).

### 3.3 Tiling Rule

```
iter(@x/M) -> lambda  ==  tile(@x/M, T) -> lambda
```

For any lambda (map or reduce), the result is the same before and after tiling, because:

- `iter(@x/M)` and `tile(@x/M, T)` define **exactly the same iteration space** (with the remainder guard from [Section 2.4](#24-iterations)).
- **map** is an unordered operation on single elements of the space.
- **reduce** is an unordered (associative and commutative) operation over one dimension of the space.

This guarantees that tiling is always safe: map and reduce can be factored in and out of tiled expressions freely.

---

## 4. Types, Shapes, and Lambda Expressions

### 4.1 Types and Shapes

The IR is strongly typed.

- **Scalars:** `f32`, `f16`, `f8`, `bf16`, `i32`, `i16`, `i8`, `u32`, `u16`, `u8`, and `bool`. The concrete supported `f8` formats must be specified before their backend implementation.
- **Tensors:** `tensor<dtype=f32, shape=[M, N]>`.
- A shape dimension is either an integer constant or a runtime dimension value such as `M`. Symbolic dimensions are runtime values, not necessarily compile-time template parameters. Rank, element types, and preset tile/thread-group sizes are compile-time information.
- Runtime dimensions must be nonnegative and consistent between operands that share an iteration dimension; incompatible dimensions are rejected by launch-time validation.

### 4.2 Typed Scalar Expressions

Map and reduce lambdas contain typed scalar expression graphs. A lambda sees operand **values**, not induction variables, tensor addresses, or indices. Tensor indexing and iteration ownership remain structural parts of TCA outside the lambda.

The expression IR supports constants, scalar parameters, arithmetic, comparisons, Boolean operations, value selection, explicit casts, and calls to supported CUDA scalar builtins such as `expf`. Calls must have registered signatures and be pure: arbitrary CUDA functions, memory operations, and synchronization are not lambda expressions. The builtin registry determines valid argument/result types and backend spelling.

```
map(..., lambda(x: f16): expf(cast<f32>(x))) as (@m, @n) |
reduce(init=0.0f, lambda(acc: f32, now: f32): acc + now)
```

A cast produces a value of its explicitly named target type. Operators and builtin calls are checked against their typing rules; inference must determine every expression's type unambiguously, otherwise compilation fails. Conversions are represented explicitly rather than silently changing the reduction's numerical policy.

Numerical policy is pinned in the frontend spec: IEEE 754 per-operation semantics, FMA contraction permitted by default and documented per operation, no flush-to-zero, no silent precision reduction, no fast-math. For a fixed compiled kernel, execution is deterministic (no atomics, fixed schedule); parallel reductions remain schedule-dependent across different compilations as stated in [Section 2.5](#25-functions).

The initializer and combiner determine the reducer accumulator/result type. Low-precision values can be cast to `f32` before reduction, and a fused map can cast the result to the output type. In v0 every reducer is **homogeneous**: the input element type, the initializer type, both combiner parameter types, and the result type are one and the same scalar type. Mixed-type combiners are deferred; when added they must have a declared, type-checked signature.

### 4.3 Bounds and Lambda Invocation

The generated runtime invocation checks logical indices before loading tensor operands or calling a lambda. A map lambda is invoked only for valid iteration points whose required tensor values exist. A reduce combiner receives only valid produced values; padded elements do not participate. Scalar operands require no tensor-index check.

These predicates apply to computation and memory access, not collective participation. Threads with no valid elements still call reducer `Init`/`Combine` and cooperative tile operations in uniform control flow. An empty reduction returns the declared initializer. No user lambda needs to implement indexing or out-of-bounds handling.

---

## 5. Compilation Pipeline (by Example)

The running example is a matmul followed by a scale:

```
// Illustrative frontend syntax, similar to TCA. Not in SSA form, for simplicity.

kernel f32[M][K] matmul_scale(f32 lhs[M][N], f32 rhs[N][K], f32 scale) {
  dimvar @m;   // induction variables
  dimvar @n;
  dimvar @k;

  let output: tensor<f32, [M, K]> =
      map on(lhs[@m/M][@n/N], rhs[@n/N][@k/K]) as (@m, @k, @n)
        apply(l: f32, r: f32) -> f32: l * r                     |
      reduce(init = 0.0f)                       // reduces @n, the last of `as`
        apply(acc: f32, now: f32) -> f32: acc + now             |
      map as (@m, @k)
        apply(%0: f32) -> f32: %0 * scale;

  return output;
}
```

### 5.1 Graph Construction and Optional Frontend Lowering

Implementation starts with direct construction of the typed TCA graph. Parsing is a separate component and is not required for the first compiler implementation. A future frontend builds an AST and integrates through an AST-to-TCA pass; its syntax is sugar over the same IR.

A frontend `kernel` function is the entry for a group of statements. At the TCA level it has one basic block and no explicit control-flow graph branches. This logical kernel is not necessarily one physical CUDA launch: unsupported fusion boundaries may require multiple launches. The v0 execution interface is conservative and fully specified: each statement cluster is lowered to exactly one `__global__` kernel; named intermediates that cross cluster boundaries are materialized in a per-launch global-memory workspace owned by the runtime; and the compiler emits one host launch function per frontend kernel that validates dynamic extents at launch, allocates output storage, and issues the cluster launches in dependency order. Cross-cluster fusion and alternative synchronization mechanisms are future work.

TCA is an SSA-style IR:

1. Local `%` variables are in SSA form and immutable. Def-use chains through memory are linked by `store`/`load`.
2. There is no control flow, so there are no PHI nodes.
3. A kernel is a single basic block.
4. The basic block contains statements. Pure computation (`map`, `reduce`) has no side effects; memory effects are expressed only through `load`/`store` and are ordered explicitly during [Effect Linearization](#56-effect-linearization).

```
statement #8:
  iter(@m, @k, @n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) |
  reduce(init=0.0, %0, lambda(acc, now): acc + now)                                      |
  iter() -> map(%0, lambda(v): v * scale) as (@m, @k)                                    |
  store(%output[@m][@k], %0)

statement #9:
  return %output
```

This is a map + reduce formulation. `@n` is last in `as`, so it is the reduced dimension and the innermost loop. (Statement IDs come from the graph's shared node-ID counter: this kernel's seven parameters `@m, @k, @n, lhs, rhs, scale, output` are nodes 1-7, so the statements built from them are #8 and #9.)

### 5.2 Simplification

- **Dead code elimination.** Any SSA definition without a use is dead, and the chain to its right is removed along with it, unless the chain contains a `store`.
- **Identity removal.** An identity map is removed by applying the iteration clean rule.
  - The only identity map is `map(a): a`. A map with arity ≥ 2 cannot be an identity.
- **Canonicalization.** Every iteration is factored from compound form into standard form:

```
statement #8:
  iter(@m) * iter(@k) * iter(@n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) |
  reduce(init=0.0, %0, lambda(acc, now): acc + now)                                                    |
  iter() -> map(%0, lambda(v): v * scale) as (@m, @k)                                                  |
  store(%output[@m][@k], %0)

statement #9:
  return %output
```

### 5.3 Loop Merge

**Precondition:** the two statements are **data independent**, meaning neither uses a value or memory location defined by the other.

Apply the iteration merge rule:

```
// Two independent statements whose iteration chains partially overlap.

// statement #1
iter(@m) * iter(@k) * iter(@n1) -> map(a[@m][@n1], b[@n1][@k], lambda(l, r): l * r) as (@m, @k, @n1) |
store(%first[@m][@k][@n1], %0);

// statement #2
iter(@m) * iter(@k) * iter(@j) -> map(c[@m][@k][@j], lambda(v): v * scale) as (@m, @k, @j) |
store(%second[@m][@k][@j], %0);

=== loop merge ===>

// Longest common prefix: (@m, @k).
// -> and -> { } can be regrouped freely.

iter(@m) * iter(@k) -> {
  iter(@n1) -> map(...) | store(...);
  iter(@j)  -> map(...) | store(...);
}
```

### 5.4 Loop Fusion

**Precondition:** the first statement produces a value that the second statement consumes, and the **first statement's `as` list is a prefix of the second statement's `as` list**. For example, the first iterates `(A, B, C)` and the second iterates `(A, B, C, ...)`.

**Transformation:** remove the overlapping iterations from the second statement and attach the remainder to the end of the first statement's chain. Because `store` returns the stored value, the pipe chain stays intact.

**Additional precondition:** the consumer must take the producer's value through the pipe, at the same iteration point. A consumer that reloads the producer's buffer at different indices (for example a transpose) reads values computed by other threads. Such statements are not fused and stay in separate kernels.

### 5.5 Iteration Factoring

Covered by the canonicalization step in [Section 5.2](#52-simplification).

### 5.6 Effect Linearization

This pass proactively materializes every load and store, removes all pipes, and makes data dependencies explicit and linear.

1. **Remove `|`.** Mechanically rewrite each `-> LHS | RHS` as:

   ```
   -> {
     %var1 = LHS;
     %var2 = RHS(%var1, ...);
   }
   ```

   Chains `-> LHS1 | LHS2 | ...` are handled recursively, applying the lambda combine rule as needed.

2. **Combine lambdas.** Apply the lambda combine rule to flatten the result into blocks of lambdas.

### 5.7 Automatic Parallelization

The compiler assigns a parallelization strategy to every `iter`. Users never bind loops to CUDA threads, blocks, or grids manually. Parallelism is a **property of an iteration**, and assigning it selects a specialized lowering.

#### 5.7.1 Remapping the CUDA Thread Hierarchy

CUDA organizes threads as Grid → Block → Thread, each with up to three dimensions (`blockIdx.{x,y,z}`, `threadIdx.{x,y,z}`). This fixed 3-D model does not suit our runtime, which needs to schedule arbitrary partitions. We therefore remap it while preserving CUDA semantics:

- **Blocks and threads stay separate concepts**, because CTA-scoped resources such as shared memory define a parallelism scope.
- Instead of `x/y/z`, we use sequential indices: `block(0), block(1), ...` and `thread(0), thread(1), ...`.

**Design**

1. **Linearize CUDA ids.** Flatten `blockIdx` and `threadIdx` into:
   - the CUDA Block Linear Id (**CBLId**), and
   - the CUDA Thread Linear Id (**CTLId**).
2. **Compute totals:** `cuda_total_blocks` and `cuda_total_threads_per_cta`.
3. Each logical Block/Thread object has:
   - a **logical id**, used when translating iterations, and
   - its **CUDA linear id** (CBLId or CTLId), used to derive warp index and lane id.
4. **Partition the resources into N shards.**
   - **Blocks** are partitioned at any integer granularity by percentage (for example 30% / 50% / 20%). To keep the arithmetic stable, each shard's start is computed from the cumulative count of the shards before it, and the **last shard takes the remainder** (subtraction instead of multiplication), so that rounding never loses or duplicates blocks.
   - **Threads per CTA** are partitioned in **whole warps** (multiples of 32). This keeps warp-level reductions simple, because each dimension's thread group is always a whole number of warps. The parameters are:
     - `number_of_warps_per_cta`
     - allocation on warp boundaries. For example, `ThreadDim<0, 1>(cuda_total_threads_per_cta)` assigns one warp to thread shard 0.

#### 5.7.2 Sharding Parameters and Warp-Based Assignment

| Level | Parameters | Constraint |
|---|---|---|
| Grid | shard index; percentage of launched blocks | Percentages across all block shards of a statement cluster (SC) sum to 100%. |
| Thread | group index; number of warps | An assigned thread group contains a whole number of **physical CUDA warps**. Each reduced output is processed by N complete warps; no physical warp mixes independent output reductions. CTA thread counts stay within device limits. |

`Thread` names warp-based parallelism, not arbitrary subwarp partitioning. Group identity and physical warp membership must be explicit in the runtime mapping. Merely giving a group a size divisible by 32 is insufficient: its members must be complete physical warps, and each group's scratch memory must be distinct from other groups'.

A reduce still removes exactly one dimension, the last in the producer's `as` order. A two-dimensional reduction is expressed as two chained reduces. Whole-warp scheduling simplifies each individual collective; it does not make different outputs or successive reductions share one unindexed scratch buffer.

**Resolved — whole-warp flat groups only.** The preset table below uses only flat thread groups whose size is a whole number of physical warps (a multiple of 32 threads). A 2-D product-coordinate layout (for example `16 × 16`) is **not** a valid thread-group shape in v0: it does not allocate complete physical warps to groups, so it is excluded from the presets. Instead, every group is a 1-D shard of N warps, and coverage of an iteration dimension always comes from the group's strided loop ([Section 5.7.4](#57-interpretation)). A `(warp, lane)` → multidimensional coordinate mapping may be introduced in a later milestone, at which point 2-D presets can be re-added deliberately; disjoint warp allocations alone do not define coverage of a Cartesian-product iteration space.

#### 5.7.3 Parallelism Property

An `iter` is a conceptual loop. Its parallelism property specifies how it is materialized:

| Property | Meaning |
|---|---|
| `Sequential` | A sequential loop. This is the default and is always correct. |
| `Thread(ThreadK)` | Parallel within a CTA using thread group `ThreadK`. Assignment is currently **per warp**; per-warp-group assignment is planned. |
| `CTA(BlockJ, ThreadK)` | Parallel across CTAs using block group `BlockJ`, and within each CTA using thread group `ThreadK`. |

#### 5.7.4 Interpretation

Every loop is a **tile-between loop nest plus a tile-inside loop nest**. A parallelism assignment bridges the scheduling strategy and the loop structure that the backend generates.

Example: a map with three tiled iterations, `@k` (tile T1), `@m` (tile T2), and `@n` (tile T3).

**(1) `iter(@k) -> CTA(Block0, Thread0)`**

- Every map needs **at least one CTA assignment**, because CUDA cannot run without CTAs. A single-CTA assignment is always possible but useless.
- A CTA assignment is always **two-level**:
  1. The dimension is split into tiles, and each tile is owned by a CTA in block group `Block0`.
  2. Within the CTA, the elements of that tile are covered by thread group `Thread0`.
- In the backend, the block index selects the tile (stride `T1`), and `Thread0` covers the tile with a **strided loop**. Because the loop is strided, correctness does not depend on whether `T1` is smaller than, equal to, or larger than the size of `Thread0`.

**(2) `iter(@m) -> Thread(Thread1)`**

Within every CTA, thread group `Thread1` covers this iteration space concurrently, one tile after another.

**(3) `iter(@n) -> Sequential()`**

Each thread covers this iteration space alone, as a sequential loop.

#### 5.7.5 Assignment Rules

1. Each map has **at least one** CTA assignment.
2. All other iterations default to `Sequential`.
3. Iterations are translated **independently**, iteration by iteration rather than scheduling the whole loop nest tree (LNT) at once, which keeps codegen simple.
4. `reduce` must always be lowerable, whatever parallelism is assigned to the reduced dimension (see [Section 5.10](#510-reduce-lowering)).

#### 5.7.6 Static Strategy Selection (v0)

v0 assigns parallelism from a **fixed table of preset strategies**. There is no search and no cost model. The pass classifies the root map of each statement cluster, picks the **first matching preset**, and instantiates it.

The selector sits behind one interface, so a later optimizer (ILP/LP or another combinatorial solver) can replace the table without touching any lowering pass:

```
select(const StatementCluster&) -> Assignment   // parallelism + thread-group sizes + tile sizes
```

**Classification inputs**

| Symbol | Meaning |
|---|---|
| `P` | Parallel dimensions: induction variables that index the final `store`. |
| `R` | The reduced dimension of the first `reduce` (empty for map-only clusters). |
| `p_c` | The last `P` dimension in `as` order (the innermost parallel loop). |
| `p_o` | The first `P` dimension in `as` order other than `p_c` (the outermost parallel loop). |
| `n_in` | Number of tensor inputs of the root map. |
| `ext(x)` | Static extent of dimension `x`, if known at compile time. |

**Presets (first match wins)**

| # | Name | Match | Assignment (unlisted dims: `Sequential`) | Thread groups (whole warps) | Tile sizes |
|---|---|---|---|---|---|
| S1 | Elementwise 1-D | `R = ∅`, `\|P\| = 1` | `p_c → CTA(Block0, Thread0)` | `Thread0 = 8` (256 threads) | `T(p_c) = 1024` |
| S2 | Elementwise N-D | `R = ∅`, `\|P\| ≥ 2` | `p_o → CTA(Block0, Thread0)`; `p_c → Thread(Thread1)` | `Thread0 = 1` (32 threads); `Thread1 = 7` (224 threads) | `T(p_o) = 32`, `T(p_c) = 224` |
| S3 | Matmul-like | `R ≠ ∅`, `\|P\| ≥ 2`, `n_in ≥ 2`, two inputs share `R` | `p_o → CTA(Block0, Thread0)`; `p_c → Thread(Thread1)` | `Thread0 = 2` (64 threads); `Thread1 = 6` (192 threads) | `T(p_o) = 64`, `T(p_c) = 192`, `T(R) = 16` |
| S4 | Short reduction | `R ≠ ∅`, `\|P\| ≥ 1`, `ext(R) ≤ 256` | `p_c → CTA(Block0, Thread0)` | `Thread0 = 8` (256 threads) | `T(p_c) = 256`, `T(R) = ext(R)` |
| S5 | Row reduction | `R ≠ ∅`, `\|P\| ≥ 1` | outermost `P → CTA(Block0, Thread0)`; `R → Thread(Thread1)` | `Thread0 = 1` (32 threads, unit tile: 1 active lane); `Thread1 = 7` (224 threads) | `T(P) = 1`, `T(R) = 224` |
| S6 | Degenerate | `P = ∅` | implicit unit iteration `iter(@u/1) → CTA(Block0, Thread0)`; `R → Thread(Thread1)` if present | `Thread0 = 1` (32 threads, unit tile: 1 active lane); `Thread1 = 7` (224 threads) | `T(R) = 224` |

Tile sizes follow the loop table in [Section 5.9](#59-iteration-lowering): a `cta_for` tile is owned by one CTA and covered by its thread group, and a `thread_for` tile is owned by one thread. In S3, for example, each CTA computes a 64 × 192 output tile per step; each thread covers one output point and accumulates over the whole reduced dimension. Per-thread multi-element accumulator tiles (the classic matmul register tile) arise whenever a tile size exceeds its group's coverage — for example `T(p_o) = 128` with a 2-warp group gives each thread 2 rows — and become first-class when a 2-D `(warp, lane)` coordinate mapping is defined in a later milestone.

**Rules common to all presets**

- Every thread group contains a whole number of physical warps (a multiple of 32 threads). No preset uses a sub-warp group or a 2-D product-coordinate group. All presets use 256 threads per CTA (8 warps), partitioned into whole-warp groups.
- `Block0` always takes 100% of the grid. The grid size is `ceil(ext(CTA dim) / T)`, computed at launch.
- Tile sizes and thread-group sizes are **compile-time constants**. Extents may be dynamic and are handled by the remainder guard ([Section 2.4](#24-iterations)). A known extent smaller than its tile clamps the tile to the extent, rounded up to the group size.
- No preset assigns `CTA` to a reduced dimension. A full reduction (S6) therefore runs on a single CTA. This is correct but slow; a split-kernel strategy is future work.
- **Budget fallback:** if the caches chosen in [Section 5.8](#58-cta-cache-allocation) exceed the per-CTA budget, halve `T(R)` first, then drop caches back to global memory. Selection never fails.
- **Known v0 limitation:** grid-level parallelism comes from a single dimension (S2, S3, S5). When `ext / T` is below the SM count, the GPU is underused. 2-D CTA sharding is the first planned extension.

#### 5.7.7 Fused Statement Clusters with Nested Maps

1. Parallelism optimization only applies to the **first reduce** of a statement cluster.
2. Fusion can nest one map inside another, so one LNT can sit inside another LNT. A structure with more than one LNT is a **loop nest cluster**. Its root, created by the first map, is the **root LNT (RLNT)**.
3. A nested map **does not reassign** the parallelism of its enclosing loops. Re-mapping within a small window is possible in principle but is out of scope for now. Instead, it takes a *maximal prefix assignment*:
   1. Part of the nested map's loop tree coincides with the RLNT. This can be determined during the pass from the currently active iteration objects.
   2. Let **E** be the set of parallelism objects assigned in the RLNT.
   3. Let **D** be the set of parallelism objects already used by the loops that enclose the nested map.
   4. The nested map binds the unused set **E − D** to its own dimensions. If E − D is empty, all of its loops are sequential.
   - A parallelism object only defines a thread grouping, so it can be bound to any dimension and tile. All threaded accesses use strided loops, which preserves correctness:

```c++
for (int i = 0; i < M /* dynamic */; i += num_threads_in_group) {
  int idx = i + thread_index_in_group;
  ...
}
```

#### 5.7.8 Race Freedom

Parallelism assignment cannot introduce data races, because both the program and the assignment are **structural**:

- A `map` at point `x` reads its inputs and produces a value only for `x`.
- `reduce<@r>` combines values only along `@r`, and only through the reducer, whose lowering owns all cross-thread communication.
- An assignment only distributes iteration points among threads. It never changes which point computes which value.

This argument relies on the following invariants. TCA's construction rules guarantee them; the IR verifier checks them after this pass, mainly to catch compiler bugs:

1. **Store indexing.** Each `store` indexes its target with exactly the induction variables that are live (not reduced) at that point, each once, as plain variables (tile offsets allowed). Distinct iteration points therefore write distinct addresses. A store that omits a live variable, such as `store(out[@m])` under `iter(@m) * iter(@k)` with no reduce over `@k`, is an implicit reduction and is rejected; it must be written as a `reduce`.
2. **Reduced dimensions go through the reducer.** Values along a reduced variable meet only inside a `Reducer`.
3. **In-kernel data flows through pipes.** Within a fused kernel, producer values reach consumers through `|`, not through memory (see [Section 5.4](#54-loop-fusion)).
4. **Shared caches are written only by the runtime** `loadtile`/`storetile` variants ([Section 5.11](#511-cache-lowering)).
5. **Collectives run at CTA-uniform scope.** Calls that contain a CTA barrier (cooperative `loadtile`/`storetile`, and `Combine` of the Thread and CTA reducers) must be executed by every thread of the CTA the same number of times. Violating this is not a race but a deadlock (`__syncthreads()` in divergent code). Codegen guarantees it by emitting `thread_for` loops with CTA-uniform trip counts and predicated tails ([Section 5.12](#512-code-generation)).

### 5.8 CTA Cache Allocation

**New functions:** `loadtile` and `storetile` mark **tiled** loads and stores. The loads and stores produced by the hoisting rule become `loadtile`/`storetile` and are lowered after loop tiling is emitted.

**Budget:** SMEM plus fragment (register) allocation has a per-CTA budget, for example 64 KB.

> Note: more than 48 KB of shared memory per block requires dynamic shared memory with an explicit opt-in (`cudaFuncSetAttribute`).

CTA-local caches apply to **tiles only**.

**Candidate selection**

1. Start with an empty candidate set for the map's lambda operands.
2. For each **unordered pair** of map input operands that **share at least one dimension**, add both operands to the candidate set.

**Placement (greedy)**

- Limits: at most **2 cached operands** per map (the user may supply many), and at most **2 dimensions** per cached operand.
- Work from the **innermost** `iter` outward, maximizing reuse for the current map.
- For each candidate:
  - Use the hoisting rule to find the outermost valid `iter` scope.
  - Choose the storage class from the parallelism of the covered iterations:
    1. All covered iterations are `Sequential` → **FRAG** (registers).
    2. Otherwise → **SMEM**.

  > Note: register arrays stay in registers only when all indices are compile-time constants after unrolling; otherwise they spill to local memory. FRAG placement should require fully unrolled inner tiles.

**Rewrite:** insert the cache load/store statements and annotate the storage class with a suffix (`/gmem`, `/smem`, `/frag`):

```
%cache/frag : tensor<...> = loadtile(%...)
```

### 5.9 Iteration Lowering

**Pass ordering:** iteration lowering, including tile-loop factoring, runs **before** reduce lowering ([Section 5.10](#510-reduce-lowering)). During this pass `reduce` stays an abstract operation, so the reducer can be placed once the final loop structure is known.

Before this pass, every `reduce` is annotated with the induction variable it reduces, written `reduce<@k>`. The variable is the last one in the producing map's `as` clause. Tiling splits that variable into two loops, so "the innermost dimension" no longer identifies it unambiguously.

#### 5.9.1 Loop Nest Order

The loop order is fully determined by the root map's `as` clause. For `as (@v0, @v1, ..., @vn)`:

1. **Tile-between loops** are emitted outermost, in `as` order: `@v0_o, @v1_o, ..., @vn_o`.
2. **Tile-inside loops** follow, **in the same order**: `@v0_i, @v1_i, ..., @vn_i`.
3. A dimension with tile size 1 has a trivial tile-inside loop, which is removed.

For the matmul with `as (@m, @k, @n)`, the nest is:

```
@m_o -> @k_o -> @n_o -> @m_i -> @k_i -> @n_i
```

The compiler never reorders loops in v0. Every later decision derives from this fixed order: reducer placement and the element-walk brackets ([Section 5.10](#510-reduce-lowering)), and the hoisting level of caches ([Section 5.8](#58-cta-cache-allocation)). Under the R3 element-bracket model the reducer state is a scalar per output element — the non-reduced inside loops between the reduced dim's tile-between loop and `Reduce` become the uniform element walk, and no accumulator tile exists.

The example below uses `as (@m, @k)` with `@k` reduced.

Each `iter` is lowered according to its parallelism property into one of three loop nodes: `seq_for`, `thread_for`, and `cta_for`.

```
// Input. @m: Thread parallel; @k: Sequential
iter(@m) * iter(@k) -> {
  %valmap = map(...) as (@m, @k);
  %valred = reduce<@k>(%valmap);
}

// After loop-type lowering
thread_for(@m, T_m, <thread group>) -> {
  seq_for(@k, T_k) -> {
    %valmap = map(...);
    %valred = reduce<@k>(%valmap);
  }
}
```

**Factoring out tile loops**

The current IR becomes the **tile-inside** loop nest, and a matching **tile-between** loop nest is added around it with new induction variables:

| Loop type | Tile-between (outer) | Tile-inside (inner) |
|---|---|---|
| `cta_for` | `cta_for` | `thread_for` |
| `thread_for` | `thread_for` | `seq_for` |
| `seq_for` | `seq_for` | `seq_for` |

```
thread_for(@m_o(step=T_m)/M, <thread group>) -> {
  seq_for(@k_o(step=T_k)/K) -> {
    // Original body. Only the bounds of the original induction variables
    // change (@m -> @m_i/T_m, @k -> @k_i/T_k). Objects that reference these
    // variables, such as loadtile/storetile, remain valid.
    seq_for(@m_i/T_m, offset=@m_o) -> {
      seq_for(@k_i/T_k, offset=@k_o) -> {
        %valmap = map(...);
        %valred = reduce<@k>(%valmap);
      }
    }
  }
}
```

### 5.10 Reduce Lowering

A reduce is materialized as a specialized runtime object. Scalar interface names are illustrative; tiled accumulator state additionally uses slot indices.

```c++
template <typename T>
class Reducer {
 public:
  void Init();
  void Reduce(T value);
  T Combine(); // every participating thread receives the final result
};
```

The specialization depends on the reduced dimension: **Sequential**, **Thread (warp-based)**, or **CTA (global)**. Each parallel `Combine` returns the same final value to every thread participating in the **same logical reduction**. Different output groups receive their respective results, not one CTA-wide scalar.

**(1) Sequential.** Accumulate valid elements locally and return the accumulator. No collective synchronization is needed.

**(2) Thread (warp-based).** Each logical output is reduced by N complete physical warps. All lanes participate in the collective even when their strided input ranges are empty. The runtime first forms one partial per warp, then implements either of these equivalent broadcast contracts:

- **Leader combine and broadcast:** write group-local warp partials to shared memory; synchronize; one group leader combines them and writes the result to the group's result slot (which may reuse its partial slot 0); synchronize again; every participating thread reads that slot.
- **Replicated local combine:** write group-local warp partials; synchronize; every participating thread combines those same partials locally in the same order and returns the result. Synchronize before the shared buffer is reused.

Scratch storage is scoped to a logical output group and accumulator slot, for example `partials[group][slot][warp_in_group]`. With one group and one scalar slot this becomes a simple N-element array. Runtime-owned barriers follow their declared participation scope; a CTA barrier still requires all CTA threads to call it uniformly.

Whole-warp allocation allows a full shuffle mask only when **all physical lanes execute the shuffle**. Out-of-bounds lanes remain participants but contribute no logical element. A correct shuffle tree also avoids consuming undefined out-of-range shuffle sources.

**Initialization semantics.** The user initializer is applied once per logical output, not once per thread or warp. For an identity initializer (such as zero for sum), private accumulators can use that identity. For a general associative/commutative combiner without a declared identity, private partials require validity tags so empty participants contribute nothing. The final combine applies the user initializer once; an empty reduction returns it. This matters for custom lambdas and nonzero initializers.

**(3) CTA (global).** Participating CTAs publish their output-group/slot partials, coordinate globally, and obtain the same final value on every participating thread. The runtime can combine once and broadcast or perform an identical local combine over published partials. Global scratch storage cannot be reused until all readers have finished.

Broadcast does not replace global synchronization. An implementation using `cooperative_groups::grid_group::sync()` still requires a supported cooperative launch, a residency-limited grid, and uniform collective invocation across the grid. A global scratch buffer must be allocated before use; its ownership and launch-time sizing belong to the execution interface. Alternatively, a multi-launch reduction requires an explicit physical-kernel execution plan.

**v0 scheduling policy:** no preset assigns CTA parallelism to a reduced dimension. S6 runs a full reduction on **one CTA**, not multiple CTAs. Global reduction support can therefore be deferred without blocking the preset-based implementation.

**Result availability versus store ownership.** Broadcasting a reduction result makes it available everywhere in its participant group; it does not give every participant permission to store to the same output address. The lowering must preserve one owner per logical store, using an ownership predicate or a redistribution of distinct output points. Pure fused lambdas may run on replicated values; their memory writes must still be uniquely owned. Ownership also needs to be carried through chained reductions so replicas are not counted as additional input elements. The concrete ownership representation is an implementation decision still to be specified.

**IR lowering of reduce**

Reduce lowering runs on the already tiled loop structure produced by [Section 5.9](#59-iteration-lowering). For `reduce<@r>`, where `@r` has been split into `@r_o` (tile-between) and `@r_i` (tile-inside) — **as amended by the R1/R2/R3 ruling (2026-10-23): per-output-element brackets over a UNIFORM element walk, scalar-only reducer protocol, compiler-owned staging**:

1. **Element walk (R3).** Every loop of a *non-reduced* dimension between `@r_o` and the `Reduce` call (here `@m_i`) DISSOLVES as a per-thread walk and is replaced by a fresh **uniform sequential element loop** (init 0, step 1, the dissolved loop's own clamped bound) placed at `@r_o`'s position; `@r_o` and `@r_i` move inside it. The uniform walk is what keeps `Combine`'s CTA barrier uniformly reached ([Section 5.12](#512-code-generation)): every thread executes the same number of element iterations, and threads whose reduced-dim strips saturate to zero trips contribute the identity initializer.
2. **Bracket placement (the bracket law).** With `X = @r_o` (the tile-between loop) and `Y = @r_i` (the tile-inside loop): `Init` sits in **X's parent** — the loop enclosing `X` — as the statement **immediately before `X`** (the reduction spans ALL rounds of `X`); `Reduce` is **inside `Y`**, per element (any other reduced-dim walks nest at `Y`'s level); `Combine` is the statement **immediately after `X`'s body closes**, in X's parent, paired with `Init` — one reduction per output element, spanning all tiles of `@r`, never per-tile partials, and never per `X`-round. For multi-dim outputs the output-element loops wrap the bracket from OUTSIDE: the **innermost output-element loop IS X's parent** — with ≥2 dissolved between dims the element loops nest outer→inner in dim order and the bracket sits at the innermost, one bracket per output element tuple. The T==1 form is the same law over the single `K` loop: `Init` immediately before it, `Reduce` inside it, `Combine` immediately after, and the fused consumers (point 4) follow `Combine` in that same block so the emitted `Combine` local and its consumers share one lexical scope.
3. **Reducer state (R1).** The state is a **scalar accumulator** — `Reduce(value)` folds one element per call and `Combine()` returns `T`. There is no accumulator tile and no slot index: the element is the loop variable of the uniform walk. The warp/global construction carries `(init, combiner, ThreadMapping, staging row)`; the staging row is one `#warps × sizeof(T)` allocation in the compiler's dynamic-SMEM pool at a disjoint offset, handed to the runtime as a `SmemView` — the runtime keeps NO static shared scratch.
4. **Fused consumers.** Operations that consume the `Combine` value are placed after `Combine` inside the element loop, their non-reduced coordinates rebuilt onto the element dimvar. The per-element output store is **ownership-gated** by the runtime's mapping query `is_first_in_cta()` — one owner per output element ([Section 5.12](#512-code-generation)); every other thread computes the same value and stores nothing.

```
thread_for(@m_o(step=T_m)/M, <thread group>) -> {
  %staging = smem<#warps * sizeof(T)>();    // R1: compiler-pool staging row
  %reducer = <specialized Reducer>(init, combiner, thread_map, %staging);

  seq_for(@m_e/T_m) -> {                    // R3: UNIFORM element walk (dissolves @m_i)
    %reducer.Init();                        // per-element bracket opens
    thread_for(@k_o(step=T_k)/K) -> {       // tile-between loop of the reduced @k
      seq_for(@k_i/T_k, offset=@k_o) -> {   // mirror threads saturate to 0 trips
        %valmap = map(...);
        %reducer.Reduce(%valmap);           // R1: scalar protocol — value only
      }
    }
    %valred = %reducer.Combine();           // returns T: every participant receives it
    if (thread_map.is_first_in_cta())       // R2: single-owner store
      map(%valred, lambda(v): v * scale) as (@m) | store(%output[...], %0);
  }
}
```

**Consequences**

- `Combine` runs **once per output element**, after every tile of `@r`. For the Thread and CTA specializations, the shuffle tree and `__syncthreads()` are paid once per element, not once per tile element.
- If `@r` is thread-parallel, its tile-between loop is a `thread_for` and its tile-inside loop is a `seq_for` ([Section 5.9](#59-iteration-lowering)). Each thread's private scalar accumulator persists across both loops inside the element bracket; the warp/CTA combine happens once per element afterwards.
- Threads that own no element of a reduced-dim strip see it saturate to zero trips (the arithmetic mirror gate: the strip offset carries `+ index(ctaThreadDim) * extent` of the independent dim, the FULL extent), so their `Reduce` never fires and their accumulator holds the identity initializer.
- When `@r` is untiled (`T = 1`) the element set is empty and the rule reduces to the simple form: `Init` before the `@r` loop and `Combine` after it, in the bracket loop's parental block.
- The staging row is counted against the per-CTA SMEM budget in [Section 5.8](#58-cta-cache-allocation) (dynamic pool, disjoint offsets in v0).

### 5.11 Cache Lowering

Each `loadtile`/`storetile` is lowered to a call to a **variant from the C++ runtime library**. Codegen emits the buffer declaration and the call. The runtime variant implements the copy and **all synchronization**; codegen never emits barriers for caches.

1. Emit the SMEM or FRAG declaration, sized by the tile shape. A variant may request extra space, for example for padding.
2. Choose the variant from the enclosing tile loop type (names are illustrative):

| Enclosing scope | Storage | Variant | Behavior |
|---|---|---|---|
| `cta_for` / `thread_for` | SMEM | `tca::rt::LoadTileCta<...>` | All threads of the CTA copy cooperatively, using `cp.async` or TMA where available. |
| `seq_for` | FRAG | `tca::rt::LoadTileThread<...>` | Per-thread copy into registers. No synchronization. |

`storetile` has matching variants.

**Runtime contract** (what the compiler relies on)

- **Visibility.** When a cooperative `loadtile` returns, the whole tile is visible to every thread of the CTA.
- **Reuse safety.** A cooperative `loadtile` that refills a buffer first waits until all reads of the previous contents are finished (write-after-read). Double buffering can later be added inside the runtime without IR changes.
- **Store ordering.** A cooperative `storetile` waits until all writes to the tile are finished before copying it out.
- **Bounds.** The runtime checks logical indices before accessing global tensor operands. Invalid tile slots may contain unspecified padding, but map invocation is suppressed for invalid points and reduce skips their contributions. User lambdas see only valid values, not indices or padding. These predicates never suppress collective `Init`/`Combine` calls or required cooperative tile calls.
- **Uniform call.** Every thread of the CTA calls a cooperative variant, the same number of times (invariant 5 in [Section 5.7.8](#578-race-freedom)).

At this point every `loadtile`/`storetile` already sits inside a tiled loop, because iteration lowering has run.

### 5.12 Code Generation

After lowering, emitting C++ is straightforward. `thread_for` and `cta_for` become block- or thread-strided loops. The generated code wraps all loops in runtime iterator objects:

```c++
for (auto it = <cta/thread/seq iterator>(); it.Has(); it.Next()) { ... }
```

For illustration only, using CUDA's default partitioning instead of our sharding:

```c++
// cta_for: each CTA owns one tile per step
for (int base = blockIdx.x * T; base < M; base += T * gridDim.x) {
  // tile = [base, min(base + T, M))
}

// thread_for: CTA-uniform trip count; out-of-range threads are predicated, not exited,
// so every thread reaches collective runtime calls (Section 5.7.8, invariant 5)
for (int base = 0; base < T; base += blockDim.x) {
  int  idx    = tile_base + base + threadIdx.x;
  bool active = (base + threadIdx.x < T) && (idx < M);
  // body runs for all threads; `active` predicates its loads, stores, and Reduce calls
}
```

### 5.13 IR Node Suite

The single-tier IR (Section 5.1) is one module, `internal/ir`, organized into submodule files by node family. Kinds are append-only with a versioned enumeration (`KindVersion`), so existing constants never renumber; the pass manager's legality windows fail closed on kinds no window knows.

| Family | Kinds | Node structs in |
|--------|-------|-----------------|
| FamilyScalarExpr | `const`, `param`, `binop`, `unaryop`, `select`, `cast`, `call` | `construct.go` |
| FamilyAbstract | `iter`, `itercluster`, `map`, `reduce`, `store`, `return`, `inputparam`, and the value-level `dimvar`/`bdimvar` | `construct.go`, `dimvar.go` |
| FamilyTileOp | `loadtile`, `storetile` | `tile.go` |
| FamilyLoop | the lowered `for` loop | `loop.go` |
| FamilyCollective | the runtime-object constructions (`runtimekernelinfo`, the grid/thread mappings, the three reducer specializations) and their method calls/attribute reads | `node_runtime.go` |

The lowered suite is what iteration lowering (Section 5.9) and cache lowering (Section 5.11) produce into the same node suite:

- **`dimvar`** — an induction variable as a value object: operands `[dim, init, step]`, identity is the `(init, step)` tuple, never the name. Printed `@x(init=0, step=1)`.
- **`bdimvar`** — the bounded form `@x/M` (Section 2.3): operands `[iv(dimvar), max]`.
- **`loadtile` / `storetile`** — cooperative tile accesses: operands `[tensor, bivs...]` and `[value, tensor, bivs...]`; the tensor operand must resolve to a tensor-typed `InputParam` (the verifier fails closed otherwise).
- **`forloop`** — the explicit lowered loop: operands `[iv(dimvar), bound, body]`; the body is a single child subgraph root.

**IV/BIV relations** (formal definitions in `doc/optana/induction_variable.md`): two IVs are equal iff their `(init, step)` tuples are equal — the same node, or two constant pairs with equal values; a constant is never equal to a non-constant. A BIV is `((iv tuple), Max)`; two BIVs are equal iff their dimvars are equal AND their maxes are equal. The include relation `A ~> B` holds iff `A.max <= B.max` **for normalized IVs only** (`init=0, step=1` as constants); `~>` is not commutative. Belong is its reverse: `A <~ B` iff `B ~> A`. Any non-normalized IV makes both relations false.
