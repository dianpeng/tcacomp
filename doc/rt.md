# TCA Runtime Library — Final Design (runtime/)

- **Purpose.** The C++ runtime library every tcacomp-generated kernel links
  against. This is the IMPLEMENTATION-READY design: the single input for
  the library task, consolidating every contract the compiler has already
  pinned. The implementer should not need to read the Go compiler.
- **Readers.** The engineer implementing `runtime/`; reviewers of generated
  code.
- **Sources (all already ruled).** doc/cg.md §3/§7/§8 (the emitted surface),
  doc/runtime_objects.md (the 44-entry contract inventory), doc/tca.md
  §5.10–5.12 (reduction/cache/codegen semantics), doc/optana/loop_tiling.md
  + loop_tile_lower.md (portion grammar, masks, loop primitives), and the
  scalfold reference (`/home/ps/Workspace/cuda/project/scalfold`) for
  mechanics (header-only structure, the function-scope `__shared__` trick,
  host-wrapper style).
- **Conformance targets.** The two committed goldens —
  `internal/cg/cxx/testdata/matmul_store.cu` and `reduce_warp.cu` — are the
  exact call shapes the library must compile and run. Every signature below
  is quoted from them.

## 1. Shape & build

Header-only, the scalfold structure:

```
runtime/
├── CMakeLists.txt          # INTERFACE library target tcacomp_runtime
├── tca/
│   ├── runtime.h           # THE umbrella — the only include generated code uses
│   ├── tensor.h            # host: Tensor<T, Rank> (the launcher contract type)
│   ├── check.h             # host: TCA_REQUIRE / TCA_CUDA_CHECK / Throw
│   ├── launch.h            # host: ValidateLaunch (+ device-attribute queries)
│   ├── mapping.h           # device: KernelInfo, GridMapping, ThreadMapping
│   ├── iterator.h          # device: Seq/Cta/ThrIterator
│   ├── view.h              # device: Layout, TensorView, SmemView, Frag
│   ├── loader.h            # device: LoadTileCta / LoadTileThread
│   └── reducer.h           # device: Seq/Warp/GlobalReducer (scalar protocol)
```

- CMake: `project(tcacomp_runtime LANGUAGES CXX CUDA)`; `add_library(
  tcacomp_runtime INTERFACE)` + `target_include_directories`; consumers add
  `CUDA::cudart`. `CMAKE_CUDA_ARCHITECTURES` is set by the GENERATED
  CMakeLists (pinned 86) — the library itself stays arch-generic where
  possible and guards `cp.async` behind `__CUDA_ARCH__ >= 800`.
- C++20 both languages; `--expt-relaxed-constexpr` is already passed by the
  generated CMakeLists.
- No `.cu` TUs required in v0 (everything header-inline, `__host__
  __device__` where dual-use); explicit instantiations come later if link
  times demand them.
- `runtime.h` includes the CUDA headers (`cuda_runtime.h`,
  `cuda_fp16.h`, `cuda_bf16.h`) and every `tca/*.h` above.

## 2. Design principles (all previously ruled — restated as implementation
constraints)

1. **The runtime owns ALL synchronization and guards.** The compiler emits
   zero branches; iterator `Has()` is the CTA-UNIFORM round walk; per-thread
   /per-CTA validity is enforced inside the views' bounds predicates and
   the loaders' tail clamping ("predicated, not exited" — collective calls
   stay uniformly reached).
2. **Maps are opaque.** GridMapping/ThreadMapping own the portion
   decomposition; the compiler never inlines linearization arithmetic.
3. **The layout seam (cg.md D6).** Every memory access goes through a view
   carrying a layout descriptor; v0 ships `Layout::RowMajor` only. The
   future coalescing/swizzle passes change descriptors, never call sites.
4. **No runtime-static SMEM (R1, 2026-10-23).** The runtime keeps NO
   `static __shared__` scratch — reducer staging is one row of the
   COMPILER's dynamic-SMEM pool, handed to the ctor as a `SmemView<T, 1u>`
   (§4.6). The pool is the only SMEM owner; the launcher sizes it.
5. **Determinism and totality.** No allocation, no host calls on device
   paths; every accessor is O(1).

## 3. The conformance table — every call the goldens make

| Generated call (verbatim shape) | Implementation |
|---|---|
| `tca::rt::KernelInfo kinfo;` | §4.1 |
| `tca::rt::GridMapping grid_map0(kinfo, -1);` / `ThreadMapping thread_map1(kinfo, 4, -1);` | §4.2 |
| `gm.index(0u)`, `tm.dimension(1u)` | §4.2 |
| `tca::rt::SeqIterator(init, step, bound)` / `CtaIterator(gm, dosoc, loc, init, step, bound)` / `ThrIterator(tm, dosoc, loc, init, step, bound)` + `.Has() .Next() .Index()` | §4.3 |
| `tca::rt::TensorView<float, 3u> out_v(out, d0, d1, d2);` + `.Get(c...)` `.Set(v, c...)` | §4.4 |
| `tca::rt::SmemView<float, 1u> lhs_tile(reinterpret_cast<float*>(tca_smem + 0u), 16384u);` + `.Get(i)` | §4.4 |
| `tca::rt::LoadTileCta(lhs_tile, lhs_v, m_base, k_round);` | §4.5 |
| `tca::rt::WarpReducer<float> reducer0(0.0f, RedwarpCombine0, thread_map1, tca::rt::SmemView<float, 1u>(reinterpret_cast<float*>(tca_smem + N), 16u));` + `.Init()` `.Reduce(v)` `.Combine()` → `T` | §4.6 |
| `tca::rt::umin(a, b)` (umax registered, unused yet) | §4.7 |
| `tca::Tensor<float, 3u>& out` + `.Data()` `.Dim(i)` | §5 |
| `TCA_REQUIRE(cond, "msg")`, `TCA_CUDA_CHECK(expr)` | §5 |
| `tca::rt::ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes)` | §5 |

## 4. Device surface

### 4.1 KernelInfo

```c++
struct KernelInfo {
  __device__ KernelInfo() = default;
  __device__ unsigned total_cta() const;             // gridDim.x
  __device__ unsigned total_thread_per_cta() const;  // blockDim.x
};
```

### 4.2 GridMapping / ThreadMapping — the portion decomposition

Both are the same math over different spaces: the CTA space (gridDim.x,
portion grammar Σ=**PortionCTA=10**) and the per-CTA thread space
(blockDim.x, warp portions Σ=**PortionWarp=8**, CTA-relative — warp
allocations never span CTAs).

```c++
struct GridMapping   { __device__ GridMapping(KernelInfo, /*int portions...*/);
                       __device__ unsigned index(unsigned d) const;
                       __device__ unsigned dimension(unsigned d) const; };
struct ThreadMapping { /* same face over threadIdx.x */ };
```

- Constructor portions: each ≥ 1; `-1` (the remainder) only last. `ndim` is
  implied by the argument count (the verifier pinned this on the IR side).
- **Decomposition.** Let `Q = total / kPortion` (CTA: `total/10`; thread:
  `total_warps/8`, extents in warps × 32). Dimension d's extent is
  `portion_d × Q`, the `-1` tail taking `total − (n−1 sums)`. Tuple
  coordinates: decompose the linear id (`blockIdx.x` / `threadIdx.x`)
  row-major over the extents — `coord_d = (id / stride_d) % extent_d`.
  `index(d) = coord_d`; `dimension(d) = extent_d`. Pure arithmetic — no
  `__ballot_sync`, CPU-testable.
- **The FLAT index() — ruling 2026-10-23.** `index()` with NO argument
  returns the map holder's ABSOLUTE FLATTENED 1-D index:
  `ThreadMapping::index()` is the thread's linear id in the CTA (the
  `threadIdx.x` analog); `GridMapping::index()` the CTA's linear id (the
  `blockIdx.x` analog). `index(d)` (the per-dimension tuple coordinate) is
  UNCHANGED. The compiler NEVER inlines `threadIdx.x`/`blockIdx.x` itself —
  the maps stay the opaque seam; lane/warp identity derives from the flat
  form (`index() % 32`, `index() / 32`).
- **Warp/lane queries — v0 surface (ruling addenda 2026-10-23).**
  `ThreadMapping` additionally carries, all pure functions of `index()`:
  `warp_index()` (= `index()/32`), `lane_id()` (= `index()%32`),
  `is_first_in_warp()`, `is_first_in_cta()` (`index() == 0`), and
  `warp_count()` (the CTA's EXISTING warp count, `total threads/32`,
  obtained THROUGH the map — never a magic constant). These are
  runtime-INTERNAL control flow (reducer staging ownership, fold-existing-
  warps bounds, the staging-extent cross-check); generated code stays
  branch-free — the queries never appear as compiler-emitted conditionals.

### 4.3 Iterators — the uniform round walk

```c++
struct SeqIterator { __device__ SeqIterator(unsigned init, unsigned step, unsigned bound);
                     __device__ bool Has(); __device__ void Next(); __device__ unsigned Index(); };
struct CtaIterator { __device__ CtaIterator(GridMapping, unsigned dosoc, unsigned loc,
                                            unsigned init, unsigned step, unsigned bound); /* + face */ };
struct ThrIterator { __device__ ThrIterator(ThreadMapping, unsigned dosoc, unsigned loc,
                                            unsigned init, unsigned step, unsigned bound); /* + face */ };
```

- ALL THREE walk identically: `base = init; while (base < bound) { body;
  base += step; }`. `Index()` returns the current dimvar value
  (`init + k·step`) — the 0-based round-base identity; the per-thread /
  per-CTA rank (`map.index(d)·T`) enters only as in-body offset terms the
  compiler already emitted as attribute reads.
- **`Has()` is CTA-uniform for the ROUND walks — and PER-THREAD by design
  for the coupled-inside walk (amended, flat-index() ruling 2026-10-23).**
  The between-round walks keep every thread of the CTA agreeing on the trip
  count (this is what makes every collective call uniformly reached). The
  CTA dim's COUPLED tile-inside walk is a thread iterator whose INIT is the
  per-thread flat rank (`thread_map.index()`), step the coupled thread
  count, bound the tile max: threads with `flatRank >= tile_max` take ZERO
  trips (the whole body no-ops) — per-thread trip counts are the
  single-ownership mechanism, not a uniformity violation (nothing
  collective sits inside the coupled walk). Lane/warp identity at such
  sites derives from the flat form: `index() % 32` / `index() / 32`.
- v0 Cta/Thr carry the mapping/dosoc/loc for future ownership-guard
  queries; the walk ignores them.
- The bound expression is the compiler's saturating F1 chain
  (`umin(T, extent − umin(offset, extent))`) — already evaluated per
  iteration by the emitted call.

### 4.4 Layout + the views (the memory shim)

```c++
enum class Layout : unsigned { RowMajor = 0 /* future: swizzle/coalescing */ };

template <typename T, unsigned Rank> struct TensorView {   // GMEM
  __device__ TensorView(T* data, /*unsigned dims[Rank]*/...);
  __device__ T Get(unsigned coords...);      // bounds-PREDICATED: invalid → unspecified value, never a wild access
  __device__ void Set(T v, unsigned coords...);
};
template <typename T, unsigned Rank> struct SmemView {     // pool slice
  __device__ SmemView(T* base, unsigned extents...);       // + Get/Set (in-window by construction; no suppression needed)
};
template <typename T, unsigned Rank, unsigned... Dims> struct Frag { ... };  // per-thread registers, same face
```

- Row-major strides computed at construction; `Get/Set` map logical
  coordinates through the layout descriptor (v0: identity + stride math).
- **Bounds contract (tca.md §5.11):** GMEM views suppress out-of-range
  points (return an unspecified-but-safe value — `T{}`); this is the
  whole-body-no-op MECHANISM for thr-guarded iterations. SMEM/FRAG views
  are indexed in-window by construction and need no suppression.
- `TensorView` construction from a `Tensor` host object never happens on
  device — the launcher unpacks (§5).

### 4.5 The loaders

```c++
// Cooperative CTA fill: ALL threads of the CTA call it the same number of
// times (invariant 5). The runtime owns the copy, ALL barriers, and tail
// clamping. The compiler emits NO synchronization anywhere.
__device__ void LoadTileCta(SmemView& dst, const TensorView& src, unsigned baseCoords...);
// Per-thread fill into a Frag: plain copy loop, no synchronization.
__device__ void LoadTileThread(Frag& dst, const TensorView& src, unsigned baseCoords...);
```

- **Window→axis contract (pinned):** the destination window covers the
  source tensor's **TRAILING `dst.Rank()` axes**. `baseCoords` carries one
  coordinate per source axis IN ORDER — the trailing axes get their round
  bases, the leading axes their fixed coordinates. (tile-lower's tie
  construction produces trailing windows; a future non-trailing window
  would grow an explicit axis list — a compiler-visible signature change.)
- **Enumeration (cooperative):** window elements are striped over the whole
  CTA — `for (i = threadIdx.x; i < windowElems; i += blockDim.x)` — each
  element's window coordinates decompose row-major; the source point is
  `baseCoords` with each trailing axis offset by its window coordinate;
  out-of-range source points are clamped/skipped (the tail-clamp contract:
  a partial tile is filled; padding content is unspecified but SAFE).
- **Ordering contract:** barrier AFTER the copy (visibility) and, on
  refill, the write-after-read wait BEFORE (reuse safety). v0: a
  `__syncthreads()` pair is correct and sufficient (uniform-call
  guaranteed); double buffering is a future internal change.
- **`cp.async`:** optional fast path under `__CUDA_ARCH__ >= 800` with a
  plain-copy fallback — an internal decision, invisible to callers.

### 4.6 The reducers — the scalar protocol (R1/R2 ruling 2026-10-23)

```c++
template <typename T> struct SeqReducer { SeqReducer(T init, T (*fn)(T, T));
                                          void Init(); void Reduce(T v); T Combine(); };
template <typename T> struct WarpReducer { WarpReducer(T init, T (*fn)(T, T),
                                           const ThreadMapping&, SmemView<T, 1u> staging);
                                           /* same method face */ };
template <typename T> struct GlobalReducer { /* v0-unreachable scaffolding, same ctor face */ };
```

- **Scalar-only (R1):** NO Slots template parameter, no slot argument.
  `Reduce(value)` folds ONE element per call; `Combine()` ALWAYS returns
  `T`. Multi-output reductions re-iterate the whole [Init → Reduce-walk →
  Combine] bracket per output element (tca.md §5.10 R3) — the bracket is
  cheap relative to the walk it brackets.
- **Construction carries ALL state** `(init, combiner)` for SeqReducer;
  WarpReducer/GlobalReducer additionally take the mapping and the staging
  view (below). `Init()` is argument-free and re-initializes;
  `Combine()` finalizes ONCE per bracket.
- **SeqReducer:** per-thread `T acc`; no synchronization; `Combine()`
  returns the local accumulator.
- **WarpReducer (the 2026-10-24 protocol, identity-init simplification):**
  1. NO per-thread accumulator in device code. `Init()` resets ALL SMEM
     slots to the declared init single-writer: each warp's first lane
     (`map.is_first_in_warp()`) rewrites its OWN slot
     (`map.warp_index()`); Init is uniformly called by every CTA thread
     (the bracket law), so every slot resets and warps that never Reduce
     hold the identity init.
  2. `Reduce(v)` captures `unsigned mask = __activemask()` at entry and
     runs the `fn` down-tree over the PARTICIPATING lanes INSIDE Reduce:
     every lane in the mask executes each `__shfl_down_sync(mask, part,
     off)` uniformly, discarding the read when its source lane falls
     outside the active prefix (`lane_id() + off < __popc(mask)`). The
     warp partial lands in the first active lane, which folds it into
     its warp's slot: `slot[warp] = fn(slot[warp], warpPartial)` — the
     per-call slot fold, single writer per address. PREFIX ALIGNMENT:
     the compiler's bounds gates produce prefix-aligned active sets (the
     F1 tileMax kills lanes from the top within a warp; the mirror gate
     kills whole warps at the 256-thread/8-warp alignment), so the first
     active lane is lane 0 and the guarded tree folds exactly the active
     lanes' values. Non-prefix masks are out of scope for v0.
  3. `Combine()`: `__syncthreads()` FIRST (the slot writes must be
     visible), then EVERY thread folds all `map.warp_count()` slots in
     ascending warp order from the identity init — the barrier is the
     protocol's ONLY synchronization (tca.md §5.10's
     replicated-local-combine broadcast; the per-element brackets each
     re-read it).
  4. **Staging ownership (R1):** the scratch is ONE ROW of the COMPILER's
     dynamic-SMEM pool — `#warps × sizeof(T)` bytes at a disjoint pool
     offset (D8), passed as `SmemView<T, 1u>(reinterpret_cast<T*>(tca_smem
     + offset), warp_extent)`. The runtime keeps NO `static __shared__`;
     `kMaxWarpsPerCTA` is gone. Warp/lane/count identity is runtime-
     INTERNAL control flow derived from the mapping (§4.2) — generated
     code stays branch-free.
  5. **Host branch:** no warps — the sequential algebra through a
     HOST-ONLY accumulator member (absent from nvcc compilation: neither
     the device nor the host pass carries it); CPU unit tests stand the
     staging row up as a stack array.
- **Initialization semantics:** v0 requires the declared init to be an
  identity for the combiner over partials (0 for `+`). General
  non-identity inits need validity tags (tca.md §5.10) — future work, and
  the interface already carries the init so no signature change.
- **GlobalReducer:** registered scaffolding only (no v0 preset assigns CTA
  parallelism to a reduced dim). Implement as a hard `static_assert`-free
  stub that compiles; enablement needs the multi-CTA scratch/ownership
  plan (tca.md §5.10(3)).

### 4.7 u32 pair

```c++
__device__ unsigned umin(unsigned a, unsigned b);
__device__ unsigned umax(unsigned a, unsigned b);
```

## 5. Host surface

```c++
template <typename T, unsigned Rank> struct Tensor {
  Tensor() = default;                                    // non-owning
  template <typename... Dims> static Tensor From(T* data, Dims... dims);
  T* Data() const;                                       // the device pointer
  unsigned Dim(unsigned axis) const; unsigned Rank() const;
  Layout layout() const;                                 // v0: RowMajor
};

[[noreturn]] void Throw(const char* what);               // std::runtime_error
#define TCA_REQUIRE(cond, what) ...
#define TCA_CUDA_CHECK(expr) ...                          // scalfold CUDA_CHECK: __FILE__:__LINE__ + cudaGetErrorString

void ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes);
```

- `ValidateLaunch` asserts the tiling grammar (gridDim.x a positive
  multiple of PortionCTA=10; blockDim.x a whole-warp multiple with warp
  count a multiple of PortionWarp=8, ≥256 threads) and that `smemBytes +
  the kernel's static scratch` fits the device — queried via
  `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` minus
  the reserved per-block bytes. **NEVER hardcode sm_86's 101,376 B**
  (runtime_objects §1.1; the occupancy-API off-by-1KB bug note).
- The launcher the compiler generates already derives and cross-validates
  every dim from the Tensors; the library adds nothing to that protocol.

## 6. SMEM accounting (one table the launcher asserts)

| Region | Owner | Sizing |
|---|---|---|
| Dynamic pool (tile buffers) | compiler | Σ buffer extents, disjoint v0 offsets (D8); passed as the launch's dynamic-smem bytes |
| Reducer staging rows | compiler | one `#warps×sizeof(T)` row per reducer at a disjoint dynamic-pool offset (R1/D8); the pool sum is the launch's dynamic-smem bytes |

Both must fit the queried per-block ceiling; `ValidateLaunch` is the
assertion point (the static side is compile-time visible in the same TU).

## 7. Numeric & index contracts (inherited, restated)

- Everything index-shaped is **u32**; iterator walks and map reads never
  wrap (0-based rounds, `base < extent` guards).
- The F1 saturating bound chain is the compiler's product; the runtime's
  loaders clamp tails, the views suppress out-of-range points. No guard
  branches anywhere in library code except inside loaders/enumeration
  loops.

## 8. Test strategy

1. **CPU-testable units (plain gtest, no GPU):** portion decomposition
   (index/dimension over sample totals), iterator walks, view stride math,
   the scalar reducer algebra (SeqReducer folds; the WarpReducer host
   branch with a stack-array staging stand-in).
2. **Golden compile conformance:** build the two committed `.cu` goldens
   against the headers — they must compile cleanly (the linker surface is
   the launcher symbols only).
3. **E2E (the stack decision's harness):** `TCACOMP_E2E=1`-gated Go tests;
   content-addressed fixture cache under `os.TempDir()`; run the compiled
   binaries on the A6000; compare against a CPU reference computed in Go;
   `compute-sanitizer` memcheck/racecheck on the same fixtures.
4. Only necessary tests (design.md) — the goldens + one e2e per worked
   example + the CPU unit layer.

## 9. Implementation order

1. `tca/` skeleton + CMake INTERFACE + umbrella header; check.h.
2. Mapping + iterators (+ CPU unit tests of the decomposition math).
3. Views (Layout, TensorView/SmemView/Frag) + umin/umax.
4. Host Tensor + ValidateLaunch (+ device-attribute query).
5. Loaders (plain-copy first; `cp.async` fast path behind arch check).
6. Reducers (Seq, then Warp with the staging protocol; Global stub).
7. Golden-compile conformance + the matmul e2e + the reduce e2e +
   compute-sanitizer pass.

## 10. Out of scope / future (registered, deliberately not blocking)

- Non-RowMajor layout descriptors (the coalescing/swizzle passes).
- `warp_index`/`lane_index`/`is_first` map queries; group-scoped reducer
  staging; non-identity-init validity tags; double-buffered loaders.
- `storetile` variants (v0 is loads-only); GlobalReducer enablement;
  multi-LNT portion sharding; f16/bf16 compute-type policy (scalfold
  `TypeTrait::Compute` precedent — decide when a half fixture exists).
