// tca/mapping.h — KernelInfo + the factorization maps (doc/fix/fix4.md §4.1;
// supersedes the pre-fix4 grammar of doc/rt.md §4.2).
//
// The maps are OPAQUE: they own the ABSOLUTE-EXTENT factorization — each CTA
// dim's extent is that dim's absolute LoC (the number of CTAs sharing the
// dim), each WARP dim's extent is that dim's K_d (the number of warps), all
// extents >= 2 by construction (K=1 dims are pruned upstream; the LoC
// presets start at 2). No share-of-total grammar, no -1 remainder, no total
// division: the launch IS what you factorize (fix4 §1.2).
//
// Multi-LNT launches run the max-over-LNT CTA/warp totals, so an individual
// LNT's shape product can be SMALLER than the launch (fix4 §3.2): surplus
// ranks' coordinates wrap in-shape via the modulo (the decomposition is
// total for ANY flat rank the launch can produce), and their bodies
// whole-body-skip via the compiler's loop guards. Deliberately NO assert
// ties a shape product to the launch total — warp_count()/total() answer
// the DERIVED launch values through KernelInfo instead.
//
// Pure arithmetic — no CUDA intrinsics — so the whole header is
// CPU-testable.
#ifndef TCA_RT_MAPPING_H_
#define TCA_RT_MAPPING_H_

// Host-compilation shim: when compiled by a plain host compiler (the CPU unit
// layer), __host__/__device__ are no-ops. Applied exactly once per TU.
#if !defined(__CUDACC__) && !defined(TCA_HOST_SHIM)
#define TCA_HOST_SHIM 1
#define __device__
#define __host__
#endif

#include <cassert>

namespace tca {
namespace rt {

// u32 pair (doc/rt.md §4.7) — index-arithmetic family lives with the maps.
__host__ __device__ inline unsigned umin(unsigned a, unsigned b) { return a < b ? a : b; }
__host__ __device__ inline unsigned umax(unsigned a, unsigned b) { return a > b ? a : b; }

// KernelInfo — dual-compile pattern so the mapping math is CPU-testable.
// Device: stateless, reads gridDim.x / blockDim.x. Host (plain g++ CPU tests
// or the nvcc host pass): members ctas_/threads_, set by the host-only
// explicit ctor; the host default ctor zeroes them. total_cta() /
// total_thread_per_cta() return the DERIVED launch values (fix4 §4.1) — the
// launch the compiler picked, which multi-LNT launches size at the
// max-over-LNT totals.
struct KernelInfo {
  // Defaulted functions are implicitly __host__ __device__ under nvcc (and
  // the shim no-ops the annotations for plain g++), so no annotation here:
  // annotating a defaulted ctor triggers nvcc warning #20012-D.
  KernelInfo() = default;

#ifndef __CUDACC__
  // Host-only injection point for CPU unit tests.
  explicit KernelInfo(unsigned ctas, unsigned threads) : ctas_(ctas), threads_(threads) {}
#endif

  __host__ __device__ unsigned total_cta() const {
#if defined(__CUDA_ARCH__)
    return gridDim.x;
#else
    return ctas_;
#endif
  }
  __host__ __device__ unsigned total_thread_per_cta() const {
#if defined(__CUDA_ARCH__)
    return blockDim.x;
#else
    return threads_;
#endif
  }

#if !defined(__CUDA_ARCH__)
 private:
  unsigned ctas_ = 0u;
  unsigned threads_ = 0u;
#endif
};

namespace detail {

inline constexpr unsigned kMaxMapDims = 8u;

// Factorization decomposition shared by both maps (fix4 §4.1). `extents` are
// the constructor's ABSOLUTE per-dim extents (CTAs for GridMapping, WARPS
// for ThreadMapping), copied VERBATIM — no total division, no remainder
// branch, no unit multiplier. Strides are row-major over the extents (last
// dim stride 1).
struct Decomposition {
  unsigned ndim = 0u;
  unsigned extents[kMaxMapDims] = {};  // ABSOLUTE per-dim extents, verbatim
  unsigned strides[kMaxMapDims] = {};  // row-major strides over the extents

  // n == 0 is LEGAL (fix4 T5, the zero-extent ThreadMapping admission): a
  // schedule with zero WARP dims launches ONE warp and still needs the map
  // object (the flat query, lane_id, is_first_in_cta, ThrIterator) — with
  // ndim 0 no coord(d) call is possible and warp_count()/total() are
  // kinfo-derived, so the stride loop is skipped entirely.
  __host__ __device__ void build(const unsigned* shape, unsigned n) {
    assert(n <= kMaxMapDims);
    ndim = n;
    for (unsigned d = 0u; d < n; ++d) {
      assert(shape[d] >= 2u);  // fix4 §4.1: all extents >= 2 by construction
      extents[d] = shape[d];
    }
    if (n == 0u) {
      return;  // no strides to derive; coord() is uncallable by construction
    }
    // Row-major strides over the extents (last dim stride 1).
    strides[n - 1u] = 1u;
    for (unsigned d = n - 1u; d-- > 0u;) {
      strides[d] = strides[d + 1u] * extents[d + 1u];
    }
  }

  // Row-major coordinate of `id` over the extents. Total for ANY u32 rank:
  // the modulo wraps surplus ranks in-shape (fix4 §3.2), and extents >= 2
  // keep every divide well-defined.
  __host__ __device__ unsigned coord(unsigned d, unsigned id) const {
    return (id / strides[d]) % extents[d];
  }
};

}  // namespace detail

// GridMapping — the CTA-space map. Each constructor extent is its dim's
// ABSOLUTE LoC (the CTA count owning that dim; all >= 2); the flat rank is
// the CTA's blockIdx.x. index(d) = this CTA's rank in dim d (CTA-uniform);
// dimension(d) = the extent verbatim; index() (no-arg) = the flat CTA rank
// (blockIdx.x analog, ruling 2026-10-23).
class GridMapping {
 public:
  // ndim explicit, then ndim ABSOLUTE extents (CTA counts per dim).
  template <typename... E>
  __host__ __device__ GridMapping(const KernelInfo& kinfo, unsigned ndim, E... loC)
      : kinfo_(kinfo) {
    static_assert(sizeof...(E) >= 1, "GridMapping needs at least one extent");
    const unsigned shape[] = {static_cast<unsigned>(loC)...};
    assert(ndim == static_cast<unsigned>(sizeof...(E)));
    dec_.build(shape, ndim);
  }

  __host__ __device__ unsigned dimension(unsigned d) const { return dec_.extents[d]; }
  __host__ __device__ unsigned index(unsigned d) const { return dec_.coord(d, linear_id()); }
  // The FLAT form (ruling 2026-10-23): no-arg index() is this CTA's absolute
  // 1-D id — the blockIdx.x analog. index(d) (the tuple coordinate) is the
  // reverse stride-shape decomposition of it.
  __host__ __device__ unsigned index() const { return linear_id(); }

#ifndef __CUDACC__
  // CPU unit-test hook: stand in for blockIdx.x.
  void set_linear_id(unsigned id) { id_ = id; }
#endif

 private:
  __host__ __device__ unsigned linear_id() const {
#if defined(__CUDA_ARCH__)
    return blockIdx.x;
#else
    return id_;
#endif
  }

  KernelInfo kinfo_{};  // the derived launch values (gridDim), kept for map-side queries
  detail::Decomposition dec_{};
  unsigned id_ = 0u;
};

// ThreadMapping — the per-CTA thread-space map (warp allocations never span
// CTAs; ranks are CTA-relative). Each constructor extent is its dim's K_d —
// an ABSOLUTE warp count (all >= 2 when present), in WARP units. ThreadMap
// carries WARP dims ONLY (fix4 §1.3 dimensionality ruling): the CTA dim's
// coupled inside walk uses the no-arg flat thread rank and needs NO map
// slot. index(d) decomposes the WARP RANK (index()/32) row-major over the
// warp extents — WARP units, not threads; lane_id() resolves the lane
// inside the warp.
//
// The ZERO-EXTENT form (ndim 0, no extents) is LEGAL (fix4 T5): a legal
// schedule with zero WARP dims — e.g. a reduced dim forced CTA/SEQ while
// ReducedDimWarp=false — launches ONE warp (the empty product of warp
// K_d = 1) and still needs the map object: the flat index() query,
// lane_id(), is_first_in_cta(), and the coupled ThrIterator all read it.
// No coord(d) call is possible at ndim 0 (there is no d), and
// warp_count()/total() are kinfo-DERIVED launch values, unaffected.
class ThreadMapping {
 public:
  // ndim explicit, then ndim ABSOLUTE extents (warp counts per dim); the
  // empty pack (ndim 0) is the one-warp admission above.
  template <typename... E>
  __host__ __device__ ThreadMapping(const KernelInfo& kinfo, unsigned ndim, E... warpsPerDim)
      : kinfo_(kinfo) {
    static_assert(sizeof...(E) >= 0, "ThreadMapping admits the zero-extent form");
    assert(ndim == static_cast<unsigned>(sizeof...(E)));
    if constexpr (sizeof...(E) > 0) {
      const unsigned shape[] = {static_cast<unsigned>(warpsPerDim)...};
      dec_.build(shape, ndim);
    } else {
      dec_.build(nullptr, 0u);
    }
  }

  __host__ __device__ unsigned dimension(unsigned d) const { return dec_.extents[d]; }
  // WARP units: first divide the flat rank by 32 (the warp rank), THEN
  // row-major-decompose it over the K shape.
  __host__ __device__ unsigned index(unsigned d) const { return dec_.coord(d, warp_index()); }
  // The FLAT form (ruling 2026-10-23): no-arg index() is the thread's
  // absolute flattened 1-D id in the CTA — the threadIdx.x analog. lane/warp
  // identity DERIVES from it (index()%32, index()/32); the compiler never
  // inlines threadIdx.x itself — the map stays the opaque seam.
  __host__ __device__ unsigned index() const { return linear_id(); }

  // Warp/lane identity and count, first-class on the map (ruling addenda
  // 2026-10-23): pure functions of index(); runtime-internal control flow
  // (generated code stays branch-free — these never appear as
  // compiler-emitted conditionals).
  __host__ __device__ unsigned warp_index() const { return index() / 32u; }
  __host__ __device__ unsigned lane_id() const { return index() % 32u; }
  __host__ __device__ bool is_first_in_warp() const { return lane_id() == 0u; }
  __host__ __device__ bool is_first_in_cta() const { return index() == 0u; }
  // DERIVED launch value (fix4 §4.1): the CTA's warp count = the launch's
  // blockDim.x / 32 — NOT the shape product. Multi-LNT launches run surplus
  // warps whose coordinates wrap in-shape and whose bodies the compiler's
  // guards skip (§3.2); no assert ties the shape product to the launch
  // total. The fold-existing-warps bound and the staging-extent cross-check
  // read it here, never a magic constant.
  __host__ __device__ unsigned warp_count() const { return kinfo_.total_thread_per_cta() / 32u; }
  // DERIVED launch value (fix4 §4.1): the map's FLAT thread space (fix3
  // R-J/D-n) — the coordination total the cooperative loader stripes rows
  // over, every thread of the launch's blockDim (the compiler never inlines
  // blockDim arithmetic). NOT the shape product: surplus threads coordinate
  // through the guards, and the loader must still stripe over the whole CTA.
  __host__ __device__ unsigned total() const { return kinfo_.total_thread_per_cta(); }

#ifndef __CUDACC__
  // CPU unit-test hook: stand in for threadIdx.x.
  void set_linear_id(unsigned id) { id_ = id; }
#endif

 private:
  __host__ __device__ unsigned linear_id() const {
#if defined(__CUDA_ARCH__)
    return threadIdx.x;
#else
    return id_;
#endif
  }

  KernelInfo kinfo_{};  // the derived launch values (blockDim), the warp_count/total source
  detail::Decomposition dec_{};
  unsigned id_ = 0u;
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_MAPPING_H_
