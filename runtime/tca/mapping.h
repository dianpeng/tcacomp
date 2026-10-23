// tca/mapping.h — KernelInfo + the portion-decomposition maps (doc/rt.md §4.1–4.2).
//
// The maps are OPAQUE: they own the portion grammar (PortionCTA=10 over the
// CTA space, PortionWarp=8 over the per-CTA warp space); the compiler never
// inlines linearization arithmetic. Pure arithmetic — no CUDA intrinsics — so
// the whole header is CPU-testable.
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

// Portion grammar constants (runtime_objects.md §1.2): CTA dim portions sum
// to 10; warp dim portions sum to 8. The launch assert (launch.h) and both
// map decompositions consume these.
inline constexpr unsigned kPortionCTA = 10u;
inline constexpr unsigned kPortionWarp = 8u;
inline constexpr unsigned kWarpsPerWarpPortionMin = 8u;  // minimum config = 8 warps (256 threads)

// u32 pair (doc/rt.md §4.7) — index-arithmetic family lives with the maps.
__host__ __device__ inline unsigned umin(unsigned a, unsigned b) { return a < b ? a : b; }
__host__ __device__ inline unsigned umax(unsigned a, unsigned b) { return a > b ? a : b; }

// KernelInfo — dual-compile pattern so the mapping math is CPU-testable.
// Device: stateless, reads gridDim.x / blockDim.x. Host (plain g++ CPU tests
// or the nvcc host pass): members ctas_/threads_, set by the host-only
// explicit ctor; the host default ctor zeroes them.
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
// The `-1` remainder portion, as it appears after the unsigned cast.
inline constexpr unsigned kPortionRemainder = 0xFFFFFFFFu;

// Portion decomposition shared by both maps (doc/rt.md §4.2). `units` are
// CTAs (GridMapping) or warps (ThreadMapping); the exposed extents carry a
// unit multiplier (CTA: 1; warp: 32 threads).
struct Decomposition {
  unsigned ndim = 0u;
  unsigned unit_extents[kMaxMapDims] = {};   // extents in units (CTAs / warps)
  unsigned exposed[kMaxMapDims] = {};        // what dimension() reports
  unsigned strides[kMaxMapDims] = {};        // row-major strides over exposed

  __host__ __device__ void build(unsigned total_units, unsigned portion_const,
                                 unsigned unit_multiplier, const unsigned* portions,
                                 unsigned n) {
    // Grammar: each portion >= 1 or exactly -1; -1 only last; at least one dim.
    assert(n >= 1u && n <= kMaxMapDims);
    ndim = n;
    const unsigned quantum = total_units / portion_const;
    unsigned concrete_sum = 0u;
    for (unsigned d = 0u; d < n; ++d) {
      const unsigned p = portions[d];
      assert(p >= 1u || p == kPortionRemainder);
      assert(p != kPortionRemainder || d + 1u == n);
      if (p == kPortionRemainder) {
        unit_extents[d] = total_units - concrete_sum;  // the remainder tail
      } else {
        unit_extents[d] = p * quantum;
        concrete_sum += unit_extents[d];
      }
      exposed[d] = unit_extents[d] * unit_multiplier;
    }
    // Row-major strides over the exposed extents (last dim stride 1).
    strides[n - 1u] = 1u;
    for (unsigned d = n - 1u; d-- > 0u;) {
      strides[d] = strides[d + 1u] * exposed[d + 1u];
    }
  }

  // Row-major coordinate of `id` over the exposed extents.
  __host__ __device__ unsigned coord(unsigned d, unsigned id) const {
    return (id / strides[d]) % exposed[d];
  }
};

}  // namespace detail

// GridMapping — the CTA-space map. total = kinfo.total_cta(); extents in CTAs
// are portion_d * (total / kPortionCTA), the -1 tail taking the remainder.
// index(d) = this CTA's rank in dim d (CTA-uniform); dimension(d) = dim d's
// CTA LoC.
class GridMapping {
 public:
  template <typename... P>
  __host__ __device__ GridMapping(const KernelInfo& kinfo, P... portions) {
    static_assert(sizeof...(P) >= 1, "GridMapping needs at least one portion");
    const unsigned prts[] = {static_cast<unsigned>(portions)...};
    dec_.build(kinfo.total_cta(), kPortionCTA, 1u, prts, static_cast<unsigned>(sizeof...(P)));
  }

  __host__ __device__ unsigned dimension(unsigned d) const { return dec_.exposed[d]; }
  __host__ __device__ unsigned index(unsigned d) const { return dec_.coord(d, linear_id()); }

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

  detail::Decomposition dec_{};
  unsigned id_ = 0u;
};

// ThreadMapping — the per-CTA thread-space map (warp allocations never span
// CTAs; ranks are CTA-relative). Warp extents are portion_d * (warps /
// kPortionWarp); exposed in threads (x32). index(d) decomposes threadIdx.x
// row-major over the thread-granular extents.
class ThreadMapping {
 public:
  template <typename... P>
  __host__ __device__ ThreadMapping(const KernelInfo& kinfo, P... portions) {
    static_assert(sizeof...(P) >= 1, "ThreadMapping needs at least one portion");
    const unsigned prts[] = {static_cast<unsigned>(portions)...};
    dec_.build(kinfo.total_thread_per_cta() / 32u, kPortionWarp, 32u, prts,
               static_cast<unsigned>(sizeof...(P)));
  }

  __host__ __device__ unsigned dimension(unsigned d) const { return dec_.exposed[d]; }
  __host__ __device__ unsigned index(unsigned d) const { return dec_.coord(d, linear_id()); }

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

  detail::Decomposition dec_{};
  unsigned id_ = 0u;
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_MAPPING_H_
