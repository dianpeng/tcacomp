// tca/reducer.h — the reducers (doc/rt.md §4.6).
//
// Scalar-only reduction protocol (R1/R2 ruling 2026-10-23): the compiler
// stamps NO Slots template parameter — Reduce(value) folds one element per
// call and Combine() always returns T. Warp staging lives in the COMPILER'S
// dynamic-SMEM pool, not in runtime-static scratch: the ctor receives the
// reducer's staging row as a SmemView<T, 1u> (extent = the CTA's warp
// count) and the mapping as a const ThreadMapping&. Warp/lane/count
// identity is runtime-internal control flow derived from the map
// (warp_index()/lane_id()/is_first_in_warp()/warp_count()) — generated code
// stays branch-free and never inlines threadIdx arithmetic. The combiner
// parameter is the CONCRETE function-pointer type T (*)(T, T) — the
// goldens pass a function-template NAME (RedwarpCombine0), which only
// binds against a concrete target type. The runtime owns every combiner
// call.
#ifndef TCA_RT_REDUCER_H_
#define TCA_RT_REDUCER_H_

// Host-compilation shim (see mapping.h).
#if !defined(__CUDACC__) && !defined(TCA_HOST_SHIM)
#define TCA_HOST_SHIM 1
#define __device__
#define __host__
#endif

#include <cassert>

#include "mapping.h"
#include "view.h"

namespace tca {
namespace rt {

// SeqReducer — per-thread accumulators, no synchronization; Combine returns
// the local accumulator. Fully __host__ __device__ (CPU-testable).
template <typename T>
class SeqReducer {
 public:
  __host__ __device__ SeqReducer(T init, T (*fn)(T, T)) : init_(init), fn_(fn) {
    assert(fn != nullptr);
    acc_ = init_;
  }

  // Argument-free; re-initializes from the construction-time init.
  __host__ __device__ void Init() { acc_ = init_; }

  // One element per call (R1: the slot argument is gone).
  __host__ __device__ void Reduce(T v) { acc_ = fn_(acc_, v); }

  // Called once per output element; always returns T.
  __host__ __device__ T Combine() { return acc_; }

 private:
  T init_;
  T (*fn_)(T, T);
  T acc_;
};

// WarpReducer — the v0 protocol (R1/R2 ruling 2026-10-23):
//   1. Init/Reduce on the private accumulator as above;
//   2. Combine(): full-warp shuffle down-tree (__shfl_down_sync(0xffffffff,
//      ...) — valid because every CTA thread participates and out-of-group
//      lanes hold the identity init), the warp's first lane stages its
//      partial into the ctor-provided SMEM staging row AT THE WARP'S INDEX,
//      __syncthreads(), then every thread folds ALL staged warp partials in
//      ascending warp order from the identity init (tca.md §5.10's
//      replicated-local-combine broadcast — every participant receives the
//      final value; the element loop's per-element brackets each read it).
// Staging ownership (R1): the compiler owns the scratch — one disjoint
// row of #warps × sizeof(T) bytes in the kernel's dynamic-SMEM pool, handed
// in as a SmemView<T, 1u>. The runtime keeps NO static __shared__.
template <typename T>
class WarpReducer {
 public:
  __host__ __device__ WarpReducer(T init, T (*fn)(T, T), const ThreadMapping& map,
                                  SmemView<T, 1u> staging)
      : init_(init), fn_(fn), map_(map), staging_(staging) {
    assert(fn != nullptr);
    acc_ = init_;
  }

  __host__ __device__ void Init() { acc_ = init_; }

  __host__ __device__ void Reduce(T v) { acc_ = fn_(acc_, v); }

  __host__ __device__ T Combine() {
#if defined(__CUDA_ARCH__)
    // Device passes see only this branch (nvcc eliminates the inactive one).
    T part = acc_;
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
      part = fn_(part, __shfl_down_sync(0xffffffffu, part, off));
    }
    if (map_.is_first_in_warp()) {
      staging_.Set(part, map_.warp_index());
    }
    __syncthreads();  // legal: Combine is uniformly called by every CTA thread
    // Deterministic replicated-local fold over every staged warp (ascending).
    T out = init_;
    for (unsigned w = 0u; w < map_.warp_count(); ++w) {
      out = fn_(out, staging_.Get(w));
    }
    return out;
#else
    // Host (CPU unit tests / nvcc host pass): no warps — the seq algebra.
    return acc_;
#endif
  }

 private:
  T init_;
  T (*fn_)(T, T);
  const ThreadMapping& map_;
  SmemView<T, 1u> staging_;
  T acc_;
};

// GlobalReducer — registered scaffolding only (no v0 preset assigns CTA
// parallelism to a reduced dim). Compiling stub, same face; bodies trap at
// runtime. Enablement needs the multi-CTA scratch/ownership plan (tca.md
// §5.10(3)) — deliberately no static_assert.
template <typename T>
class GlobalReducer {
 public:
  __host__ __device__ GlobalReducer(T init, T (*fn)(T, T), const ThreadMapping& map,
                                    SmemView<T, 1u> staging)
      : init_(init), fn_(fn), map_(map), staging_(staging) {
    assert(fn != nullptr);
    (void)map_;
    (void)staging_;
  }

  __host__ __device__ void Init() { assert(!"GlobalReducer: not enabled in v0"); }
  __host__ __device__ void Reduce(T v) {
    (void)v;
    assert(!"GlobalReducer: not enabled in v0");
  }
  __host__ __device__ T Combine() {
    assert(!"GlobalReducer: not enabled in v0");
    return init_;
  }

 private:
  T init_;
  T (*fn_)(T, T);
  const ThreadMapping& map_;
  SmemView<T, 1u> staging_;
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_REDUCER_H_
