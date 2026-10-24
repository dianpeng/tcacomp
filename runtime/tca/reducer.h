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

// WarpReducer — the 2026-10-24 protocol ruling (supersedes the R1/R2-era
// per-thread-accumulator + Combine-internal shuffle):
//   1. Init() resets ALL SMEM slots single-writer: each warp's first lane
//      rewrites its OWN slot; Init is uniformly called by every CTA thread
//      (the bracket law), so every slot resets, and warps that never
//      Reduce hold the identity init.
//   2. Reduce(v) does NOT accumulate per-thread: per call, an
//      __activemask()-derived down-tree combines fn across the
//      PARTICIPATING lanes INSIDE Reduce; the warp partial lands in the
//      first active lane, which folds it into its warp's SMEM slot
//      (slot[warp] = fn(slot[warp], warpPartial) — single writer per
//      address).
//   3. Combine(): __syncthreads() FIRST (the slot writes must be visible),
//      then EVERY thread folds all warp_count() slots in ascending warp
//      order from the identity init — the replicated-local-combine
//      broadcast (tca.md §5.10). That barrier is the protocol's ONLY
//      synchronization.
// Staging ownership (R1, unchanged): the compiler owns the scratch — one
// disjoint row of #warps × sizeof(T) bytes in the kernel's dynamic-SMEM
// pool, handed in as a SmemView<T, 1u>. The runtime keeps NO static
// __shared__. No per-thread accumulator exists in device code.
template <typename T>
class WarpReducer {
 public:
  __host__ __device__ WarpReducer(T init, T (*fn)(T, T), const ThreadMapping& map,
                                  SmemView<T, 1u> staging)
      : init_(init), fn_(fn), map_(map), staging_(staging) {
    assert(fn != nullptr);
#ifndef __CUDACC__
    host_acc_ = init_;
#endif
  }

  // Single-writer reset of ALL slots (see the class comment): each warp's
  // first lane rewrites its own slot; the uniform bracket call guarantees
  // every warp's first lane is present.
  __host__ __device__ void Init() {
#if defined(__CUDA_ARCH__)
    if (map_.is_first_in_warp()) {
      staging_.Set(init_, map_.warp_index());
    }
#elif !defined(__CUDACC__)
    host_acc_ = init_;
#endif
  }

  // One element per call (R1): the activemask-derived guarded down-tree
  // folds the participating lanes' values into the warp partial, and the
  // first active lane folds it into its warp's slot.
  __host__ __device__ void Reduce(T v) {
#if defined(__CUDA_ARCH__)
    // PREFIX-ALIGNMENT CONTRACT (the correctness argument, per the
    // 2026-10-24 ruling): the compiler's bounds gates produce
    // PREFIX-aligned active sets — the F1 tileMax kills lanes from the top
    // within a warp (a lane's per-thread offset grows with its lane id, so
    // higher lanes exhaust earlier) and the mirror gate kills whole warps
    // at the 256-thread/8-warp alignment — so the first active lane is
    // lane 0 and the guarded tree below folds EXACTLY the active lanes'
    // values into it. Non-prefix masks are out of scope for v0.
    const unsigned mask = __activemask();
    T part = v;
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
      // Every lane in the mask executes the shuffle uniformly (the sync
      // flag names the mask); a lane whose source lane falls outside the
      // active prefix DISCARDS the read.
      const T got = __shfl_down_sync(mask, part, off);
      if (map_.lane_id() + off < __popc(mask)) {
        part = fn_(part, got);
      }
    }
    // The first active lane under prefix alignment is lane 0 (see the
    // contract above); it alone folds the warp partial into its warp's
    // slot — single writer per address.
    if (map_.lane_id() == 0u) {
      staging_.Set(fn_(staging_.Get(map_.warp_index()), part), map_.warp_index());
    }
#elif !defined(__CUDACC__)
    host_acc_ = fn_(host_acc_, v);
#else
    (void)v;  // nvcc host pass: the reducer is device-only
#endif
  }

  // Barrier FIRST, then the all-threads ascending slot fold — the
  // protocol's ONLY synchronization.
  __host__ __device__ T Combine() {
#if defined(__CUDA_ARCH__)
    __syncthreads();  // legal: Combine is uniformly called by every CTA thread
    // Deterministic replicated-local fold over every staged warp (ascending).
    T out = init_;
    for (unsigned w = 0u; w < map_.warp_count(); ++w) {
      out = fn_(out, staging_.Get(w));
    }
    return out;
#elif !defined(__CUDACC__)
    return host_acc_;
#else
    return init_;  // nvcc host pass: never called (the kernel is device-only)
#endif
  }

 private:
  T init_;
  T (*fn_)(T, T);
  const ThreadMapping& map_;
  SmemView<T, 1u> staging_;
#ifndef __CUDACC__
  // Host-only stand-in (plain CPU compilation): no warps exist on the CPU
  // layer, so the host branch runs the sequential algebra. Under nvcc this
  // member exists in NEITHER pass — device code carries no accumulator
  // register.
  T host_acc_;
#endif
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
