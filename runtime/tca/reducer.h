// tca/reducer.h — the reducers, <T, Slots> (doc/rt.md §4.6).
//
// Slots = the flat accumulator-tile extent the compiler stamps as the
// template argument (0 = scalar reduction); the runtime sizes all state from
// it. v0 slot addressing is FLAT: Reduce(slot, v) takes exactly one index.
// Construction carries ALL state (init, combiner); the combiner parameter is
// the CONCRETE function-pointer type T (*)(T, T) — the goldens pass a
// function-template NAME (RedwarpCombine0), which only binds against a
// concrete target type. The runtime owns every combiner call.
#ifndef TCA_RT_REDUCER_H_
#define TCA_RT_REDUCER_H_

// Host-compilation shim (see mapping.h).
#if !defined(__CUDACC__) && !defined(TCA_HOST_SHIM)
#define TCA_HOST_SHIM 1
#define __device__
#define __host__
#endif

#include <cassert>

namespace tca {
namespace rt {

// SlotValue — the Slots > 0 Combine() return: a thin array wrapper the
// emitted epilogue reads via .Get(slot_iv).
template <typename T, unsigned Slots>
class SlotValue {
 public:
  static constexpr unsigned kSlots = Slots > 0u ? Slots : 1u;

  __host__ __device__ SlotValue() {
    for (unsigned s = 0u; s < kSlots; ++s) vals_[s] = T{};
  }

  __host__ __device__ T Get(unsigned i) const { return vals_[i]; }
  __host__ __device__ void Set(unsigned i, T v) { vals_[i] = v; }

 private:
  T vals_[kSlots];
};

// SeqReducer — per-thread accumulators, no synchronization; Combine returns
// the local accumulators (copy). Fully __host__ __device__ (CPU-testable).
template <typename T, unsigned Slots>
class SeqReducer {
 public:
  static constexpr unsigned kSlots = Slots > 0u ? Slots : 1u;

  __host__ __device__ SeqReducer(T init, T (*fn)(T, T)) : init_(init), fn_(fn) {
    assert(fn != nullptr);
    for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = init_;
  }

  // Argument-free; re-initializes from the construction-time init.
  __host__ __device__ void Init() {
    for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = init_;
  }

  // Scalar form (Slots == 0).
  __host__ __device__ void Reduce(T v) { acc_[0u] = fn_(acc_[0u], v); }

  // Flat slot form (Slots > 0).
  __host__ __device__ void Reduce(unsigned slot, T v) { acc_[slot] = fn_(acc_[slot], v); }

  // Called once per bracket. T when Slots == 0; SlotValue<T, Slots> otherwise.
  __host__ __device__ auto Combine() {
    if constexpr (Slots == 0u) {
      return acc_[0u];
    } else {
      SlotValue<T, Slots> out;
      for (unsigned s = 0u; s < Slots; ++s) out.Set(s, acc_[s]);
      return out;
    }
  }

 private:
  T init_;
  T (*fn_)(T, T);
  T acc_[kSlots];
};

// WarpReducer — the v0 protocol (identity-init simplification):
//   1. Init/Reduce on the private tile as above;
//   2. Combine(): per-slot full-warp shuffle down-tree
//      (__shfl_down_sync(0xffffffff, ...) — valid because inactive lanes hold
//      the identity init), lane 0 of each warp stages its per-slot partials
//      into the static scratch; __syncthreads(); every thread then folds ALL
//      staged partials per slot locally in ascending warp order (tca.md
//      §5.10's replicated-local-combine broadcast — every participant
//      receives the final value; the re-iteration reads any slot).
// Scratch ruling: function-scope static __shared__ inside the template — one
// allocation per <T, Slots> instantiation, capacity kMaxWarpsPerCTA = 16
// (512/32 default config). Over-provisioned vs the reduction group by
// design: warps outside the group hold the identity init and stage it too.
template <typename T, unsigned Slots>
class WarpReducer {
 public:
  static constexpr unsigned kSlots = Slots > 0u ? Slots : 1u;
  static constexpr unsigned kMaxWarpsPerCTA = 16u;

  __host__ __device__ WarpReducer(T init, T (*fn)(T, T)) : init_(init), fn_(fn) {
    assert(fn != nullptr);
    for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = init_;
  }

  __host__ __device__ void Init() {
    for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = init_;
  }

  __host__ __device__ void Reduce(T v) { acc_[0u] = fn_(acc_[0u], v); }
  __host__ __device__ void Reduce(unsigned slot, T v) { acc_[slot] = fn_(acc_[slot], v); }

  __host__ __device__ auto Combine() {
#if defined(__CUDA_ARCH__)
    // Device passes see only this branch (nvcc eliminates the inactive one).
    static __shared__ T stage[kSlots][kMaxWarpsPerCTA];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned cta_warps = blockDim.x >> 5;
    assert(warp < kMaxWarpsPerCTA);

    T part[kSlots];
    for (unsigned s = 0u; s < kSlots; ++s) part[s] = acc_[s];
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
      for (unsigned s = 0u; s < kSlots; ++s) {
        part[s] = fn_(part[s], __shfl_down_sync(0xffffffffu, part[s], off));
      }
    }
    if (lane == 0u) {
      for (unsigned s = 0u; s < kSlots; ++s) stage[s][warp] = part[s];
    }
    __syncthreads();  // legal: the bracket is uniformly called by every CTA thread
    // Deterministic replicated-local fold over every staged warp (ascending).
    for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = init_;
    for (unsigned w = 0u; w < cta_warps; ++w) {
      for (unsigned s = 0u; s < kSlots; ++s) acc_[s] = fn_(acc_[s], stage[s][w]);
    }
    if constexpr (Slots == 0u) {
      return acc_[0u];
    } else {
      SlotValue<T, Slots> out;
      for (unsigned s = 0u; s < Slots; ++s) out.Set(s, acc_[s]);
      return out;
    }
#else
    // Host (CPU unit tests / nvcc host pass): no warps — the seq algebra.
    if constexpr (Slots == 0u) {
      return acc_[0u];
    } else {
      SlotValue<T, Slots> out;
      for (unsigned s = 0u; s < Slots; ++s) out.Set(s, acc_[s]);
      return out;
    }
#endif
  }

 private:
  T init_;
  T (*fn_)(T, T);
  T acc_[kSlots];
};

// GlobalReducer — registered scaffolding only (no v0 preset assigns CTA
// parallelism to a reduced dim). Compiling stub, same face; bodies trap at
// runtime. Enablement needs the multi-CTA scratch/ownership plan (tca.md
// §5.10(3)) — deliberately no static_assert.
template <typename T, unsigned Slots>
class GlobalReducer {
 public:
  static constexpr unsigned kSlots = Slots > 0u ? Slots : 1u;

  __host__ __device__ GlobalReducer(T init, T (*fn)(T, T)) : init_(init), fn_(fn) {
    assert(fn != nullptr);
  }

  __host__ __device__ void Init() { assert(!"GlobalReducer: not enabled in v0"); }
  __host__ __device__ void Reduce(T v) {
    (void)v;
    assert(!"GlobalReducer: not enabled in v0");
  }
  __host__ __device__ void Reduce(unsigned slot, T v) {
    (void)slot;
    (void)v;
    assert(!"GlobalReducer: not enabled in v0");
  }
  __host__ __device__ auto Combine() {
    assert(!"GlobalReducer: not enabled in v0");
    if constexpr (Slots == 0u) {
      return init_;
    } else {
      return SlotValue<T, Slots>();
    }
  }

 private:
  T init_;
  T (*fn_)(T, T);
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_REDUCER_H_
