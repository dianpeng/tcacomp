// runtime/tests/conformance_new_api.cu — compile-only nvcc conformance for
// the fix4 §4 runtime API (doc/fix/fix4.md §8 T3). Pins the EXACT call
// shapes tile-lower (T4) and codegen/final-lower (T5) emit: explicit-ndim
// map constructors over ABSOLUTE extents, the CTA/WARP round walks plus the
// coupled inside walk, the §4.4-unchanged WarpReducer bracket, and the
// 4-arg ValidateLaunch + launch at the §3.3 worked-example shape. No
// runner — the OBJECT target's build IS the test.
#include "tca/runtime.h"

template <typename T>
__device__ static __forceinline__ T ConformanceCombine0(T p0, T p1) {
  return (p0 + p1);
}

namespace {

__global__ void __launch_bounds__(64) ConformanceNewApi(float* __restrict__ sink) {
  // dynamic SMEM pool (65536 B): one pool, disjoint v0 offsets.
  extern __shared__ __align__(16) char tca_smem[];

  // fix4 §3.2 map binding: one pair per LNT, explicit ndim + ABSOLUTE
  // extents — 2 CTA dims (LoC = 4, 16 -> 64 CTAs) and 2 WARP dims
  // (K = 2, 8 -> 16 warps = 512 threads).
  tca::rt::KernelInfo kinfo;
  tca::rt::GridMapping grid_map0(kinfo, 2u, 4u, 16u);
  tca::rt::ThreadMapping thread_map1(kinfo, 2u, 2u, 8u);

  // Map queries (§4.1), all consumed below.
  float v = 0.0f;
  v += static_cast<float>(grid_map0.index());
  v += static_cast<float>(grid_map0.index(0u));
  v += static_cast<float>(grid_map0.index(1u));
  v += static_cast<float>(grid_map0.dimension(0u));
  v += static_cast<float>(thread_map1.index());
  v += static_cast<float>(thread_map1.index(0u));
  v += static_cast<float>(thread_map1.index(1u));
  v += static_cast<float>(thread_map1.lane_id());
  v += static_cast<float>(thread_map1.warp_index());
  v += thread_map1.is_first_in_warp() ? 1.0f : 0.0f;
  v += thread_map1.is_first_in_cta() ? 1.0f : 0.0f;
  v += static_cast<float>(thread_map1.warp_count());
  v += static_cast<float>(thread_map1.total());

  // §3.1 CTA round walk: loc = the dim's ABSOLUTE LoC; step = LoC x T =
  // 4 x 256 from the caller; bound = the saturating F1 tile_max.
  for (tca::rt::CtaIterator cta_it(grid_map0, 0u, 4u, 0u, 4u * 256u, 4096u);
       cta_it.Has(); cta_it.Next()) {
    v += static_cast<float>(cta_it.Index());
  }

  // §3.1 WARP-dim round walk: loc = K_d (warp units); step = K x T = 2 x 32.
  for (tca::rt::ThrIterator warp_it(thread_map1, 0u, 2u, 0u, 2u * 32u, 1024u);
       warp_it.Has(); warp_it.Next()) {
    v += static_cast<float>(warp_it.Index());
  }

  // §3.1 the coupled inside walk: init = the no-arg flat thread rank,
  // step = blockDim — both caller-supplied; the iterator is walk-agnostic.
  for (tca::rt::ThrIterator inside_it(thread_map1, 0u, 2u, thread_map1.index(), blockDim.x,
                                      600u);
       inside_it.Has(); inside_it.Next()) {
    v += static_cast<float>(inside_it.Index());
  }

  // §4.4 reducer face, UNCHANGED: Init -> Reduce(value) -> Combine bracket
  // over the compiler-owned staging row (one slot per warp of the launch).
  tca::rt::WarpReducer<float> reducer0(0.0f, ConformanceCombine0, thread_map1,
                                       tca::rt::SmemView<float, 1u>(
                                           reinterpret_cast<float*>(tca_smem + 0u), 16u));
  reducer0.Init();
  reducer0.Reduce(v);
  v += reducer0.Combine();

  if (v == -1.0f) sink[0] = v;  // keep every query live; never taken
}

}  // namespace

// The launcher-shape host function T5 emits (fix4 §4.3/§5.1): validate,
// then launch, then check — the §3.3 worked-example shape. gridDim 16
// (M=CTA LoC=16), blockDim 64 (K=2 warps x 32 lanes), 64 KB dynamic SMEM
// (the rhs window at the MaxCache cap), and the compiler-emitted MaxCTA
// policy constant 512 — deliberately NO default argument: the launcher
// must pass it. Compile-only (never called in this TU).
inline void conformance_launch_shape(float* sink) {
  tca::rt::ValidateLaunch(dim3(16u, 1u, 1u), dim3(64u, 1u, 1u), 65536u, 512u);
  ConformanceNewApi<<<dim3(16u, 1u, 1u), dim3(64u, 1u, 1u), 65536u, 0>>>(sink);
  TCA_CUDA_CHECK(cudaGetLastError());
}
