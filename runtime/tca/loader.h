// tca/loader.h — the tile loaders (doc/rt.md §4.5). nvcc-only (device
// constructs; never included by the CPU unit layer).
//
// The runtime owns the copy, ALL barriers, and tail clamping — the compiler
// emits no synchronization anywhere. Window->axis contract (pinned): the
// destination window covers the source tensor's TRAILING dst.kRank axes;
// baseCoords carries one coordinate per source axis in order (trailing axes
// get their round bases, leading axes their fixed coordinates).
#ifndef TCA_RT_LOADER_H_
#define TCA_RT_LOADER_H_

#include "tca/view.h"

namespace tca {
namespace rt {
namespace detail {

// Copy one window element: decompose linear window index `i` row-major over
// the destination extents, offset each trailing source axis by its window
// coordinate. The source view's bounds suppression IS the tail clamp — a
// partial tile is filled; padding content is unspecified but safe.
template <typename DstT, typename SrcT, unsigned DR, unsigned SR>
__device__ void copy_window_elem(DstT& dst, const SrcT& src, const unsigned (&base)[SR],
                                 unsigned i) {
  static_assert(DR >= 1u && DR <= SR, "window must cover 1..src.kRank trailing axes");
  unsigned w[DR];
  unsigned rem = i;
  for (unsigned d = DR; d-- > 0u;) {
    w[d] = rem % dst.Extent(d);
    rem /= dst.Extent(d);
  }
  unsigned c[SR];
  for (unsigned a = 0u; a < SR; ++a) c[a] = base[a];
  for (unsigned j = 0u; j < DR; ++j) c[SR - DR + j] += w[j];
  dst.SetAt(src.GetAt(c), w);
}

template <typename DstT>
__device__ unsigned window_elems(const DstT& dst, unsigned rank) {
  unsigned n = 1u;
  for (unsigned d = 0u; d < rank; ++d) n *= dst.Extent(d);
  return n;
}

}  // namespace detail

// Cooperative CTA fill: ALL threads of the CTA call it the same number of
// times per round; window elements are striped over the whole CTA
// (for i = threadIdx.x; i < windowElems; i += blockDim.x). Ordering contract:
// one __syncthreads() BEFORE the copy loop (refill write-after-read wait) and
// one AFTER (visibility). v0 is a plain copy; the cp.async fast path
// (__CUDA_ARCH__ >= 800) is a future INTERNAL change, invisible to callers.
template <typename ST, unsigned DR, typename GT, unsigned SR, typename... Bases>
__device__ void LoadTileCta(SmemView<ST, DR>& dst, const TensorView<GT, SR>& src,
                            Bases... bases) {
  static_assert(sizeof...(Bases) == SR, "one base coordinate per source axis");
  const unsigned base[SR] = {static_cast<unsigned>(bases)...};
  const unsigned elems = detail::window_elems(dst, DR);

  __syncthreads();  // refill write-after-read wait (reuse safety)
  for (unsigned i = threadIdx.x; i < elems; i += blockDim.x) {
    detail::copy_window_elem<SmemView<ST, DR>, TensorView<GT, SR>, DR, SR>(dst, src, base, i);
  }
  __syncthreads();  // visibility before any read
}

// Per-thread fill into a Frag: same window math, single-thread copy loop, no
// synchronization (a per-thread fill is not a cooperative op).
template <typename FragT, typename ViewT, typename... Bases>
__device__ void LoadTileThread(FragT& dst, const ViewT& src, Bases... bases) {
  static_assert(FragT::kRank >= 1u && FragT::kRank <= ViewT::kRank,
                "window must cover 1..src.kRank trailing axes");
  static_assert(sizeof...(Bases) == ViewT::kRank, "one base coordinate per source axis");
  constexpr unsigned DR = FragT::kRank;
  constexpr unsigned SR = ViewT::kRank;
  const unsigned base[SR] = {static_cast<unsigned>(bases)...};

  const unsigned elems = detail::window_elems(dst, DR);
  for (unsigned i = 0u; i < elems; ++i) {
    detail::copy_window_elem<FragT, ViewT, DR, SR>(dst, src, base, i);
  }
}

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_LOADER_H_
