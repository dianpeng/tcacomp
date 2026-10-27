// tca/iterator.h — the uniform round walk (doc/rt.md §4.3; loc semantics per
// doc/fix/fix4.md §4.2).
//
// ALL THREE iterator types walk identically: base = init; Has() is
// `base < bound`; Next() is `base += step`; Index() returns base. Has() is
// CTA-uniform by construction (init/step/bound are uniform quantities) — this
// is what makes every collective call uniformly reached. The bound is the
// compiler's saturating F1 chain, already evaluated per iteration by the
// emitted call. Cta/Thr carry the mapping/dosoc/loc for ownership-guard
// queries; the v0 walk ignores them.
//
// loc semantics (fix4 §4.2) — the iterator stays WALK-AGNOSTIC, it never
// decomposes loc; only the arguments' meaning is pinned:
//   - CtaIterator (a CTA-assigned dim): loc = the dim's ABSOLUTE LoC (the
//     CTA count owning it); the caller supplies init/step/bound with step =
//     LoC × T_d (fix4 §3.1 CTA row); the rank is the map's tuple coordinate
//     grid_map.index(d).
//   - ThrIterator over a WARP dim: loc = that dim's K_d (WARP units); the
//     caller supplies step = K_d × T_d; the rank is thread_map.index(d), in
//     warp units.
//   - ThrIterator as the CTA dim's COUPLED inside walk (fix4 §3.1): init =
//     thread_map.index() (the no-arg flat rank), step = blockDim — both
//     arrive from the caller; loc carries the owning dim's extent.
#ifndef TCA_RT_ITERATOR_H_
#define TCA_RT_ITERATOR_H_

// Host-compilation shim (see mapping.h).
#if !defined(__CUDACC__) && !defined(TCA_HOST_SHIM)
#define TCA_HOST_SHIM 1
#define __device__
#define __host__
#endif

#include "tca/mapping.h"

namespace tca {
namespace rt {

struct SeqIterator {
  __host__ __device__ SeqIterator(unsigned init, unsigned step, unsigned bound)
      : base_(init), step_(step), bound_(bound) {}

  __host__ __device__ bool Has() const { return base_ < bound_; }
  __host__ __device__ void Next() { base_ += step_; }
  __host__ __device__ unsigned Index() const { return base_; }

 private:
  unsigned base_;
  unsigned step_;
  unsigned bound_;
};

// A CTA-assigned dim's round walk (fix4 §3.1 CTA row): loc = the dim's
// ABSOLUTE LoC, step = LoC × T_d from the caller, dosoc = the dim's GridMap
// slot.
struct CtaIterator {
  __host__ __device__ CtaIterator(const GridMapping& map, unsigned dosoc, unsigned loc,
                                  unsigned init, unsigned step, unsigned bound)
      : map_(map), dosoc_(dosoc), loc_(loc), base_(init), step_(step), bound_(bound) {}

  __host__ __device__ bool Has() const { return base_ < bound_; }
  __host__ __device__ void Next() { base_ += step_; }
  __host__ __device__ unsigned Index() const { return base_; }

 private:
  GridMapping map_;  // stored for ownership-guard queries (unused in v0)
  unsigned dosoc_;
  unsigned loc_;  // the dim's ABSOLUTE LoC (CTA count)
  unsigned base_;
  unsigned step_;
  unsigned bound_;
};

// A thread-assigned walk (fix4 §3.1): over a WARP dim, loc = K_d (WARP
// units), step = K_d × T_d; as the CTA dim's coupled inside walk, init = the
// no-arg flat thread rank and step = blockDim, both caller-supplied.
struct ThrIterator {
  __host__ __device__ ThrIterator(const ThreadMapping& map, unsigned dosoc, unsigned loc,
                                  unsigned init, unsigned step, unsigned bound)
      : map_(map), dosoc_(dosoc), loc_(loc), base_(init), step_(step), bound_(bound) {}

  __host__ __device__ bool Has() const { return base_ < bound_; }
  __host__ __device__ void Next() { base_ += step_; }
  __host__ __device__ unsigned Index() const { return base_; }

 private:
  ThreadMapping map_;  // stored for ownership-guard queries (unused in v0)
  unsigned dosoc_;
  unsigned loc_;  // WARP dims: K_d (warp units); coupled walk: the owning dim's extent
  unsigned base_;
  unsigned step_;
  unsigned bound_;
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_ITERATOR_H_
