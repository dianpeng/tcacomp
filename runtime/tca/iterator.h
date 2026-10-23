// tca/iterator.h — the uniform round walk (doc/rt.md §4.3).
//
// ALL THREE iterator types walk identically: base = init; Has() is
// `base < bound`; Next() is `base += step`; Index() returns base. Has() is
// CTA-uniform by construction (init/step/bound are uniform quantities) — this
// is what makes every collective call uniformly reached. The bound is the
// compiler's saturating F1 chain, already evaluated per iteration by the
// emitted call. Cta/Thr carry the mapping/dosoc/loc for future
// ownership-guard queries; the v0 walk ignores them.
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

struct CtaIterator {
  __host__ __device__ CtaIterator(const GridMapping& map, unsigned dosoc, unsigned loc,
                                  unsigned init, unsigned step, unsigned bound)
      : map_(map), dosoc_(dosoc), loc_(loc), base_(init), step_(step), bound_(bound) {}

  __host__ __device__ bool Has() const { return base_ < bound_; }
  __host__ __device__ void Next() { base_ += step_; }
  __host__ __device__ unsigned Index() const { return base_; }

 private:
  GridMapping map_;  // stored for future ownership-guard queries (unused in v0)
  unsigned dosoc_;
  unsigned loc_;
  unsigned base_;
  unsigned step_;
  unsigned bound_;
};

struct ThrIterator {
  __host__ __device__ ThrIterator(const ThreadMapping& map, unsigned dosoc, unsigned loc,
                                  unsigned init, unsigned step, unsigned bound)
      : map_(map), dosoc_(dosoc), loc_(loc), base_(init), step_(step), bound_(bound) {}

  __host__ __device__ bool Has() const { return base_ < bound_; }
  __host__ __device__ void Next() { base_ += step_; }
  __host__ __device__ unsigned Index() const { return base_; }

 private:
  ThreadMapping map_;  // stored for future ownership-guard queries (unused in v0)
  unsigned dosoc_;
  unsigned loc_;
  unsigned base_;
  unsigned step_;
  unsigned bound_;
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_ITERATOR_H_
