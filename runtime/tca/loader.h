// tca/loader.h — the tile loaders (doc/rt.md §4.5, fix3.md D-m/D-n, R-H/R-J).
// The load primitive is a COORDINATION MAPPING: the caller passes (1) the
// grid/thread MAPPING object that assigns this thread's share of the work,
// (2) the destination view, and (3) the source as a WINDOWED view whose
// offset is implicit (never raw bases). The runtime composes the mapping
// into a valid source->dest copy; the compiler NEVER inlines the loader's
// thread enumeration (R-H: fusion happens here, at the hardware
// instruction, and only at codegen).
//
// The cooperative CTA entry point is nvcc-only (barriers); the per-thread
// entry point and the row-striped copy core (detail::load_tile_rows) are
// dual-compiled and pinned by the CPU unit layer.
//
// The runtime owns the copy, ALL barriers, and tail clamping — the compiler
// emits no synchronization anywhere. Window->axis contract (pinned): the
// destination window covers the source's TRAILING dst.kRank axes; the
// windowed base carries one coordinate per source axis in order (trailing
// axes get their round bases, leading axes their fixed coordinates).
//
// Enumeration (fix3 D-n): threads stripe the leading (rank-1) rows — div/mod
// is paid ONCE PER ROW, never per element; the per-row tail clamp
// umin(dstTrailing, srcDim - base) replaces the per-element bounds
// predicate; each row's trailing axis is ONE FLAT CONTIGUOUS RUN in both
// views (row-major, trailing stride 1 — one LDG+STS per element plus the
// tail comparison). Out-of-range padding is ZERO-FILLED (the retained
// scalar reference path's suppression semantics: a suppressed GetAt yields
// T{} — safe padding, deterministic content).
//
// Fast paths, both FALLING BACK to the scalar run copy when their
// eligibility fails (checked per row at runtime):
//   - a 128-bit vector path (uint4 on device) when the element is 4 bytes,
//     the trailing extent %4 == 0, and both row base pointers are 16B
//     aligned;
//   - a cp.async path (LDGSTS) for SMEM destinations on __CUDA_ARCH__ >=
//     800: 16B per thread per instruction, zero-fill via the src-size/zfill
//     operand, ONE __pipeline_commit() + __pipeline_wait_prior(0) before
//     the trailing barrier. C9 is pinned: exactly one __syncthreads()
//     before and one after the fill; the runtime owns ALL barriers and tail
//     clamping; bounds predication semantics unchanged; cp.async's
//     per-thread wait is NOT a barrier.
#ifndef TCA_RT_LOADER_H_
#define TCA_RT_LOADER_H_

#include <cstdint>

#include "tca/mapping.h"
#include "tca/view.h"

#if defined(__CUDACC__)
#include <cuda_pipeline_primitives.h>  // __pipeline_* (cp.async, guarded below)
#endif

namespace tca {
namespace rt {
namespace detail {

// The 16-byte copy unit of the vector path: uint4 under nvcc (the LDG.128 /
// STS.128 / LDGSTS.128 granularity); a CPU stand-in with identical size and
// alignment for the host-compiled unit layer.
#if defined(__CUDACC__)
using vec16 = uint4;
#else
struct alignas(16) vec16 {
  unsigned x, y, z, w;
};
#endif

// vec16_ok reports 128-bit eligibility for one row copy: both endpoints
// 16-byte aligned (element count, type size, and source strides decide this
// at runtime — a partial tile's round base is not guaranteed aligned).
__host__ __device__ inline bool vec16_ok(const void* d, const void* s) {
  return (reinterpret_cast<uintptr_t>(d) & 15u) == 0u &&
         (reinterpret_cast<uintptr_t>(s) & 15u) == 0u;
}

// copy_row is the retained SCALAR reference path plus its vector
// accelerator: [0, count) copies from the contiguous source run, [count,
// trailing) zero-fills (suppression semantics). The scalar loop is the
// CPU-unit-layer and fallback implementation; the vec16 path activates only
// under its per-row eligibility check.
template <typename E>
__host__ __device__ void copy_row(E* dst, const E* src, unsigned count, unsigned trailing) {
  if constexpr (sizeof(E) == 4u) {
    if (trailing % 4u == 0u && vec16_ok(dst, src)) {
      // 128-bit vector run + scalar tail + scalar zero-fill (the zero-fill
      // tail is a cold ragged-tile path; a misaligned 16B store there would
      // be wrong, and the simple loop keeps the contract obvious).
      const unsigned n4 = count >> 2;
      const vec16* s4 = reinterpret_cast<const vec16*>(src);
      vec16* d4 = reinterpret_cast<vec16*>(dst);
      for (unsigned i = 0u; i < n4; ++i) d4[i] = s4[i];
      for (unsigned k = n4 << 2; k < count; ++k) dst[k] = src[k];
      for (unsigned k = count; k < trailing; ++k) dst[k] = E{};
      return;
    }
  }
  for (unsigned k = 0u; k < count; ++k) dst[k] = src[k];
  for (unsigned k = count; k < trailing; ++k) dst[k] = E{};
}

// cp_async_row issues one row's copy as cp.async 16B chunks (sm_80+,
// SMEM destinations only): every chunk covers 16 destination bytes, the
// zfill operand zero-fills the bytes past the row's valid source bytes —
// tail clamp and padding zero-fill in the instruction itself. The caller
// owns commit/wait placement (one commit, wait_prior(0) before the
// trailing barrier). nvcc-only: the intrinsics do not exist for a plain
// host TU (the CPU unit layer never instantiates this path).
#if defined(__CUDACC__)
template <typename E>
__device__ void cp_async_row(char* dst, const char* src, unsigned count, unsigned trailing) {
  const unsigned total = trailing * sizeof(E);
  const unsigned valid_bytes = count * sizeof(E);
  for (unsigned off = 0u; off < total; off += 16u) {
    int v = static_cast<int>(valid_bytes) - static_cast<int>(off);
    if (v > 16) v = 16;
    if (v < 0) v = 0;
    __pipeline_memcpy_async(dst + off, src + off, 16u, static_cast<size_t>(16u - v));
  }
}
#endif  // __CUDACC__

// load_tile_rows is the row-striped window copy core (dual-compiled: the
// CPU unit layer pins it directly). The source arrives as a WINDOWED view
// (fix3 R-J: the offset is implicit — win.Base(a) supplies the per-axis
// base). kCpAsync selects the SMEM cp.async issue path (device-only; the
// host instantiation of that branch is empty and never called).
template <bool kCpAsync, typename DstT, typename WinT, unsigned DR, unsigned SR>
__host__ __device__ void load_tile_rows(DstT& dst, const WinT& win, unsigned tid, unsigned nthreads) {
  static_assert(DR >= 1u && DR <= SR, "window must cover 1..src.kRank trailing axes");
  static_assert(sizeof(typename DstT::ElementType) == sizeof(typename WinT::ElementType),
                "source and destination element sizes must match");
  using E = typename DstT::ElementType;

  unsigned base[SR];
  for (unsigned a = 0u; a < SR; ++a) base[a] = win.Base(a);

  // Leading (DR-1) dst axes index the rows (DR == 1: one row); the trailing
  // axis is the per-row contiguous run.
  unsigned nrows = 1u;
  for (unsigned d = 0u; d + 1u < DR; ++d) nrows *= dst.Extent(d);
  const unsigned trailing = dst.Extent(DR - 1u);

  // Row-invariant tail clamp (fix3 D-n ii): only leading coordinates vary
  // across rows, so the trailing-axis clamp computes once.
  const unsigned last_dim = win.Dim(SR - 1u);
  const unsigned count =
      base[SR - 1u] < last_dim ? umin(trailing, last_dim - base[SR - 1u]) : 0u;

  E* const dbase = dst.Data();
  const E* const sbase = win.Data();
  for (unsigned row = tid; row < nrows; row += nthreads) {
    // Decompose the row index into leading window coords — once per row.
    unsigned w[DR > 1u ? DR - 1u : 1u];
    unsigned rem = row;
    for (unsigned d = DR - 1u; d-- > 0u;) {
      w[d] = rem % dst.Extent(d);
      rem /= dst.Extent(d);
    }
    // Full source coordinate of the row start; a LEADING coordinate outside
    // the source kills the whole row (its window cells zero-fill).
    unsigned c[SR];
    for (unsigned a = 0u; a < SR; ++a) c[a] = base[a];
    bool alive = true;
    for (unsigned j = 0u; j + 1u < DR; ++j) {
      c[SR - DR + j] += w[j];
      if (c[SR - DR + j] >= win.Dim(SR - DR + j)) alive = false;
    }
    E* const drow = dbase + static_cast<unsigned long long>(row) * trailing;
    if (!alive) {
      for (unsigned k = 0u; k < trailing; ++k) drow[k] = E{};
      continue;
    }
    const E* const srow = sbase + win.FlatIndex(c);
    if constexpr (kCpAsync) {
#if defined(__CUDA_ARCH__)
      if (sizeof(E) == 4u && trailing % 4u == 0u &&
          vec16_ok(drow, reinterpret_cast<const void*>(srow))) {
        cp_async_row<E>(reinterpret_cast<char*>(drow), reinterpret_cast<const char*>(srow),
                        count, trailing);
      } else {
        copy_row(drow, srow, count, trailing);
      }
#endif
    } else {
      copy_row(drow, srow, count, trailing);
    }
  }
}

}  // namespace detail

// Per-thread fill into a Frag: this thread's OWN windowed tile, walked
// row-wise single-threaded, no synchronization (a per-thread fill is not a
// cooperative op — the mapping parameter is the coordination contract per
// R-J; the per-thread enumeration has no cross-thread share to compose).
// Portable (no device intrinsics): the CPU unit layer exercises it directly.
template <typename FragT, typename GT, unsigned SR>
__host__ __device__ void LoadTileThread(const ThreadMapping& mapping, FragT& dst,
                                        const TensorWindow<GT, SR>& src) {
  static_cast<void>(mapping);
  static_assert(FragT::kRank >= 1u && FragT::kRank <= SR,
                "window must cover 1..src.kRank trailing axes");
  constexpr unsigned DR = FragT::kRank;
  detail::load_tile_rows<false, FragT, TensorWindow<GT, SR>, DR, SR>(dst, src, 0u, 1u);
}

#if defined(__CUDACC__)

// Cooperative CTA fill: EVERY thread of the CTA participates — the MAPPING
// supplies the flat thread rank (mapping.index()) and the coordination
// total (mapping.total()) the rows stripe over (R-J: the compiler never
// inlines threadIdx/blockDim arithmetic into generated code). Ordering
// contract (C9, pinned): one __syncthreads() BEFORE the copy (refill
// write-after-read wait) and one AFTER (visibility); the runtime owns ALL
// barriers and tail clamping. The cp.async fast path (__CUDA_ARCH__ >=
// 800) is INTERNAL — one commit + wait_prior(0) before the trailing
// barrier; the per-thread wait is not a barrier.
template <typename ST, unsigned DR, typename GT, unsigned SR>
__device__ void LoadTileCta(const ThreadMapping& mapping, SmemView<ST, DR>& dst,
                            const TensorWindow<GT, SR>& src) {
  __syncthreads();  // refill write-after-read wait (reuse safety)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  detail::load_tile_rows<true, SmemView<ST, DR>, TensorWindow<GT, SR>, DR, SR>(
      dst, src, mapping.index(), mapping.total());
  __pipeline_commit();
  __pipeline_wait_prior(0);
#else
  detail::load_tile_rows<false, SmemView<ST, DR>, TensorWindow<GT, SR>, DR, SR>(
      dst, src, mapping.index(), mapping.total());
#endif
  __syncthreads();  // visibility before any read
}

#endif  // __CUDACC__

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_LOADER_H_
