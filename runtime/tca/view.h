// tca/view.h — Layout + the memory shim (doc/rt.md §4.4).
//
// Every memory access goes through a view carrying a layout descriptor (the
// cg.md D6 seam): v0 ships Layout::RowMajor only — the future coalescing /
// swizzle passes change descriptors, never call sites. GMEM views are
// bounds-PREDICATED (an out-of-range point is suppressed, never a wild
// access); SMEM/FRAG views are in-window by construction.
#ifndef TCA_RT_VIEW_H_
#define TCA_RT_VIEW_H_

// Host-compilation shim (see mapping.h).
#if !defined(__CUDACC__) && !defined(TCA_HOST_SHIM)
#define TCA_HOST_SHIM 1
#define __device__
#define __host__
#endif

namespace tca {
namespace rt {

enum class Layout : unsigned { RowMajor = 0 /* future: swizzle/coalescing */ };

namespace detail {

// Row-major shape shared by all three view kinds: extents + precomputed
// strides. Get/Set map logical coordinates through the layout descriptor
// (v0: identity + stride math).
template <unsigned Rank>
struct RowMajorShape {
  unsigned dims[Rank] = {};
  unsigned strides[Rank] = {};

  __host__ __device__ void set_dims(const unsigned (&d)[Rank]) {
    for (unsigned i = 0u; i < Rank; ++i) dims[i] = d[i];
    strides[Rank - 1u] = 1u;
    for (unsigned i = Rank - 1u; i-- > 0u;) strides[i] = strides[i + 1u] * dims[i + 1u];
  }

  template <typename... Coords>
  __host__ __device__ unsigned offset(Coords... coords) const {
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    unsigned off = 0u;
    for (unsigned i = 0u; i < Rank; ++i) off += c[i] * strides[i];
    return off;
  }

  template <typename... Coords>
  __host__ __device__ bool in_range(Coords... coords) const {
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    for (unsigned i = 0u; i < Rank; ++i) {
      if (c[i] >= dims[i]) return false;
    }
    return true;
  }

  __host__ __device__ unsigned offset_arr(const unsigned (&c)[Rank]) const {
    unsigned off = 0u;
    for (unsigned i = 0u; i < Rank; ++i) off += c[i] * strides[i];
    return off;
  }

  __host__ __device__ bool in_range_arr(const unsigned (&c)[Rank]) const {
    for (unsigned i = 0u; i < Rank; ++i) {
      if (c[i] >= dims[i]) return false;
    }
    return true;
  }
};

}  // namespace detail

// TensorView — a GMEM window. Get is bounds-PREDICATED (tca.md §5.11): any
// coordinate >= its dimension suppresses the access and yields T{} — the
// whole-body-no-op mechanism for guarded iterations. Set silently drops
// out-of-range writes (one structural owner per address). T may be
// const-qualified (Set is then simply never instantiated).
template <typename T, unsigned Rank>
class TensorView {
 public:
  static constexpr unsigned kRank = Rank;
  using ElementType = T;

  template <typename... Dims>
  __host__ __device__ TensorView(T* data, Dims... dims) : data_(data) {
    static_assert(sizeof...(Dims) == Rank, "dimension count mismatch");
    const unsigned d[Rank] = {static_cast<unsigned>(dims)...};
    shape_.set_dims(d);
  }

  template <typename... Coords>
  __host__ __device__ T Get(Coords... coords) const {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    return GetAt(c);
  }

  template <typename... Coords>
  __host__ __device__ void Set(T v, Coords... coords) {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    SetAt(v, c);
  }

  __host__ __device__ T GetAt(const unsigned (&coords)[Rank]) const {
    if (!shape_.in_range_arr(coords)) return T{};  // suppression — safe padding
    return data_[shape_.offset_arr(coords)];
  }

  __host__ __device__ void SetAt(T v, const unsigned (&coords)[Rank]) {
    if (!shape_.in_range_arr(coords)) return;  // out-of-range: silently dropped
    data_[shape_.offset_arr(coords)] = v;
  }

  __host__ __device__ unsigned Dim(unsigned axis) const { return shape_.dims[axis]; }

  // Loader-facing accessors (fix3 D-n): the row-striped copy core walks
  // contiguous runs, so it needs the raw base and the row-major flat index
  // beside the checked Get/Set face. T may be const-qualified.
  __host__ __device__ T* Data() const { return data_; }
  __host__ __device__ unsigned FlatIndex(const unsigned (&c)[Rank]) const {
    return shape_.offset_arr(c);
  }

 private:
  T* data_;
  detail::RowMajorShape<Rank> shape_{};
};

// SmemView — a dynamic-pool slice. Indexed in-window by construction: Get/Set
// are unchecked (the loaders own tail clamping).
template <typename T, unsigned Rank>
class SmemView {
 public:
  static constexpr unsigned kRank = Rank;
  using ElementType = T;

  template <typename... Extents>
  __host__ __device__ SmemView(T* base, Extents... extents) : base_(base) {
    static_assert(sizeof...(Extents) == Rank, "extent count mismatch");
    const unsigned e[Rank] = {static_cast<unsigned>(extents)...};
    shape_.set_dims(e);
  }

  template <typename... Coords>
  __host__ __device__ T Get(Coords... coords) const {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    return GetAt(c);
  }

  template <typename... Coords>
  __host__ __device__ void Set(T v, Coords... coords) {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    SetAt(v, c);
  }

  __host__ __device__ T GetAt(const unsigned (&coords)[Rank]) const {
    return base_[shape_.offset_arr(coords)];
  }

  __host__ __device__ void SetAt(T v, const unsigned (&coords)[Rank]) {
    base_[shape_.offset_arr(coords)] = v;
  }

  __host__ __device__ unsigned Extent(unsigned axis) const { return shape_.dims[axis]; }

  // Loader-facing accessors (fix3 D-n): see TensorView.
  __host__ __device__ T* Data() const { return base_; }
  __host__ __device__ unsigned FlatIndex(const unsigned (&c)[Rank]) const {
    return shape_.offset_arr(c);
  }

 private:
  T* base_;
  detail::RowMajorShape<Rank> shape_{};
};

// Frag — a per-thread register tile. Same Get/Set face, extents from Dims
// (compile-time). Not exercised by the v0 goldens but part of the registered
// surface.
template <typename T, unsigned Rank, unsigned... Dims>
class Frag {
 public:
  static constexpr unsigned kRank = Rank;
  static constexpr unsigned kElems = (Dims * ... * 1u);
  using ElementType = T;

  __host__ __device__ Frag() {
    static_assert(Rank >= 1u, "Frag rank must be >= 1");
    static_assert(sizeof...(Dims) == Rank, "extent count mismatch");
    const unsigned e[Rank] = {Dims...};
    shape_.set_dims(e);
  }

  template <typename... Coords>
  __host__ __device__ T Get(Coords... coords) const {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    return GetAt(c);
  }

  template <typename... Coords>
  __host__ __device__ void Set(T v, Coords... coords) {
    static_assert(sizeof...(Coords) == Rank, "coordinate count mismatch");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    SetAt(v, c);
  }

  __host__ __device__ T GetAt(const unsigned (&coords)[Rank]) const {
    return data_[shape_.offset_arr(coords)];
  }

  __host__ __device__ void SetAt(T v, const unsigned (&coords)[Rank]) {
    data_[shape_.offset_arr(coords)] = v;
  }

  __host__ __device__ unsigned Extent(unsigned axis) const { return shape_.dims[axis]; }

  // Loader-facing accessors (fix3 D-n): see TensorView.
  __host__ __device__ T* Data() { return data_; }
  __host__ __device__ unsigned FlatIndex(const unsigned (&c)[Rank]) const {
    return shape_.offset_arr(c);
  }

 private:
  detail::RowMajorShape<Rank> shape_{};
  T data_[kElems > 0u ? kElems : 1u] = {};
};

// TensorWindow — a GMEM view windowed at a base coordinate (fix3.md R-J /
// D-m/D-n: the loader's source face). The offset is IMPLICIT in the window:
// the cooperative loaders receive (mapping, dstView, srcWindow) and compose
// the mapping into a valid source->dest copy — the caller never passes raw
// bases and the compiler never inlines the loader's thread enumeration.
// The window keeps the underlying view's bounds-PREDICATED semantics: the
// row-striped core clamps per row against the SOURCE dims at the windowed
// coordinates (out-of-range padding zero-fills).
template <typename T, unsigned Rank>
class TensorWindow {
 public:
  using ElementType = T;
  static constexpr unsigned kRank = Rank;

  template <typename... Coords>
  __host__ __device__ TensorWindow(const TensorView<T, Rank>& v, Coords... coords) : v_(v) {
    static_assert(sizeof...(Coords) == Rank, "one base coordinate per source axis");
    const unsigned c[Rank] = {static_cast<unsigned>(coords)...};
    for (unsigned a = 0u; a < Rank; ++a) base_[a] = c[a];
  }

  // The windowed base coordinate of source axis a (the implicit offset).
  __host__ __device__ unsigned Base(unsigned axis) const { return base_[axis]; }
  // The SOURCE extent of axis a (the clamp domain — not the tile extent).
  __host__ __device__ unsigned Dim(unsigned axis) const { return v_.Dim(axis); }
  // Loader-facing accessors (fix3 D-n): the underlying view's raw face.
  __host__ __device__ T* Data() const { return v_.Data(); }
  __host__ __device__ unsigned FlatIndex(const unsigned (&c)[Rank]) const {
    return v_.FlatIndex(c);
  }

 private:
  TensorView<T, Rank> v_;
  unsigned base_[Rank] = {};
};

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_VIEW_H_
