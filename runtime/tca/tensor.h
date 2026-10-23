// tca/tensor.h — the host launcher contract type (doc/rt.md §5).
//
// Tensor<T, Rank> is NON-OWNING: device pointer + shape + layout in one
// object. The generated launcher takes const Tensor& inputs / Tensor& stored
// outputs, derives every dim from the tensor occurrences, and cross-validates
// with named TCA_REQUIREs (the enforced shape contract).
#ifndef TCA_RT_TENSOR_H_
#define TCA_RT_TENSOR_H_

#include "tca/view.h"  // Layout

namespace tca {

// The template parameter is spelled `R` internally: a Rank() method would
// shadow the template parameter name inside the class body.
template <typename T, unsigned R>
struct Tensor {
  static constexpr unsigned kRank = R;

  Tensor() = default;

  template <typename... Dims>
  __host__ static Tensor From(T* data, Dims... dims) {
    static_assert(sizeof...(Dims) == R, "dimension count mismatch");
    Tensor t;
    t.data_ = data;
    const unsigned d[R] = {static_cast<unsigned>(dims)...};
    for (unsigned i = 0u; i < R; ++i) t.dims_[i] = d[i];
    return t;
  }

  __host__ T* Data() const { return data_; }
  __host__ unsigned Dim(unsigned axis) const { return dims_[axis]; }
  __host__ unsigned Rank() const { return kRank; }
  __host__ rt::Layout layout() const { return rt::Layout::RowMajor; }

 private:
  T* data_ = nullptr;
  unsigned dims_[R] = {};
};

}  // namespace tca

#endif  // TCA_RT_TENSOR_H_
