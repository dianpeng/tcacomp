// tca/check.h — the host failure surface (doc/rt.md §5).
//
// Everything host-side traps through Throw (std::runtime_error); the macros
// add file:line context. TCA_CUDA_CHECK follows the scalfold CUDA_CHECK shape
// (kernel launches return void — call sites follow each launch with
// TCA_CUDA_CHECK(cudaGetLastError())).
#ifndef TCA_RT_CHECK_H_
#define TCA_RT_CHECK_H_

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tca {
namespace rt {

[[noreturn]] inline void Throw(const char* what) { throw std::runtime_error(what); }

}  // namespace rt
}  // namespace tca

// TCA_REQUIRE(cond, what): throw with a file:line prefix when cond fails.
#define TCA_REQUIRE(cond, what)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ::tca::rt::Throw((std::string("TCA_REQUIRE failed at ") + __FILE__ +     \
                        ":" + std::to_string(__LINE__) + ": " + (what))        \
                           .c_str());                                          \
    }                                                                          \
  } while (0)

// TCA_CUDA_CHECK(expr): evaluate expr exactly once; on any cudaError_t !=
// cudaSuccess throw with __FILE__:__LINE__ and cudaGetErrorString.
#define TCA_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    const cudaError_t tca_cuda_check_err_ = (expr);                            \
    if (tca_cuda_check_err_ != cudaSuccess) {                                  \
      ::tca::rt::Throw((std::string("CUDA error at ") + __FILE__ + ":" +       \
                        std::to_string(__LINE__) + ": " +                      \
                        cudaGetErrorString(tca_cuda_check_err_))               \
                           .c_str());                                          \
    }                                                                          \
  } while (0)

#endif  // TCA_RT_CHECK_H_
