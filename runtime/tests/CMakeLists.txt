# Runtime test targets (TCACOMP_RUNTIME_TESTS=ON, doc/rt.md §8.1-8.2):
# the plain-g++ CPU unit layer + compile-conformance objects.

# CPU unit layer: plain C++20, no GPU, no gtest, no linked libraries (the
# INTERFACE link only carries the include directory).
add_executable(tcacomp_cpu_tests cpu_tests.cpp)
target_link_libraries(tcacomp_cpu_tests PRIVATE tcacomp_runtime)
target_compile_options(tcacomp_cpu_tests PRIVATE -Wall -Wextra)

# NEW-API compile conformance (fix4 §8 T3): pins the exact factorization
# call shapes T4/T5 will emit — explicit-ndim map ctors over ABSOLUTE
# extents, the CTA/WARP round walks + the coupled inside walk, the
# §4.4-unchanged WarpReducer bracket, and the 4-arg ValidateLaunch — at
# sm_86 (compile-only OBJECT lib). Mirrors the generated
# matmul_store_cmake.txt — plus the CUDAToolkit lookup that provides the
# CUDA::cudart imported target (`LANGUAGES CUDA` alone does not define it
# in CMake 4.x).
#
# STOOD DOWN in this commit (fix4 §8 T3): the two golden conformance
# targets (tca_golden_matmul / tca_golden_reduce) — the committed goldens
# still speak the pre-fix4 transitional map ctor and stop compiling the
# moment the ctor flips. T4/T5 regenerate the goldens and RESTORE the
# golden targets; until then this new-API object is the nvcc conformance
# gate.
find_package(CUDAToolkit REQUIRED)

add_library(tca_conformance_new OBJECT conformance_new_api.cu)
set_target_properties(tca_conformance_new PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_conformance_new PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_conformance_new PRIVATE tcacomp_runtime CUDA::cudart)
