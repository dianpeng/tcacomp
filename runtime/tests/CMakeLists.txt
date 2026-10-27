# Runtime test targets (TCACOMP_RUNTIME_TESTS=ON, doc/rt.md §8.1-8.2):
# the plain-g++ CPU unit layer + compile-conformance objects.

# CPU unit layer: plain C++20, no GPU, no gtest, no linked libraries (the
# INTERFACE link only carries the include directory).
add_executable(tcacomp_cpu_tests cpu_tests.cpp)
target_link_libraries(tcacomp_cpu_tests PRIVATE tcacomp_runtime)
target_compile_options(tcacomp_cpu_tests PRIVATE -Wall -Wextra)

# NEW-API compile conformance (fix4 §8 T3): pins the exact factorization
# call shapes T4/T5 emit — explicit-ndim map ctors over ABSOLUTE extents,
# the CTA/WARP round walks + the coupled inside walk, the §4.4-unchanged
# WarpReducer bracket, and the 4-arg ValidateLaunch — at sm_86
# (compile-only OBJECT lib). Mirrors the generated matmul_store_cmake.txt
# — plus the CUDAToolkit lookup that provides the CUDA::cudart imported
# target (`LANGUAGES CUDA` alone does not define it in CMake 4.x).
find_package(CUDAToolkit REQUIRED)

add_library(tca_conformance_new OBJECT conformance_new_api.cu)
set_target_properties(tca_conformance_new PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_conformance_new PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_conformance_new PRIVATE tcacomp_runtime CUDA::cudart)

# RESTORED golden compile conformance (fix4 §8 T4/T5 — the T3 stand-down
# debt is paid): the regenerated goldens speak the factorization grammar
# (explicit-ndim ctors over absolute extents, annotation-derived literal
# dim3 launches, the 4-arg ValidateLaunch) and compile at sm_86 against
# the T3 runtime. tca_golden_reduce additionally pins the ZERO-EXTENT
# ThreadMapping (kinfo, 0u) — the Wave-1-legal reduce schedule launches
# one warp.
set(TCA_GOLDEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../internal/cg/cxx/testdata)

add_library(tca_golden_matmul OBJECT ${TCA_GOLDEN_DIR}/matmul_store.cu)
set_target_properties(tca_golden_matmul PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_golden_matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_golden_matmul PRIVATE tcacomp_runtime CUDA::cudart)

add_library(tca_golden_reduce OBJECT ${TCA_GOLDEN_DIR}/reduce_warp.cu)
set_target_properties(tca_golden_reduce PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_golden_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_golden_reduce PRIVATE tcacomp_runtime CUDA::cudart)
