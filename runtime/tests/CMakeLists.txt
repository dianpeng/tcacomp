# Runtime test targets (TCACOMP_RUNTIME_TESTS=ON, doc/rt.md §8.1-8.2):
# the plain-g++ CPU unit layer + golden compile-conformance objects.

# CPU unit layer: plain C++20, no GPU, no gtest, no linked libraries (the
# INTERFACE link only carries the include directory).
add_executable(tcacomp_cpu_tests cpu_tests.cpp)
target_link_libraries(tcacomp_cpu_tests PRIVATE tcacomp_runtime)
target_compile_options(tcacomp_cpu_tests PRIVATE -Wall -Wextra)

# Golden compile conformance: the two committed goldens must compile as sm_86
# objects against this runtime (compile-only; the linker surface is the
# launcher symbols only). Mirrors the generated matmul_store_cmake.txt — plus
# the CUDAToolkit lookup that provides the CUDA::cudart imported target
# (`LANGUAGES CUDA` alone does not define it in CMake 4.x).
find_package(CUDAToolkit REQUIRED)
set(TCA_GOLDEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../internal/cg/cxx/testdata)

add_library(tca_golden_matmul OBJECT ${TCA_GOLDEN_DIR}/matmul_store.cu)
set_target_properties(tca_golden_matmul PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_golden_matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_golden_matmul PRIVATE tcacomp_runtime CUDA::cudart)

add_library(tca_golden_reduce OBJECT ${TCA_GOLDEN_DIR}/reduce_warp.cu)
set_target_properties(tca_golden_reduce PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(tca_golden_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(tca_golden_reduce PRIVATE tcacomp_runtime CUDA::cudart)
