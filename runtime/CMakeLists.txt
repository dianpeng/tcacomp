# TCA runtime library (doc/rt.md §1) — header-only INTERFACE target.
#
# Consumers (the compiler-generated CMakeLists) add CUDA::cudart themselves
# and pin CMAKE_CUDA_ARCHITECTURES (86); this library stays arch-generic.
# No .cu TUs — everything header-inline.
cmake_minimum_required(VERSION 3.24)
project(tcacomp_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(tcacomp_runtime INTERFACE)
target_include_directories(tcacomp_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# CPU unit layer + golden compile-conformance targets (OFF by default: a plain
# `cmake -S runtime -B <dir>` configure stays INTERFACE-only).
option(TCACOMP_RUNTIME_TESTS "Build runtime unit + conformance targets" OFF)
if(TCACOMP_RUNTIME_TESTS)
  add_subdirectory(tests)
endif()

