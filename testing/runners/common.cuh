// common.cuh — shared runner prologue for the tcacomp testing suite.
// Mirrors internal/e2e/e2e_test.go's runnerPrologue, factored across the 14
// runners: deterministic input formula, CUDA_OK macro, machine-parseable
// protocol printers, --bench argv parse, and the bench timer.
#ifndef TCA_TESTING_RUNNERS_COMMON_CUH_
#define TCA_TESTING_RUNNERS_COMMON_CUH_

#include "tca/runtime.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Deterministic input formula — MUST be identical in every runner's input
// fill AND its CPU reference: v(i) = float((i * 2654435761u) % 1000u) /
// 1000.0f (the u32 wraparound is the protocol's pinned convention).
static float elem(unsigned i) {
  return static_cast<float>((i * 2654435761u) % 1000u) / 1000.0f;
}

#define CUDA_OK(call)                                                     \
  do {                                                                    \
    if ((call) != cudaSuccess) {                                          \
      std::printf("CUDA_FAIL %s:%d\n", __FILE__, __LINE__);               \
      return 3;                                                           \
    }                                                                     \
  } while (0)

// ParseBenchArgs scans argv for "--bench <iters>"; 0 = no bench. The value
// crosses from the orchestrator's CLI into this binary — strtoul with an
// explicit 1..100000 bound (the threat register's T-t4u-03 mitigation).
static unsigned ParseBenchArgs(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--bench") == 0) {
      char* end = nullptr;
      const unsigned long v = std::strtoul(argv[i + 1], &end, 10);
      if (end == argv[i + 1] || *end != '\0' || v < 1ul || v > 100000ul) {
        std::fprintf(stderr, "bad --bench value %s (want 1..100000)\n",
                     argv[i + 1]);
        std::exit(2);
      }
      return static_cast<unsigned>(v);
    }
  }
  return 0u;
}

// Protocol printers — the exact machine-parseable stdout lines the
// orchestrator's verifier parses (superset of internal/e2e
// ParseFixtureOutput).
static void PrintCanary(unsigned n) { std::printf("CANARY %u\n", n); }
static void PrintChecksum(double gpuSum, double refSum) {
  std::printf("CHECKSUM %.17g\n", gpuSum);
  std::printf("REF_CHECKSUM %.17g\n", refSum);
}
static void PrintMaxdiff(double d) { std::printf("MAXDIFF %.17g\n", d); }
static void PrintSamples(const float* h, unsigned n) {
  const unsigned k = n < 8u ? n : 8u;
  std::printf("SAMPLES");
  for (unsigned i = 0u; i < k; ++i) {
    std::printf(" %a", static_cast<double>(h[i]));
  }
  std::printf("\n");
}

// BenchReport times N launch iterations; the per-iteration
// cudaDeviceSynchronize is INSIDE the timed region (end-to-end latency is
// the benchmark). Returns false on a sync failure (CUDA_FAIL printed).
template <typename F>
static bool BenchReport(unsigned iters, F&& launch) {
  double total = 0.0, best = 1e30, worst = 0.0;
  for (unsigned i = 0u; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    launch();
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::printf("CUDA_FAIL %s:%d\n", __FILE__, __LINE__);
      return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    total += us;
    if (us < best) best = us;
    if (us > worst) worst = us;
  }
  std::printf("BENCH_US %g %g %g\n", total / static_cast<double>(iters), best,
              worst);
  return true;
}

#endif  // TCA_TESTING_RUNNERS_COMMON_CUH_
