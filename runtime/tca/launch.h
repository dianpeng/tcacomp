// tca/launch.h — hard launch-grammar validation (doc/fix/fix4.md §4.3;
// supersedes the pre-fix4 grammar of doc/rt.md §5).
//
// TRAP, never clamp: the launch values themselves are compiler-derived
// constants (fix4 §5.1) — validation is defense-in-depth. The SMEM ceiling
// is always QUERIED from the device — never hardcode sm_86's 101,376 B
// per block (nor sm_80's 163 KB; runtime_objects.md §1.1; the occupancy-API
// off-by-1KB bug note, NVBUG 5209234).
#ifndef TCA_RT_LAUNCH_H_
#define TCA_RT_LAUNCH_H_

#include <cuda_runtime.h>

#include "tca/check.h"

namespace tca {
namespace rt {

inline constexpr unsigned kMaxWarpsPerCTA = 32u;  // hardware bound (fix4 §2.1)
inline constexpr unsigned kReservedSmemPerBlock = 1024u;  // driver-reserved per-block bytes

// Asserts the factorization launch pre-conditions (fix4 §4.3) over the 1-D
// normalized launch:
//  - grid.x > 0 and grid.x <= maxCTA — the compiler-EMITTED policy constant
//    (fix4 §2.1 MaxCTA); the parameter deliberately has NO default argument,
//    the launcher must pass it (checked > 0, fail-closed)
//  - grid.y/z == 1 (1-D normalized launch contract)
//  - block.x >= 32, a whole-warp multiple (32), and at most
//    kMaxWarpsPerCTA = 32 warps
//  - block.y/z == 1
//  - smemBytes within the device's opt-in per-block ceiling minus the
//    reserved bytes (QUERIED — never hardcoded).
//
// The dynamic-SMEM opt-in (cudaFuncSetAttribute) and the launch itself are
// enforced by the driver; errors surface through TCA_CUDA_CHECK at the
// call site.
inline void ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes, unsigned maxCTA) {
  TCA_REQUIRE(grid.x > 0u, "gridDim.x must be positive");
  TCA_REQUIRE(grid.y == 1u && grid.z == 1u, "launch must be 1-D normalized (grid.y/z == 1)");
  TCA_REQUIRE(maxCTA > 0u, "maxCTA must be positive (the compiler-emitted policy constant)");
  TCA_REQUIRE(grid.x <= maxCTA, "gridDim.x must not exceed the compiler-emitted MaxCTA");
  TCA_REQUIRE(block.x >= 32u, "blockDim.x must be at least one whole warp (32 threads)");
  TCA_REQUIRE(block.x % 32u == 0u, "blockDim.x must be a whole-warp multiple (32)");
  TCA_REQUIRE(block.x / 32u <= kMaxWarpsPerCTA,
              "blockDim.x must not exceed kMaxWarpsPerCTA=32 warps per CTA");
  TCA_REQUIRE(block.y == 1u && block.z == 1u, "launch must be 1-D normalized (block.y/z == 1)");

  // Query the CURRENT device (this toolkit's headers expose the 3-arg
  // device-explicit form of cudaDeviceGetAttribute).
  int device = 0;
  TCA_CUDA_CHECK(cudaGetDevice(&device));
  int maxOptin = 0;
  TCA_CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TCA_REQUIRE(maxOptin > static_cast<int>(kReservedSmemPerBlock),
              "device shared memory ceiling query returned an implausible value");
  TCA_REQUIRE(smemBytes <= static_cast<unsigned>(maxOptin) - kReservedSmemPerBlock,
              "shared memory bytes exceed the device per-block ceiling");
}

}  // namespace rt
}  // namespace tca

#endif  // TCA_RT_LAUNCH_H_
