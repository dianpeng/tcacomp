// tca/launch.h — hard launch-grammar validation (doc/rt.md §5).
//
// TRAP, never clamp: the map arithmetic divides by these grammars
// unconditionally (the portion decomposition assumes them). The SMEM ceiling
// is always QUERIED from the device — never hardcode sm_86's 101,376 B
// (runtime_objects.md §1.1; the occupancy-API off-by-1KB bug note).
#ifndef TCA_RT_LAUNCH_H_
#define TCA_RT_LAUNCH_H_

#include <cuda_runtime.h>

#include "tca/check.h"
#include "tca/mapping.h"  // kPortionCTA / kPortionWarp

namespace tca {
namespace rt {

inline constexpr unsigned kMinThreadsPerCTA = 256u;  // 8 warps = the minimum warp-portion config
inline constexpr unsigned kReservedSmemPerBlock = 1024u;  // driver-reserved per-block bytes

// Asserts the tiling grammar over the 1-D normalized launch
// (runtime_objects.md §1.5: gridDim.x = total CTAs, blockDim.x = threads/CTA):
//  - grid.x positive and a multiple of PortionCTA = 10
//  - block.x >= 256, a whole-warp multiple (32), warp count a multiple of
//    PortionWarp = 8
//  - grid.y/z and block.y/z == 1 (the 1-D normalized launch contract)
//  - smemBytes within the device's opt-in per-block ceiling minus the
//    reserved bytes.
//
// The reducer's static __shared__ scratch shares the same per-block ceiling:
// it is enforced by the driver at launch and surfaced through
// TCA_CUDA_CHECK at the call site (cudaFuncSetAttribute / launch errors).
inline void ValidateLaunch(dim3 grid, dim3 block, unsigned smemBytes) {
  TCA_REQUIRE(grid.x > 0u, "gridDim.x must be positive");
  TCA_REQUIRE(grid.x % kPortionCTA == 0u,
              "gridDim.x must be a multiple of PortionCTA=10");
  TCA_REQUIRE(grid.y == 1u && grid.z == 1u, "launch must be 1-D normalized (grid.y/z == 1)");
  TCA_REQUIRE(block.x >= kMinThreadsPerCTA, "blockDim.x must be >= 256 threads");
  TCA_REQUIRE(block.x % 32u == 0u, "blockDim.x must be a whole-warp multiple (32)");
  TCA_REQUIRE((block.x / 32u) % kPortionWarp == 0u,
              "warp count must be a multiple of PortionWarp=8");
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
