// scale_copy runner — elementwise affine map + tiled store:
// out[i][j] = sum_k (gain * x[i][j][k] + bias) * z[i][j][k]; the z weight
// carries the map into the WARP-reducer schedule class (see the kernel
// file for the recorded wall).
#include "common.cuh"

// e4: the const trailing reduced extent the compiler now bakes into the
// kernel (see the kernel file / design/fix/fix6.md D-fix6-02 — the emitted
// Launch wrapper prepends uint32_t e<extent> for const dims).
void ScaleCopyLaunch(uint32_t e4, const tca::Tensor<float, 3u>& x, const tca::Tensor<float, 3u>& z,
                     tca::Tensor<float, 2u>& out, float gain, float bias,
                     cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dR = 512u, dC = 512u, dK = 4u;
  const float kGain = 1.25f, kBias = -0.125f;
  const unsigned nX = dR * dC * dK, nOut = dR * dC;

  float* hX = static_cast<float*>(std::malloc(nX * sizeof(float)));
  float* hZ = static_cast<float*>(std::malloc(nX * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hX == nullptr || hZ == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nX; ++i) hX[i] = elem(i);
  for (unsigned i = 0u; i < nX; ++i) hZ[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dX = nullptr;
  float* dZ = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dX, nX * sizeof(float)));
  CUDA_OK(cudaMalloc(&dZ, nX * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dX, hX, nX * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dZ, hZ, nX * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto xT = tca::Tensor<float, 3u>::From(dX, dR, dC, dK);
  auto zT = tca::Tensor<float, 3u>::From(dZ, dR, dC, dK);
  auto outT = tca::Tensor<float, 2u>::From(dOut, dR, dC);
  ScaleCopyLaunch(dK, xT, zT, outT, kGain, kBias, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double gpuSum = 0.0, refSum = 0.0, maxdiff = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    gpuSum += static_cast<double>(hOut[i]);
  }
  for (unsigned i = 0u; i < nOut; ++i) {
    float acc = 0.0f;
    for (unsigned k = 0u; k < dK; ++k) {
      acc += (kGain * hX[i * dK + k] + kBias) * hZ[i * dK + k];
    }
    refSum += static_cast<double>(acc);
    const double d =
        std::fabs(static_cast<double>(hOut[i]) - static_cast<double>(acc));
    if (d > maxdiff) maxdiff = d;
  }

  std::printf("KERNEL ScaleCopy\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u &&
      !BenchReport(bench, [&] { ScaleCopyLaunch(dK, xT, zT, outT, kGain, kBias, 0u); })) {
    return 3;
  }

  std::free(hX);
  std::free(hZ);
  std::free(hOut);
  cudaFree(dX);
  cudaFree(dZ);
  cudaFree(dOut);
  return 0;
}
