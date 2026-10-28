// dup_expr runner — GVN dedup coverage: out[i][j] = sum_k f(x[i][j][k]) *
// z[i][j][k], f(v) = (v*v + v*v) * gain + (v*v + v*v) * bias (repeated
// subexprs; the z weight carries the map into the WARP-reducer schedule
// class — see the kernel file for the recorded wall).
#include "common.cuh"

void DupExprLaunch(const tca::Tensor<float, 3u>& x, const tca::Tensor<float, 3u>& z,
                   tca::Tensor<float, 2u>& out, float gain, float bias,
                   cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dR = 256u, dC = 256u, dK = 8u;
  const float kGain = 1.5f, kBias = 0.25f;
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
  DupExprLaunch(xT, zT, outT, kGain, kBias, 0u);
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
      const float v = hX[i * dK + k];
      const float w = hZ[i * dK + k];
      acc += ((v * v + v * v) * kGain + (v * v + v * v) * kBias) * w;
    }
    refSum += static_cast<double>(acc);
    const double d =
        std::fabs(static_cast<double>(hOut[i]) - static_cast<double>(acc));
    if (d > maxdiff) maxdiff = d;
  }

  std::printf("KERNEL DupExpr\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u &&
      !BenchReport(bench, [&] { DupExprLaunch(xT, zT, outT, kGain, kBias, 0u); })) {
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
