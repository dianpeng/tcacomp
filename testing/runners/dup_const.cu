// dup_const runner — tail const-dedup coverage: out[i][j] = sum_k p(x[i][j][k])
// with p the four-literal-reuse polynomial (the GELU constant family).
#include "common.cuh"

void DupConstLaunch(const tca::Tensor<float, 3u>& x, tca::Tensor<float, 2u>& out,
                    cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dR = 256u, dC = 256u, dK = 8u;
  const unsigned nX = dR * dC * dK, nOut = dR * dC;

  float* hX = static_cast<float*>(std::malloc(nX * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hX == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nX; ++i) hX[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dX = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dX, nX * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dX, hX, nX * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto xT = tca::Tensor<float, 3u>::From(dX, dR, dC, dK);
  auto outT = tca::Tensor<float, 2u>::From(dOut, dR, dC);
  DupConstLaunch(xT, outT, 0u);
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
      acc += 0.5f * (v + 0.5f * v * v) * (1.0f + 0.044715f * v) +
             tanhf(0.79788456f * (v + 0.044715f * v) +
                   0.79788456f * 0.5f * v) *
                 (1.0f - 0.5f);
    }
    refSum += static_cast<double>(acc);
    const double d =
        std::fabs(static_cast<double>(hOut[i]) - static_cast<double>(acc));
    if (d > maxdiff) maxdiff = d;
  }

  std::printf("KERNEL DupConst\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u && !BenchReport(bench, [&] { DupConstLaunch(xT, outT, 0u); })) {
    return 3;
  }

  std::free(hX);
  std::free(hOut);
  cudaFree(dX);
  cudaFree(dOut);
  return 0;
}
