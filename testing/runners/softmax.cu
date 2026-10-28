// softmax runner — the softmax reduction core: s[i] = sum_j exp(x[i][j])
// over x[R,C] -> [R] (the per-row normalizer; see the kernel file for the
// v0 expressibility walls that dropped the max-subtract and divide stages).
#include "common.cuh"

void SoftmaxLaunch(const tca::Tensor<float, 2u>& x, tca::Tensor<float, 1u>& s,
                   cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dR = 128u, dC = 256u;
  const unsigned nX = dR * dC, nOut = dR;

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

  auto xT = tca::Tensor<float, 2u>::From(dX, dR, dC);
  auto outT = tca::Tensor<float, 1u>::From(dOut, dR);
  SoftmaxLaunch(xT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double gpuSum = 0.0, refSum = 0.0, maxdiff = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    gpuSum += static_cast<double>(hOut[i]);
  }
  for (unsigned i = 0u; i < dR; ++i) {
    float acc = 0.0f;
    for (unsigned j = 0u; j < dC; ++j) {
      acc += expf(hX[i * dC + j]);
    }
    refSum += static_cast<double>(acc);
    const double d =
        std::fabs(static_cast<double>(hOut[i]) - static_cast<double>(acc));
    if (d > maxdiff) maxdiff = d;
  }

  std::printf("KERNEL Softmax\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u && !BenchReport(bench, [&] { SoftmaxLaunch(xT, outT, 0u); })) {
    return 3;
  }

  std::free(hX);
  std::free(hOut);
  cudaFree(dX);
  cudaFree(dOut);
  return 0;
}
