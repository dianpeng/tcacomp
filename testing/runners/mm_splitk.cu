// mm_splitk runner — plain matmul [64,128]x[128,64] -> [64,64], the
// fix4 §3.5 split-k schedule class.
#include "common.cuh"

void MmSplitkLaunch(const tca::Tensor<float, 2u>& lhs, const tca::Tensor<float, 2u>& rhs,
                   tca::Tensor<float, 2u>& out, cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dM = 64u, dK = 128u, dN = 64u;
  const unsigned nLhs = dM * dK, nRhs = dK * dN, nOut = dM * dN;

  float* hLhs = static_cast<float*>(std::malloc(nLhs * sizeof(float)));
  float* hRhs = static_cast<float*>(std::malloc(nRhs * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hLhs == nullptr || hRhs == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nLhs; ++i) hLhs[i] = elem(i);
  for (unsigned i = 0u; i < nRhs; ++i) hRhs[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dLhs = nullptr;
  float* dRhs = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dLhs, nLhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dRhs, nRhs * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dLhs, hLhs, nLhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dRhs, hRhs, nRhs * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto lhsT = tca::Tensor<float, 2u>::From(dLhs, dM, dK);
  auto rhsT = tca::Tensor<float, 2u>::From(dRhs, dK, dN);
  auto outT = tca::Tensor<float, 2u>::From(dOut, dM, dN);
  MmSplitkLaunch(lhsT, rhsT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double gpuSum = 0.0, refSum = 0.0, maxdiff = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    gpuSum += static_cast<double>(hOut[i]);
  }
  for (unsigned m = 0u; m < dM; ++m) {
    for (unsigned n = 0u; n < dN; ++n) {
      float acc = 0.0f;
      for (unsigned k = 0u; k < dK; ++k) {
        acc += hLhs[m * dK + k] * hRhs[k * dN + n];
      }
      const unsigned o = m * dN + n;
      refSum += static_cast<double>(acc);
      const double d =
          std::fabs(static_cast<double>(hOut[o]) - static_cast<double>(acc));
      if (d > maxdiff) maxdiff = d;
    }
  }

  std::printf("KERNEL MmSplitk\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u &&
      !BenchReport(bench, [&] { MmSplitkLaunch(lhsT, rhsT, outT, 0u); })) {
    return 3;
  }

  std::free(hLhs);
  std::free(hRhs);
  std::free(hOut);
  cudaFree(dLhs);
  cudaFree(dRhs);
  cudaFree(dOut);
  return 0;
}
