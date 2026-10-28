// transpose runner — written as-clause permutation + reduce:
// out[j][i] = sum_k x[i][j][k] over x[R,C,K] -> [C,R].
#include "common.cuh"

void TransposeLaunch(const tca::Tensor<float, 3u>& x, tca::Tensor<float, 2u>& out,
                     cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dR = 192u, dC = 96u, dK = 8u;
  const unsigned nX = dR * dC * dK, nOut = dC * dR;

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
  auto outT = tca::Tensor<float, 2u>::From(dOut, dC, dR);
  TransposeLaunch(xT, outT, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double gpuSum = 0.0, refSum = 0.0, maxdiff = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    gpuSum += static_cast<double>(hOut[i]);
  }
  for (unsigned i = 0u; i < dR; ++i) {
    for (unsigned j = 0u; j < dC; ++j) {
      float acc = 0.0f;
      for (unsigned k = 0u; k < dK; ++k) {
        acc += hX[(i * dC + j) * dK + k];
      }
      const unsigned o = j * dR + i;  // out[j][i]
      refSum += static_cast<double>(acc);
      const double d =
          std::fabs(static_cast<double>(hOut[o]) - static_cast<double>(acc));
      if (d > maxdiff) maxdiff = d;
    }
  }

  std::printf("KERNEL Transpose\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u && !BenchReport(bench, [&] { TransposeLaunch(xT, outT, 0u); })) {
    return 3;
  }

  std::free(hX);
  std::free(hOut);
  cudaFree(dX);
  cudaFree(dOut);
  return 0;
}
