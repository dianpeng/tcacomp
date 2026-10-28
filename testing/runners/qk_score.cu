// qk_score runner — attention scores Q@K^T with scale:
// out[i][n] = scale * sum_k q[i,k]*kk[n,k]  (scale = 1/sqrt(D)).
#include "common.cuh"

void QkScoreLaunch(const tca::Tensor<float, 2u>& q, const tca::Tensor<float, 2u>& kk,
                   tca::Tensor<float, 2u>& out, float scale, cudaStream_t tca_stream);

int main(int argc, char** argv) {
  const unsigned bench = ParseBenchArgs(argc, argv);
  const unsigned dH = 128u, dD = 64u, dN = 192u;
  const float kScale = 0.125f;  // 1 / sqrt(64)
  const unsigned nQ = dH * dD, nKk = dN * dD, nOut = dH * dN;

  float* hQ = static_cast<float*>(std::malloc(nQ * sizeof(float)));
  float* hKk = static_cast<float*>(std::malloc(nKk * sizeof(float)));
  float* hOut = static_cast<float*>(std::malloc(nOut * sizeof(float)));
  if (hQ == nullptr || hKk == nullptr || hOut == nullptr) return 2;
  for (unsigned i = 0u; i < nQ; ++i) hQ[i] = elem(i);
  for (unsigned i = 0u; i < nKk; ++i) hKk[i] = elem(i);
  for (unsigned i = 0u; i < nOut; ++i) hOut[i] = -12345.0f;  // canary

  float* dQ = nullptr;
  float* dKk = nullptr;
  float* dOut = nullptr;
  CUDA_OK(cudaMalloc(&dQ, nQ * sizeof(float)));
  CUDA_OK(cudaMalloc(&dKk, nKk * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut, nOut * sizeof(float)));
  CUDA_OK(cudaMemcpy(dQ, hQ, nQ * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dKk, hKk, nKk * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dOut, hOut, nOut * sizeof(float), cudaMemcpyHostToDevice));

  auto qT = tca::Tensor<float, 2u>::From(dQ, dH, dD);
  auto kkT = tca::Tensor<float, 2u>::From(dKk, dN, dD);
  auto outT = tca::Tensor<float, 2u>::From(dOut, dH, dN);
  QkScoreLaunch(qT, kkT, outT, kScale, 0u);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(hOut, dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned canary = 0u;
  double gpuSum = 0.0, refSum = 0.0, maxdiff = 0.0;
  for (unsigned i = 0u; i < nOut; ++i) {
    if (hOut[i] == -12345.0f) ++canary;
    gpuSum += static_cast<double>(hOut[i]);
  }
  for (unsigned h = 0u; h < dH; ++h) {
    for (unsigned n = 0u; n < dN; ++n) {
      float acc = 0.0f;
      for (unsigned k = 0u; k < dD; ++k) {
        acc += hQ[h * dD + k] * hKk[n * dD + k];
      }
      acc *= kScale;
      const unsigned o = h * dN + n;
      refSum += static_cast<double>(acc);
      const double d =
          std::fabs(static_cast<double>(hOut[o]) - static_cast<double>(acc));
      if (d > maxdiff) maxdiff = d;
    }
  }

  std::printf("KERNEL QkScore\n");
  PrintCanary(canary);
  PrintChecksum(gpuSum, refSum);
  PrintMaxdiff(maxdiff);
  PrintSamples(hOut, nOut);

  if (bench > 0u &&
      !BenchReport(bench, [&] { QkScoreLaunch(qT, kkT, outT, kScale, 0u); })) {
    return 3;
  }

  std::free(hQ);
  std::free(hKk);
  std::free(hOut);
  cudaFree(dQ);
  cudaFree(dKk);
  cudaFree(dOut);
  return 0;
}
