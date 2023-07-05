
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../../../../src/cuda/DataFormats/VectorKernels.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test vec_add_1") {
  const int N{10};
  std::vector<int> v1(N), v2(N), sum(N);
  std::iota(v1.begin(), v1.end(), 1.);
  std::iota(v2.begin(), v2.end(), 1.);

  const int size{sizeof(int) * N};
  int *d_v1, *d_v2, *d_sum;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);
  cudaMalloc(&d_sum, size);

  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_add<<<std::ceil(N/(float)(256)), 256>>>(d_v1, d_v2, d_sum, N);
  cudaMemcpy(sum.data(), d_sum, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(sum[i] == 2*(i+1));
  }

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_sum);
}

TEST_CASE("Test vec_add_2") {
  const int N{10};
  std::vector<int> v1(N), v2(N);
  std::iota(v1.begin(), v1.end(), 1.);
  std::iota(v2.begin(), v2.end(), 1.);

  const int size{sizeof(int) * N};
  int *d_v1, *d_v2;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);

  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_add<<<std::ceil(N/(float)(256)), 256>>>(d_v1, d_v2, N);
  cudaMemcpy(v1.data(), d_v1, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(v1[i] == 2*(i+1));
  }

  cudaFree(d_v1);
  cudaFree(d_v2);
}

TEST_CASE("Test vec_sub_1") {
  const int N{10};
  std::vector<int> v1(N), v2(N), sum(N);
  std::iota(v1.begin(), v1.end(), N);
  std::iota(v2.begin(), v2.end(), 0.);

  const int size{sizeof(int) * N};
  int *d_v1, *d_v2, *d_sum;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);
  cudaMalloc(&d_sum, size);

  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_sub<<<std::ceil(N/(float)(256)), 256>>>(d_v1, d_v2, d_sum, N);
  cudaMemcpy(sum.data(), d_sum, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(sum[i] == N);
  }

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_sum);
}

TEST_CASE("Test vec_sub_2") {
  const int N{10};
  std::vector<int> v1(N), v2(N);
  std::iota(v1.begin(), v1.end(), N);
  std::iota(v2.begin(), v2.end(), 0.);

  const int size{sizeof(int) * N};
  int *d_v1, *d_v2;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);

  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_sub<<<std::ceil(N/(float)(256)), 256>>>(d_v1, d_v2, N);
  cudaMemcpy(v1.data(), d_v1, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(v1[i] == N);
  }

  cudaFree(d_v1);
  cudaFree(d_v2);
}

TEST_CASE("Test vec_multiply") {
  const int N{10};
  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 0.);
  const int constant{5};

  const int size{sizeof(int) * N};
  int *d_v;
  cudaMalloc(&d_v, size);

  cudaMemcpy(d_v, v.data(), size, cudaMemcpyHostToDevice);
  vec_multiply<<<std::ceil(N/(float)(256)), 256>>>(d_v, constant, N);
  cudaMemcpy(v.data(), d_v, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(v[i] == constant * i);
  }

  cudaFree(d_v);
}

TEST_CASE("Test vec_divide") {
  const int N{10};
  std::vector<double> v(N);
  std::iota(v.begin(), v.end(), 0.);
  const double constant{5.};

  const int size{sizeof(double) * N};
  double *d_v;
  cudaMalloc(&d_v, size);

  cudaMemcpy(d_v, v.data(), size, cudaMemcpyHostToDevice);
  vec_divide<<<std::ceil(N/(float)(256)), 256>>>(d_v, constant, N);
  cudaMemcpy(v.data(), d_v, size, cudaMemcpyDeviceToHost);

  for (int i{}; i < N; ++i) {
	CHECK(v[i] == i / constant);
  }

  cudaFree(d_v);
}
