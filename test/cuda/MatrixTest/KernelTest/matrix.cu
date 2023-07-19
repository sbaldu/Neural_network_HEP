
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "DataFormats/VectorKernels.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

void verify_result(const std::vector<int> &a, const std::vector<int> &b, const std::vector<int> &c, int N) {
  for (int i{}; i < N; ++i) {
    for (int j{}; j < N; ++j) {
      int tmp{};
      for (int k{}; k < N; ++k) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      assert(tmp == c[i * N + j]);
    }
  }

  std::cout << "Success\n";
}

void verify_result(
    const std::vector<int> &a, const std::vector<int> &b, const std::vector<int> &c, int N, int K, int M) {
  for (int i{}; i < N; ++i) {
    for (int j{}; j < M; ++j) {
      int tmp{};
      for (int k{}; k < K; ++k) {
        // Accumulate the partial results
        tmp += a[i * K + k] * b[k * M + j];
      }

      assert(tmp == c[i * M + j]);
    }
  }

  std::cout << "Success\n";
}

TEST_CASE("Square matrix multiplication") {
  const int N{1 << 5};
  std::vector<int> a(N * N), b(N * N), c(N * N);
  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);

  const int size{N * N * sizeof(int)};

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  matrix_t<int> m_a(N, N, d_a);
  matrix_t<int> m_b(N, N, d_b);
  matrix_t<int> m_c(N, N, d_c);

  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_size, grid_size);

  cudaMemcpy(m_a.data, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(m_b.data, b.data(), size, cudaMemcpyHostToDevice);
  const int shared_size{2 * block_size * block_size * sizeof(int)};
  matrix_multiply<<<grid, block, shared_size>>>(m_a, m_b, m_c, block_size);
  cudaMemcpy(c.data(), m_c.data, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  verify_result(a, b, c, N);
}

TEST_CASE("Rectangular matrix multiplication") {
  const int N{1 << 5};
  const int K{1 << 8};
  const int M{1 << 5};

  std::vector<int> a(N * K), b(K * M), c(N * M);
  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);

  const int size_a{N * K * sizeof(int)};
  const int size_b{K * M * sizeof(int)};
  const int size_c{N * M * sizeof(int)};

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  matrix_t<int> m_a(N, K, d_a);
  matrix_t<int> m_b(K, M, d_b);
  matrix_t<int> m_c(N, M, d_c);

  const int block_size{32};
  const int grid_x{(int)std::ceil(M / (float)(block_size))};
  const int grid_y{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_x, grid_y);

  cudaMemcpy(m_a.data, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(m_b.data, b.data(), size_b, cudaMemcpyHostToDevice);
  const int shared_size{2 * block_size * block_size * sizeof(int)};
  matrix_multiply<<<grid, block, shared_size>>>(m_a, m_b, m_c, block_size);
  cudaMemcpy(c.data(), m_c.data, size_c, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  verify_result(a, b, c, N, K, M);
}

TEST_CASE("Alternate rectangular matrix multiplication") {
  const int N{1 << 8};
  const int K{1 << 5};
  const int M{1 << 8};

  std::vector<int> a(N * K), b(K * M), c(N * M);
  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);

  const int size_a{N * K * sizeof(int)};
  const int size_b{K * M * sizeof(int)};
  const int size_c{N * M * sizeof(int)};

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  matrix_t<int> m_a(N, K, d_a);
  matrix_t<int> m_b(K, M, d_b);
  matrix_t<int> m_c(N, M, d_c);

  const int block_size{32};
  const int grid_x{(int)std::ceil(M / (float)(block_size))};
  const int grid_y{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_x, grid_y);

  cudaMemcpy(m_a.data, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(m_b.data, b.data(), size_b, cudaMemcpyHostToDevice);
  const int shared_size{2 * block_size * block_size * sizeof(int)};
  matrix_multiply<<<grid, block, shared_size>>>(m_a, m_b, m_c, block_size);
  cudaMemcpy(c.data(), m_c.data, size_c, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  verify_result(a, b, c, N, K, M);
}
