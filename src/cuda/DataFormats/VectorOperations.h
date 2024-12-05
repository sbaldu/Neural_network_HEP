
#pragma once

#include <algorithm>
#include <concepts>
#include <ostream>
#include <vector>

#include "VectorKernels.h"
#include "Matrix.h"

// Overload sum operator for vectors
template <typename T>
std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2) {
  const size_t N{v1.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T *d_v1, *d_v2, *d_res;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);
  cudaMalloc(&d_res, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_add<<<grid_size, block_size>>>(d_v1, d_v2, d_res, N);
  cudaMemcpy(result.data(), d_res, size, cudaMemcpyDeviceToHost);

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_res);

  return result;
}

// Overload subtraction
template <typename T>
std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
  const size_t N{v1.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T *d_v1, *d_v2, *d_res;
  cudaMalloc(&d_v1, size);
  cudaMalloc(&d_v2, size);
  cudaMalloc(&d_res, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);
  vec_sub<<<grid_size, block_size>>>(d_v1, d_v2, d_res, N);
  cudaMemcpy(result.data(), d_res, size, cudaMemcpyDeviceToHost);

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_res);

  return result;
}

// Overload product of vector with constant
template <typename T, std::convertible_to<T> E>
std::vector<T> operator*(E constant, std::vector<T> vec) {
  const size_t N{vec.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T* d_res;
  cudaMalloc(&d_res, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_res, vec.data(), size, cudaMemcpyHostToDevice);
  vec_multiply<<<grid_size, block_size>>>(d_res, constant, N);
  cudaMemcpy(result.data(), d_res, size, cudaMemcpyDeviceToHost);

  cudaFree(d_res);

  return result;
}

// Overload division of vector with constant
template <typename T, std::convertible_to<T> E>
std::vector<T> operator/(std::vector<T> vec, E constant) {
  const size_t N{vec.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T* d_res;
  cudaMalloc(&d_res, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_res, vec.data(), size, cudaMemcpyHostToDevice);
  vec_divide<<<grid_size, block_size>>>(d_res, constant, N);
  cudaMemcpy(result.data(), d_res, size, cudaMemcpyDeviceToHost);

  cudaFree(d_res);

  return result;
}

// Overload of operator+= with matrices
template <typename T>
void operator+=(std::vector<T>& vec, const Matrix<T>& m) {
  const size_t N{vec.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T *d_vec, *d_mat;
  cudaMalloc(&d_vec, size);
  cudaMalloc(&d_mat, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_vec, vec.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, (m.data()).data(), size, cudaMemcpyHostToDevice);
  vec_add<<<grid_size, block_size>>>(d_vec, d_mat, N);
  cudaMemcpy(vec.data(), d_vec, size, cudaMemcpyDeviceToHost);

  cudaFree(d_vec);
  cudaFree(d_mat);
}

// Overload of operator-= with matrices
template <typename T>
void operator-=(std::vector<T>& vec, const Matrix<T>& m) {
  const size_t N{vec.size()};
  std::vector<T> result(N);

  const size_t size{sizeof(T) * N};

  // Allocate on device
  T *d_vec, *d_mat;
  cudaMalloc(&d_vec, size);
  cudaMalloc(&d_mat, size);

  // Create working division
  const int block_size{32};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};

  // Launch kernel
  cudaMemcpy(d_vec, vec.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, (m.data()).data(), size, cudaMemcpyHostToDevice);
  vec_sub<<<grid_size, block_size>>>(d_vec, d_mat, N);
  cudaMemcpy(vec.data(), d_vec, size, cudaMemcpyDeviceToHost);

  cudaFree(d_vec);
  cudaFree(d_mat);
}

// Ostream operator for vectors
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << *vec.begin();
  std::for_each(vec.begin() + 1, vec.end(), [&out](T x) {
    out << ',';
    out << x;
  });

  return out;
}
