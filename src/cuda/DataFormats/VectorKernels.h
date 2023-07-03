
#ifndef kernels_h
#define kernels_h

#pragma once

#include <cassert>

template <typename T>
struct matrix_t {
  int rows;
  int cols;
  T* data;
};

template <typename T>
__global__ void vec_add(const T* a, const T* b, T* c, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

template <typename T>
__global__ void vec_add(T* a, const T* b, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] += b[index];
  }
}

template <typename T>
__global__ void vec_sub(const T* a, const T* b, T* c, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    c[index] = a[index] - b[index];
  }
}

template <typename T>
__global__ void vec_sub(T* a, const T* b, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] -= b[index];
  }
}

template <typename T>
__global__ void vec_multiply(T* a, T constant, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] *= constant;
  }
}

template <typename T>
__global__ void vec_divide(T* a, T constant, int n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] /= constant;
  }
}

template <typename T>
__global__ void matrix_vec_multiply(const matrix_t<T> matrix, const T* vec, T* result, int n) {
  // Allocate memory on shared memory
  extern __shared__ int s_a[];
  extern __shared__ int s_b[];

  unsigned int row{blockIdx.y * blockDim.y + threadIdx.y};
  unsigned int col{blockIdx.x * blockDim.x + threadIdx.x};

  // Temporary variable containing the result of the moltiplication
  int temp{};

  assert(m1.cols == n);
  for (int i{}; i < m1.cols; i += blockIdx.x) {
    // Fill the arrays in shared memory
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * m1.cols + threadIdx.x + i];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[threadIdx.y + i];

    // Make sure that the shared arrays are completely filled
    __syncthreads();

    for (int j{}; j < blockDim.x; ++j) {
      temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    c[row * m2.cols + col] = temp;
  }
}

template <typename T>
__global__ void matrix_multiply(const matrix_t<T> a, const matrix_t<T> b, matrix_t<T> c) {
  // Allocate memory on shared memory
  extern __shared__ int s_a[];
  extern __shared__ int s_b[];

  unsigned int row{blockIdx.y * blockDim.y + threadIdx.y};
  unsigned int col{blockIdx.x * blockDim.x + threadIdx.x};

  // Temporary variable containing the result of the moltiplication
  int temp{};

  assert(m1.cols == m2.rows);
  for (int i{}; i < m1.cols; i += blockIdx.x) {
    // Fill the arrays in shared memory
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * m1.cols + threadIdx.x + i];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(threadIdx.y + i) * m2.cols + col];

    __syncthreads();

    for (int j{}; j < blockDim.x; ++j) {
      temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    c[row * m2.cols + col] = temp;
  }
}

#endif
