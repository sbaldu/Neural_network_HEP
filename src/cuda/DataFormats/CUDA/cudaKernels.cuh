
#ifndef kernels_h
#define kernels_h

#pragma once

#include <cassert>
#include <cstdint>

#include "cudaMatrix.cuh"

template <typename T1, typename T2, typename T3>
__global__ void vec_add(const T1* a, const T2* b, T3* c, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

template <typename T1, typename T2>
__global__ void vec_add(T1* a, const T2* b, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] += b[index];
  }
}

template <typename T1, typename T2, typename T3>
__global__ void vec_sub(const T1* a, const T2* b, T3* c, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    c[index] = a[index] - b[index];
  }
}

template <typename T1, typename T2>
__global__ void vec_sub(T1* a, const T2* b, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] -= b[index];
  }
}

template <typename T, typename E>
__global__ void vec_multiply(T* a, E constant, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] *= constant;
  }
}

template <typename T, typename E>
__global__ void vec_divide(T* a, E constant, std::size_t n) {
  uint32_t index{threadIdx.x + blockIdx.x * blockDim.x};

  if (index < n) {
    a[index] /= constant;
  }
}

template <typename T1, typename T2, typename T3>
__global__ void matrix_multiply(const cudaMatrix<T1>* a,
                                const cudaMatrix<T2>* b,
                                cudaMatrix<T3>* c,
                                std::size_t block_size) {
  // Allocate memory on shared memory
  extern __shared__ int shared[];
  T1* s_a{shared};
  T2* s_b{&shared[block_size * block_size]};

  uint32_t row{blockIdx.y * blockDim.y + threadIdx.y};
  uint32_t col{blockIdx.x * blockDim.x + threadIdx.x};

  // Temporary variable containing the result of the moltiplication
  T3 temp{};

  assert(a->cols() == b->rows());
  for (std::size_t i{}; i < a->cols(); i += blockDim.x) {
    // Fill the arrays in shared memory
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = (*a)[row * a->cols() + threadIdx.x + i];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = (*b)[(threadIdx.y + i) * b->cols() + col];

    __syncthreads();

    for (std::size_t j{}; j < blockDim.x; ++j) {
      temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    (*c)[row * b->cols() + col] = temp;
  }
}

#endif
