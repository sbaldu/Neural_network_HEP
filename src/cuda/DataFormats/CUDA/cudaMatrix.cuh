
#ifndef cudaMatrix_h
#define cudaMatrix_h

#pragma once

#include <iostream>

using Bytes = std::size_t;

template <typename T>
struct matrix_t {
  int rows;
  int cols;
  T* data;

  matrix_t(int n_rows, int n_cols) : rows{n_rows}, cols{n_cols} {
    const Bytes bytes{rows * cols * sizeof(T)};
    cudaMalloc(&data, bytes);
  }
  matrix_t(int n_rows, int n_cols, T* data) : data{data}, rows{n_rows}, cols{n_cols} {
    const Bytes bytes{rows * cols * sizeof(T)};
    cudaMalloc(&data, bytes);
    cudaMemcpy(&(this->data), data, bytes, cudaMemcpyHostToDevice);
  }

  ~matrix_t() { cudaFree(data); }

  __device__ T& operator[](int index) { return data[index]; }
  __device__ const T& operator[](int index) const { return data[index]; }
};

#endif
