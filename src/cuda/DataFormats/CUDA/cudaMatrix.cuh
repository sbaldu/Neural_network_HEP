
#ifndef cudaMatrix_h
#define cudaMatrix_h

#pragma once

#include <iostream>

using Bytes = std::size_t;

template <typename T>
struct cudaMatrix {
private:
  int rows;
  int cols;
  T* data;

public:
  cudaMatrix(int n_rows, int n_cols) : rows{n_rows}, cols{n_cols} {
    const Bytes bytes{rows * cols * sizeof(T)};
    cudaMalloc(&data, bytes);
  }
  cudaMatrix(int n_rows, int n_cols, T* data) : data{data}, rows{n_rows}, cols{n_cols} {
    const Bytes bytes{rows * cols * sizeof(T)};
    cudaMalloc(&data, bytes);
    cudaMemcpy(&(this->data), data, bytes, cudaMemcpyHostToDevice);
  }

  ~cudaMatrix() { cudaFree(data); }

  T* data() { return data; }
  const T* data() const { return data; }

  __device__ int rows() const { return rows; }
  __device__ int cols() const { return cols; }

  __device__ T& operator[](int index) { return data[index]; }
  __device__ const T& operator[](int index) const { return data[index]; }
};

#endif
