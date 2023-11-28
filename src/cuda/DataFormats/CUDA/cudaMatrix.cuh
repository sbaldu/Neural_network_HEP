
#ifndef cudaMatrix_h
#define cudaMatrix_h

#pragma once

#include <iostream>

using Bytes = std::size_t;

template <typename T>
struct cudaMatrix {
private:
  int m_nrows;
  int m_ncols;
  T* m_data;

public:
  cudaMatrix(int n_rows, int n_cols) : m_nrows{n_rows}, m_ncols{n_cols} {
    const Bytes bytes{m_nrows * m_ncols * sizeof(T)};
    cudaMalloc(&m_data, bytes);
  }
  cudaMatrix(int n_rows, int n_cols, T* data) : m_data{data}, m_nrows{n_rows}, m_ncols{n_cols} {
    const Bytes bytes{m_nrows * m_ncols * sizeof(T)};
    cudaMalloc(&m_data, bytes);
    cudaMemcpy(&(this->m_data), data, bytes, cudaMemcpyHostToDevice);
  }

  ~cudaMatrix() { cudaFree(m_data); }

  T* data() { return m_data; }
  const T* data() const { return m_data; }

  __device__ int rows() const { return m_nrows; }
  __device__ int cols() const { return m_ncols; }

  __device__ T& operator[](int index) { return m_data[index]; }
  __device__ const T& operator[](int index) const { return m_data[index]; }
};

#endif
