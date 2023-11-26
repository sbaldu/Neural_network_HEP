
#ifndef cudaMatrix_h
#define cudaMatrix_h

#pragma once

#include <iostream>

template <typename T>
struct matrix_t {
  T* data;
  int rows;
  int cols;

  matrix_t(int n_rows, int n_cols) : rows{n_rows}, cols{n_cols} {
	cudaMalloc(&data, rows * cols * sizeof(T));
  }
  /* matrix_t(int n_rows, int n_cols, T* data) : data{data}, rows{n_rows}, cols{n_cols} { */
	/* cudaMalloc(&data, rows * cols * sizeof(T)); */
  /* } */

  ~matrix_t() { 
	cudaFree(data);
  }

  __device__ T& operator[](int index) { return data[index]; }
  __device__ const T& operator[](int index) const { return data[index]; }
};


#endif
