
#ifndef Matrix_h
#define Matrix_h

#pragma once

#include <algorithm>
#include <concepts>
#include <iostream>
#include <iterator>
#include <vector>

#include "VectorKernels.h"

template <typename T>
class Matrix {
private:
  int m_nrows;
  int m_ncols;
  int m_size;
  std::vector<T> m_data;

public:
  // Public matrix with data allocated on device
  matrix_t<T> dev_matrix;

  Matrix() = default;
  Matrix(int n_rows, int n_cols);
  template <typename E>
  Matrix(int n_rows, int n_cols, std::vector<E> vec);
  // Create a matrix from a vector
  template <typename E>
  Matrix(std::vector<E> vec);

  ~Matrix();

  // Getters
  inline int nrows() const;
  inline int ncols() const;
  inline int size() const;
  inline const std::vector<T>& data() const;

  // Setters for dimensions
  void set_nrows(int n_rows);
  void set_ncols(int n_cols);
  void set_dim(int n_rows, int n_cols);

  // Setters for data
  inline void set_data(int i, int j, T data);
  inline void set_data(int index, T data);
  inline void set_data(std::vector<T> data_vec);

  inline const T get(int i, int j) const;

  inline Matrix transpose();

  T& operator[](int index);
  const T& operator[](int index) const;

  __host__ void memCpyHtD();
  __host__ void memCpyDtH();

  template <typename E>
  friend Matrix<E> operator+(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename U, std::convertible_to<U> E>
  friend __host__ Matrix<T> operator*(E constant, const Matrix<T>& m);

  template <typename E>
  friend __host__ Matrix<E> operator*(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename U, std::convertible_to<U> E>
  friend __host__ Matrix<U> operator*(const Matrix<U>& m1, const Matrix<E>& m2);

  template <typename E>
  friend __host__ std::vector<E> operator*(const Matrix<E>& matrix, const std::vector<E>& vec);

  template <typename U, std::convertible_to<U> E>
  friend __host__ std::vector<U> operator*(const Matrix<U>& matrix, const std::vector<E>& vec);

  __host__ Matrix<T>& operator+=(const Matrix<T>& other);
  template <typename E>
  __host__ Matrix<T>& operator+=(const Matrix<E>& other);

  __host__ Matrix<T>& operator-=(const Matrix<T>& other);
  template <typename E>
  __host__ Matrix<T>& operator-=(const Matrix<E>& other);

  template <typename E>
  __host__ Matrix<T>& operator*=(E constant);

  template <typename E>
  __host__ Matrix<T>& operator/=(E constant);

  template <typename U>
  friend std::ostream& operator<<(std::ostream& out, const Matrix<U>& m);
};

template <typename T>
Matrix<T>::Matrix(int n_rows, int n_cols)
    : m_nrows{n_rows},
      m_ncols{n_cols},
      m_size{n_rows * n_cols},
      m_data(n_rows * n_cols),
      dev_matrix{matrix_t<T>(n_rows, n_cols)} {}

template <typename T>
template <typename E>
Matrix<T>::Matrix(int n_rows, int n_cols, std::vector<E> vec) : Matrix{n_rows, n_cols} {
  m_data = std::move(vec);

  cudaMemcpy(dev_matrix.data, m_data.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice);
}

template <typename T>
template <typename E>
Matrix<T>::Matrix(std::vector<E> vec) : Matrix{static_cast<int>(vec.size()), 1} {
  m_data = std::move(vec);

  cudaMemcpy(dev_matrix.data, m_data.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice);
}

template <typename T>
Matrix<T>::~Matrix<T>() {
  cudaFree(dev_matrix.data);
}

template <typename T>
int Matrix<T>::nrows() const {
  return m_nrows;
}

template <typename T>
int Matrix<T>::ncols() const {
  return m_ncols;
}

template <typename T>
int Matrix<T>::size() const {
  return m_size;
}

template <typename T>
const std::vector<T>& Matrix<T>::data() const {
  return m_data;
}

template <typename T>
void Matrix<T>::set_nrows(int n_rows) {
  m_nrows = n_rows;
}

template <typename T>
void Matrix<T>::set_ncols(int n_cols) {
  m_ncols = n_cols;
}

template <typename T>
void Matrix<T>::set_dim(int n_rows, int n_cols) {
  m_nrows = n_rows;
  m_ncols = n_cols;
  m_size = n_rows * n_cols;
}

template <typename T>
void Matrix<T>::set_data(int i, int j, T data) {
  int index{j + m_ncols * i};
  try {
    if (index >= m_ncols * m_nrows) {
      throw(index);
    }
    m_data[index] = data;
  } catch (...) {
    std::cout << "The index " << index << " is larger that the size of the matrix\n";
  }
}

template <typename T>
void Matrix<T>::set_data(int index, T data) {
  m_data[index] = data;
  try {
    if (index >= m_ncols * m_nrows) {
      throw(index);
    }
    m_data[index] = data;
  } catch (...) {
    std::cout << "The index " << index << " is larger that the size of the matrix\n";
  }
}

template <typename T>
void Matrix<T>::set_data(std::vector<T> data_vec) {
  m_data = std::move(data_vec);

  // Copy data to the matrix_t pointer on device memory
  cudaMemcpy(dev_matrix.data, m_data.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice);
}

template <typename T>
const T Matrix<T>::get(int i, int j) const {
  return m_data[j + m_ncols * i];
}

template <typename T>
Matrix<T> Matrix<T>::transpose() {
  Matrix<T> matrix(this->m_ncols, this->m_nrows);

  for (int i{}; i < this->m_nrows; ++i) {
    for (int j{}; j < this->m_ncols; ++j) {
      matrix.set_data(j, i, this->get(i, j));
    }
  }

  return matrix;
}

template <typename T>
T& Matrix<T>::operator[](int index) {
  return m_data[index];
}

template <typename T>
const T& Matrix<T>::operator[](int index) const {
  return m_data[index];
}

template <typename T>
__host__ void Matrix<T>::memCpyHtD() {
  cudaMemcpy(this->dev_matrix.data, this->m_data.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice);
}

template <typename T>
__host__ void Matrix<T>::memCpyDtH() {
  cudaMemcpy(this->m_data.data(), this->dev_matrix.data, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
}

template <typename T>
__host__ Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> result(m1.m_nrows, m1.m_ncols);

  const int N{result.m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * N}; */

  // Launch kernel
  vec_add<<<grid_size, block_size>>>(m1.dev_matrix.data, m1.dev_matrix.data, result.dev_matrix.data, N);
  /* cudaMemcpy(result.m_data.data(), result.dev_matrix.data, size, cudaMemcpyDeviceToHost); */

  return result;
}

template <typename T, std::convertible_to<T> E>
Matrix<T> operator*(E constant, const Matrix<T>& m) {
  Matrix<T> result(m);
  constant = static_cast<T>(constant);

  const int N{result.size()};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * N}; */

  // Launch kernel
  /* cudaMemcpy(d_res, result.data().data(), size, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_m, m.data().data(), size, cudaMemcpyHostToDevice); */
  vec_add<<<grid_size, block_size>>>(result.dev_matrix.data, m.dev_matrix.data, N);
  /* cudaMemcpy(const_cast<T*>(result.data().data()), d_res, size, cudaMemcpyDeviceToHost); */

  return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
  const int N{m1.m_nrows};
  const int K{m1.m_ncols};
  const int M{m2.m_ncols};

  try {
    if (m1.m_ncols != m2.m_nrows) {
      throw(0);
    }
  } catch (int num) {
    std::cout << "The two matrices can't be multiplied because their dimensions are not compatible. \n";
  }

  Matrix<T> result(N, M);

  /* const size_t size_a{N * K * sizeof(T)}; */
  /* const size_t size_b{K * M * sizeof(T)}; */
  /* const size_t size_c{N * M * sizeof(T)}; */

  auto block_size{std::min({N, K, M})};
  if (block_size > 32) {
    block_size = 32;
  }

  const int grid_x{(int)std::ceil(M / (float)(block_size))};
  const int grid_y{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_x, grid_y);

  /* cudaMemcpy(m_a.data, m1.data().data(), size_a, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(m_b.data, m2.data().data(), size_b, cudaMemcpyHostToDevice); */
  const size_t shared_size{2 * block_size * block_size * sizeof(T)};
  matrix_multiply<<<grid, block, shared_size>>>(
      m1.dev_matrix, m2.dev_matrix, result.dev_matrix, block_size);
  /* cudaMemcpy(const_cast<T*>(result.data().data()), m_c.data, size_c, cudaMemcpyDeviceToHost); */

  return result;
}

template <typename T, std::convertible_to<T> E>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<E>& m2) {
  const int N{m1.m_nrows};
  const int K{m1.m_ncols};
  const int M{m2.m_ncols};

  try {
    if (m1.m_ncols != m2.m_nrows) {
      throw(0);
    }
  } catch (int num) {
    std::cout << "The two matrices can't be multiplied because their dimensions are not compatible. \n";
  }

  Matrix<T> result(N, M);

  /* const size_t size_a{N * K * sizeof(T)}; */
  /* const size_t size_b{K * M * sizeof(E)}; */
  /* const size_t size_c{N * M * sizeof(T)}; */

  int block_size{std::min({N, K, M})};
  if (block_size > 32) {
    block_size = 32;
  }

  const int grid_x{(int)std::ceil(M / (float)(block_size))};
  const int grid_y{(int)std::ceil(N / (float)(block_size))};
  dim3 block(block_size, block_size);
  dim3 grid(grid_x, grid_y);

  /* cudaMemcpy(m_a.data, m1.data().data(), size_a, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(m_b.data, m2.data().data(), size_b, cudaMemcpyHostToDevice); */
  const size_t shared_size{2 * block_size * block_size * sizeof(T)};
  matrix_multiply<<<grid, block, shared_size>>>(
      m1.dev_matrix, m2.dev_matrix, result.dev_matrix, block_size);
  /* cudaMemcpy(const_cast<T*>(result.data().data()), m_c.data, size_c, cudaMemcpyDeviceToHost); */

  return result;
}

template <typename T>
std::vector<T> operator*(const Matrix<T>& matrix, const std::vector<T>& vec) {
  const int N{matrix.m_ncols};
  Matrix<T> matrix_vec(N, 1, vec);
  Matrix<T> result_matrix{matrix * matrix_vec};

  return result_matrix.data();
}

template <typename T, std::convertible_to<T> E>
std::vector<T> operator*(const Matrix<T>& matrix, const std::vector<E>& vec) {
  const int N{matrix.m_size};
  Matrix<E> matrix_vec(N, 1, vec);
  Matrix<T> result_matrix{matrix * matrix_vec};

  return result_matrix.data();
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_other, other.m_data.data(), size, cudaMemcpyHostToDevice); */
  vec_add<<<grid_size, block_size>>>(this->dev_matrix.data, other.dev_matrix.data, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator+=(const Matrix<E>& other) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size_src{sizeof(T) * m_size}; */
  /* const size_t size_other{sizeof(E) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size_src, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_other, other.data().data(), size_other, cudaMemcpyHostToDevice); */
  vec_add<<<grid_size, block_size>>>(this->dev_matrix.data, other.dev_matrix.data, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size_src, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_other, other.m_data.data(), size, cudaMemcpyHostToDevice); */
  vec_sub<<<grid_size, block_size>>>(this->dev_matrix.data, other.dev_matrix.data, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator-=(const Matrix<E>& other) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size_src{sizeof(T) * m_size}; */
  /* const size_t size_other{sizeof(E) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size_src, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(d_other, other.data().data(), size_other, cudaMemcpyHostToDevice); */
  vec_sub<<<grid_size, block_size>>>(this->dev_matrix.data, other.dev_matrix.data, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size_src, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator*=(E constant) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size, cudaMemcpyHostToDevice); */
  vec_multiply<<<grid_size, block_size>>>(this->dev_matrix.data, constant, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename T>
template <typename E>
__host__ Matrix<T>& Matrix<T>::operator/=(E constant) {
  const int N{m_size};
  const int block_size{256};
  const int grid_size{(int)std::ceil(N / (float)(block_size))};
  /* const size_t size{sizeof(T) * m_size}; */

  // Launch kernel
  /* cudaMemcpy(d_vec, this->m_data.data(), size, cudaMemcpyHostToDevice); */
  vec_divide<<<grid_size, block_size>>>(this->dev_matrix.data, constant, m_size);
  /* cudaMemcpy(this->m_data.data(), d_vec, size, cudaMemcpyDeviceToHost); */

  return *this;
}

template <typename U>
std::ostream& operator<<(std::ostream& out, const Matrix<U>& m) {
  out << m.m_data[0];
  std::for_each(m.m_data.begin() + 1, m.m_data.end(), [&out](U x) {
    out << ',';
    out << x;
  });

  return out;
}

#endif
