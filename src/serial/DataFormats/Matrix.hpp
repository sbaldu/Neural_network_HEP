
#ifndef Matrix_h
#define Matrix_h

#pragma once

#include <iostream>
#include <vector>

template <typename T>
class Matrix {
private:
  int m_nrows;
  int m_ncols;
  std::vector<T> m_data;

public:
  Matrix() = default;
  Matrix(int n_rows, int n_cols);
  Matrix(int n_rows, int n_cols, std::vector<T> data);
  // Create a matrix from a vector
  Matrix(const std::vector<T>& vec);

  // Getters
  const int nrows() const;
  const int ncols() const;
  const std::vector<T>& data() const;

  // Setters for dimensions
  void set_nrows(int n_rows);
  void set_ncols(int n_cols);
  void set_dim(int n_rows, int n_cols);

  // Setters for data
  void set_data(int i, int j, T data);
  void set_data(int index, T data);

  const T get(int i, int j) const;

  Matrix transpose();

  T& operator[](int index);

  template <typename E>
  friend Matrix<E> operator+(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename E>
  friend Matrix<E> operator*(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename E>
  friend std::vector<E> operator*(const Matrix<E>& matrix, const std::vector<E>& vec);
};

template <typename T>
Matrix<T>::Matrix(int n_rows, int n_cols) : m_nrows{n_rows}, m_ncols{n_cols}, m_data(n_rows * n_cols) {}

template <typename T>
Matrix<T>::Matrix(int n_rows, int n_cols, std::vector<T> data)
    : m_nrows{n_rows}, m_ncols{n_cols}, m_data{std::move(data)} {}

template <typename T>
Matrix<T>::Matrix(const std::vector<T>& vec) : m_nrows{static_cast<int>(vec.size())}, m_ncols{1}, m_data{vec} {}

template <typename T>
const int Matrix<T>::nrows() const {
  return m_nrows;
}

template <typename T>
const int Matrix<T>::ncols() const {
  return m_ncols;
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
}

template <typename T>
void Matrix<T>::set_data(int i, int j, T data) {
  int index{j + m_ncols * i};
  try {
    if (index < m_ncols * m_nrows) {
      m_data[index] = data;
    } else {
      throw(index);
    }
  } catch (...) {
    std::cout << "The index " << index << " is larger that the size of the matrix\n";
  }
}

template <typename T>
void Matrix<T>::set_data(int index, T data) {
  try {
    if (index < m_ncols * m_nrows) {
      m_data[index] = data;
    } else {
      throw(index);
    }
  } catch (...) {
    std::cout << "The index " << index << " is larger that the size of the matrix\n";
  }
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
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> result(m1.m_nrows, m1.m_ncols);

  for (int index{}; index < m1.m_nrows * m1.m_ncols; ++index) {
	result.set_data(index, m1.m_data[index] + m2.m_data[index]);
  }

  return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> result(m1.m_nrows, m2.m_ncols);

  for (int i{}; i < m1.m_nrows; ++i) {
    for (int j{}; j < m2.m_ncols; ++j) {
      T sum{};
      for (int k{}; k < m1.m_nrows; ++k) {
        sum += m1.get(i, k) * m2.get(k, j);
      }
      result.set_data(i, j, sum);
    }
  }

  return result;
}

template <typename T>
std::vector<T> operator*(const Matrix<T>& matrix, const std::vector<T>& vec) {
  std::vector<T> result_vec(matrix.m_nrows);

  for (int i{}; i < matrix.m_nrows; ++i) {
    T sum{};
    for (int j{}; j < matrix.m_ncols; ++j) {
      sum += matrix.m_data[j + matrix.m_ncols * i] * vec[j];
    }
    result_vec[i] = sum;
  }

  return result_vec;
}

#endif
