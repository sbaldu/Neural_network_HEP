
#ifndef Matrix_h
#define Matrix_h

#pragma once

#include <algorithm>
#include <concepts>
#include <iostream>
#include <iterator>
#include <vector>

template <typename T>
class Matrix {
private:
  int m_nrows;
  int m_ncols;
  int m_size;
  std::vector<T> m_data;

public:
  Matrix() = default;
  Matrix(int n_rows, int n_cols);
  template <typename E>
  Matrix(int n_rows, int n_cols, std::vector<E> vec);
  // Create a matrix from a vector
  template <typename E>
  Matrix(std::vector<E> vec);

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

  template <typename E>
  friend Matrix<E> operator+(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename U, std::convertible_to<U> E>
  friend Matrix<T> operator*(E constant, const Matrix<T>& m);

  template <typename E>
  friend Matrix<E> operator*(const Matrix<E>& m1, const Matrix<E>& m2);

  template <typename U, std::convertible_to<U> E>
  friend Matrix<U> operator*(const Matrix<U>& m1, const Matrix<E>& m2);

  template <typename E>
  friend std::vector<E> operator*(const Matrix<E>& matrix, const std::vector<E>& vec);

  template <typename U, std::convertible_to<U> E>
  friend std::vector<U> operator*(const Matrix<U>& matrix, const std::vector<E>& vec);

  Matrix<T>& operator+=(const Matrix<T>& other);
  template <typename E>
  Matrix<T>& operator+=(const Matrix<E>& other);

  Matrix<T>& operator-=(const Matrix<T>& other);
  template <typename E>
  Matrix<T>& operator-=(const Matrix<E>& other);

  template <typename E>
  Matrix<T>& operator*=(E constant);

  template <typename E>
  Matrix<T>& operator/=(E constant);

  template <typename U>
  friend std::ostream& operator<<(std::ostream& out, const Matrix<U>& m);
};

template <typename T>
Matrix<T>::Matrix(int n_rows, int n_cols)
    : m_nrows{n_rows}, m_ncols{n_cols}, m_size{n_rows * n_cols}, m_data(n_rows * n_cols) {}

template <typename T>
template <typename E>
Matrix<T>::Matrix(int n_rows, int n_cols, std::vector<E> vec) : Matrix{n_rows, n_cols} {
  m_data = std::move(vec);
}

template <typename T>
template <typename E>
Matrix<T>::Matrix(std::vector<E> vec) : Matrix{static_cast<int>(vec.size()), 1} {
  m_data = std::move(vec);
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
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> result(m1.m_nrows, m1.m_ncols);

  for (int index{}; index < m1.m_nrows * m1.m_ncols; ++index) {
    result.set_data(index, m1.m_data[index] + m2.m_data[index]);
  }

  return result;
}

template <typename T, std::convertible_to<T> E>
Matrix<T> operator*(E constant, const Matrix<T>& m) {
  constant = static_cast<T>(constant);

  Matrix<T> result(m);
  for (int i{}; i < result.size(); ++i) {
    result[i] *= constant;
  }

  return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> result(m1.m_nrows, m2.m_ncols);

  try {
    if (m1.m_ncols != m2.m_nrows) {
      throw(0);
    }
  } catch (int num) {
    std::cout << "The two matrices can't be multiplied because their dimensions are not compatible. \n";
  }

  for (int i{}; i < m1.m_nrows; ++i) {
    for (int j{}; j < m2.m_ncols; ++j) {
      T sum{};
      for (int k{}; k < m1.m_ncols; ++k) {
        sum += m1.get(i, k) * m2.get(k, j);
      }
      result.set_data(i, j, sum);
    }
  }

  return result;
}

template <typename T, std::convertible_to<T> E>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<E>& m2) {
  Matrix<T> result(m1.m_nrows, m2.m_ncols);

  try {
    if (m1.m_ncols != m2.m_nrows) {
      throw(0);
    }
  } catch (int num) {
    std::cout << "The two matrices can't be multiplied because their dimensions are not compatible. \n";
  }

  for (int i{}; i < m1.m_nrows; ++i) {
    for (int j{}; j < m2.m_ncols; ++j) {
      T sum{};
      for (int k{}; k < m1.m_ncols; ++k) {
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

template <typename T, std::convertible_to<T> E>
std::vector<T> operator*(const Matrix<T>& matrix, const std::vector<E>& vec) {
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

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
  for (int i{}; i < this->m_size; ++i) {
    this->m_data[i] += other.m_data[i];
  }

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator+=(const Matrix<E>& other) {
  for (int i{}; this->m_size; ++i) {
    this->m_data[i] += other.data()[i];
  }

  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
  for (int i{}; i < this->m_size; ++i) {
    this->m_data[i] -= other.m_data[i];
  }

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator-=(const Matrix<E>& other) {
  for (int i{}; i < this->m_size; ++i) {
    this->m_data[i] -= other.data()[i];
  }

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator*=(E constant) {
  std::transform(this->m_data.cbegin(), this->m_data.cend(), this->m_data.begin(), [constant](auto x) {
	x *= constant;
    return x;
  });

  return *this;
}

template <typename T>
template <typename E>
Matrix<T>& Matrix<T>::operator/=(E constant) {
  std::transform(this->m_data.cbegin(), this->m_data.cend(), this->m_data.begin(), [constant](auto x) {
	x /= constant;
    return x;
  });

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
