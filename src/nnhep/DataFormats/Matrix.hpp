/// @file Matrix.hpp
/// @brief This file contains the declaration of the Matrix class.
/// @author Simone Balducci
///
///
/// @details The Matrix class is a class that represents a matrix of any size.

#ifndef Matrix_h
#define Matrix_h

#pragma once

#include <algorithm>
#include <concepts>
#include <iostream>
#include <iterator>
#include <vector>

namespace nnhep {

  /// @brief Matrix class
  /// @tparam T The type of the elements of the matrix
  template <typename T>
  class Matrix {
  private:
    int m_nrows;
    int m_ncols;
    int m_size;
    std::vector<T> m_data;

  public:
    /// @brief Default constructor
    Matrix() = default;
    /// @brief Constructor
    /// @param n_rows The number of rows of the matrix
    /// @param n_cols The number of columns of the matrix
    Matrix(int n_rows, int n_cols);
    /// @brief Constructor
    /// @tparam E The type of the elements of the vector
    /// @param n_rows The number of rows of the matrix
    /// @param n_cols The number of columns of the matrix
    template <typename E>
    Matrix(int n_rows, int n_cols, std::vector<E> vec);
    /// @brief Constructor
    /// @tparam E The type of the elements of the vector
    /// @param vec The vector to initialize the matrix with
    ///
    /// @details The matrix is initialized with the vector as a column vector
    template <typename E>
    explicit Matrix(std::vector<E> vec);

    /// @brief Get the number of rows of the matrix
    /// @return The number of rows of the matrix
    inline int nrows() const;
    /// @brief Get the number of columns of the matrix
    /// @return The number of columns of the matrix
    inline int ncols() const;
    /// @brief Get the size of the matrix
    /// @return The size of the matrix
    inline int size() const;
    /// @brief Get the data of the matrix
    /// @return The data of the matrix
    inline const std::vector<T>& data() const;

    /// @brief Set the number of rows of the matrix
    /// @param n_rows The number of rows of the matrix
    void set_nrows(int n_rows);
    /// @brief Set the number of columns of the matrix
    /// @param n_cols The number of columns of the matrix
    void set_ncols(int n_cols);
    /// @brief Set the number of rows and columns of the matrix
    /// @param n_rows The number of rows of the matrix
    void set_dim(int n_rows, int n_cols);
    /// @brief Set the data of the matrix
    /// @param i The row index of the element to set
    /// @param j The column index of the element to set
    /// @param data The data to set the element to
    inline void set_data(int i, int j, T data);
    /// @brief Set the data of the matrix
    /// @param index The index of the element to set
    /// @param data The data to set the element to
    inline void set_data(int index, T data);
    /// @brief Set the data of the matrix
    /// @param data_vec The data to set the matrix to
    inline void set_data(std::vector<T> data_vec);

    /// @brief Get an element of the matrix
    /// @param i The row index of the element to get
    /// @param j The column index of the element to get
    /// @return The element of the matrix
	///
	/// @note returns by reference
    inline T& operator()(int i, int j);
    /// @brief Get an element of the matrix
    /// @param i The row index of the element to get
    /// @param j The column index of the element to get
    /// @return The element of the matrix
	///
	/// @note returns by const reference
    inline const T& operator()(int i, int j) const;

    ///
    inline Matrix transpose();

    /// @brief Get an element of the matrix
    /// @param index The index of the element to get
    /// @return The element of the matrix
    T& operator[](int index);
    /// @brief Get an element of the matrix
    /// @param index The index of the element to get
    /// @return The element of the matrix
    const T& operator[](int index) const;

    /// @brief Add two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param m1 The first matrix
    /// @param m2 The second matrix
    /// @return The sum of the two matrices
    template <typename E>
    friend Matrix<E> operator+(const Matrix<E>& m1, const Matrix<E>& m2);
    /// @brief Multiply a matrix by a constant
    /// @tparam E The type of the elements of the matrix
    /// @param constant The constant to multiply the matrix by
    /// @param m The matrix to multiply
    /// @return The product of the matrix and the constant
    template <typename U, std::convertible_to<U> E>
    friend Matrix<T> operator*(E constant, const Matrix<T>& m);
    /// @brief Multiply two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param m1 The first matrix
    /// @param m2 The second matrix
    /// @return The product of the two matrices
    template <typename E>
    friend Matrix<E> operator*(const Matrix<E>& m1, const Matrix<E>& m2);
    /// @brief Multiply two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param m1 The first matrix
    /// @param m2 The second matrix
    /// @return The product of the two matrices
    template <typename U, std::convertible_to<U> E>
    friend Matrix<U> operator*(const Matrix<U>& m1, const Matrix<E>& m2);
    /// @brief Multiply a matrix by a vector
    /// @tparam E The type of the elements of the matrix
    /// @param matrix The matrix
    /// @param vec The vector
    /// @return The product of the matrix and the vector
    template <typename E>
    friend std::vector<E> operator*(const Matrix<E>& matrix, const std::vector<E>& vec);
    /// @brief Multiply a matrix by a vector
    /// @tparam E The type of the elements of the matrix
    /// @param matrix The matrix
    /// @param vec The vector
    /// @return The product of the matrix and the vector
    template <typename U, std::convertible_to<U> E>
    friend std::vector<U> operator*(const Matrix<U>& matrix, const std::vector<E>& vec);
    /// @brief In-place addition of two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param other The matrix to add to the current matrix
    /// @return The current matrix
    Matrix<T>& operator+=(const Matrix<T>& other);
    /// @brief In-place addition of two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param other The matrix to add to the current matrix
    /// @return The current matrix
    ///
    /// @details This function is used to add two matrices of different types
    template <typename E>
    Matrix<T>& operator+=(const Matrix<E>& other);
    /// @brief In-place subtraction of two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param other The matrix to subtract from the current matrix
    /// @return The current matrix
    Matrix<T>& operator-=(const Matrix<T>& other);
    /// @brief In-place subtraction of two matrices
    /// @tparam E The type of the elements of the matrix
    /// @param other The matrix to subtract from the current matrix
    /// @return The current matrix
    ///
    /// @details This function is used to subtract two matrices of different types
    template <typename E>
    Matrix<T>& operator-=(const Matrix<E>& other);
    /// @brief In-place multiplication of a matrix by a constant
    /// @tparam E The type of the elements of the matrix
    /// @param constant The constant to multiply the matrix by
    /// @return The current matrix
    template <typename E>
    Matrix<T>& operator*=(E constant);
    /// @brief In-place division of a matrix by a constant
    /// @tparam E The type of the elements of the matrix
    /// @param constant The constant to divide the matrix by
    /// @return The current matrix
    template <typename E>
    Matrix<T>& operator/=(E constant);
    /// @brief Print the matrix
    /// @tparam E The type of the elements of the matrix
    /// @param out The stream to print to
    /// @param m The matrix to print
    /// @return The stream
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
  T& Matrix<T>::operator()(int i, int j) {
    return m_data[j + m_ncols * i];
  }

  template <typename T>
  const T& Matrix<T>::operator()(int i, int j) const {
    return m_data[j + m_ncols * i];
  }

  template <typename T>
  Matrix<T> Matrix<T>::transpose() {
    Matrix<T> matrix(this->m_ncols, this->m_nrows);

    for (int i{}; i < this->m_nrows; ++i) {
      for (int j{}; j < this->m_ncols; ++j) {
        matrix.set_data(j, i, (*this)(i, j));
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
          sum += m1(i, k) * m2(k, j);
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
          sum += m1(i, k) * m2(k, j);
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

};  // namespace nnhep

#endif
