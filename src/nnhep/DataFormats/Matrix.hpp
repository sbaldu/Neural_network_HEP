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

#include "../../HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "../../HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "../../HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include "alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename T>
  class Matrix {
  private:
    int m_nrows;
    int m_ncols;
    int m_size;
    std::vector<T> m_data;
    cms::alpakatools::device_buffer<T[]> m_dev;

  public:
    Matrix() = delete;
    Matrix(Queue queue, int n_rows, int n_cols);
    template <typename E>
    Matrix(Queue queue, int n_rows, int n_cols, std::vector<E> vec);
    template <typename E>
    Matrix(Queue queue, std::vector<E> vec);

	// host and device buffers
	std::vector<T>& hostBuffer() { return m_data; }
	const std::vector<T>& hostBuffer() const { return m_data; }

	// host and device views
	T* hostView() { return m_data.data(); }
	const T* hostView() const { return m_data.data(); }
	T* deviceView() { return m_dev.data(); }
	const T* deviceView() const { return m_dev.data(); }

	void updateHost(Queue queue) {
	  alpaka::memcpy(queue, m_data, m_dev);
	  alpaka::wait(queue);
	}

    inline int nrows() const;
    inline int ncols() const;
    inline std::size_t size() const;

    inline void set_data(int i, int j, T data);
    inline void set_data(int index, T data);
    inline void set_data(std::vector<T> data_vec);

    inline T& operator()(int i, int j);
    inline const T& operator()(int i, int j) const;

    inline Matrix transpose();

    T& operator[](int index);
    const T& operator[](int index) const;

    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, const Matrix<U>& m);
  };

  template <typename T>
  Matrix<T>::Matrix(int n_rows, int n_cols)
      : m_nrows{n_rows},
        m_ncols{n_cols},
        m_size{n_rows * n_cols},
        m_data(n_rows * n_cols) {}

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
      std::cout << "The two matrices can't be multiplied because their dimensions are "
                   "not compatible. \n";
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
      std::cout << "The two matrices can't be multiplied because their dimensions are "
                   "not compatible. \n";
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
    std::transform(this->m_data.cbegin(),
                   this->m_data.cend(),
                   this->m_data.begin(),
                   [constant](auto x) {
                     x *= constant;
                     return x;
                   });

    return *this;
  }

  template <typename T>
  template <typename E>
  Matrix<T>& Matrix<T>::operator/=(E constant) {
    std::transform(this->m_data.cbegin(),
                   this->m_data.cend(),
                   this->m_data.begin(),
                   [constant](auto x) {
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

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
