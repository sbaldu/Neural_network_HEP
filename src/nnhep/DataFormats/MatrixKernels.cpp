
#ifndef matrixkernels_hpp
#define matrixkernels_hpp

#include "Matrix.hpp"

namespace nnhep {
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
};
#endif
