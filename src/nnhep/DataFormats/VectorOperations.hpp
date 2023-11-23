/// @file VectorOperations.hpp
/// @brief File containing the definition of the vector operations

#ifndef vec_operations_h
#define vec_operations_h

#pragma once

#include <algorithm>
#include <concepts>
#include <ostream>
#include <vector>

#include "Matrix.hpp"

/// @brief Add two vectors
/// @tparam T Type of the vector
/// @param v1 First vector
/// @param v2 Second vector
/// @return The sum of the two vectors
template <typename T>
std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2) {
  std::vector<T> result(v1.size());

  for (size_t i{}; i < v1.size(); ++i) {
    result[i] = v1[i] + v2[i];
  }

  return result;
}

/// @brief Subtract two vectors
/// @tparam T Type of the vector
/// @param v1 First vector
/// @param v2 Second vector
/// @return The difference of the two vectors
template <typename T>
std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
  std::vector<T> result(v1.size());

  for (int i{}; i < v1.size(); ++i) {
    result[i] = v1[i] - v2[i];
  }

  return result;
}

/// @brief Multiply a vector by a constant
/// @tparam T Type of the vector
/// @tparam E Type of the constant
/// @param constant Constant to multiply the vector by
/// @param vec Vector to multiply
/// @return The vector multiplied by the constant
template <typename T, std::convertible_to<T> E>
std::vector<T> operator*(E constant, std::vector<T> vec) {
  std::vector<T> result{vec};

  std::transform(result.cbegin(), result.cend(), result.begin(), [constant](auto x) {
    x *= constant;
    return x;
  });

  return result;
}

/// @brief In-place subtraction of a vector and a matrix
/// @tparam T Type of the vector and matrix
/// @param vec Vector to subtract from
/// @param m Matrix to subtract
/// @return The vector subtracted by the matrix
template <typename T>
void operator-=(std::vector<T>& vec, const Matrix<T>& m) {
  for (size_t i{}; i < vec.size(); ++i) {
    vec[i] -= m.data()[i];
  }
}

/// @brief Print a vector
/// @tparam T Type of the vector
/// @param out Output stream
/// @param vec Vector to print
/// @return The output stream
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << *vec.begin();
  std::for_each(vec.begin() + 1, vec.end(), [&out](T x) {
    out << ',';
    out << x;
  });

  return out;
}

#endif
