
#ifndef vec_operations_h
#define vec_operations_h

#pragma once

#include <algorithm>
#include <concepts>
#include <ostream>
#include <vector>

#include "Matrix.hpp"

// Overload sum operator for vectors
template <typename T>
std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2) {
  std::vector<T> result(v1.size());

  for (int i{}; i < v1.size(); ++i) {
    result[i] = v1[i] + v2[i];
  }

  return result;
}

// Overload subtraction
template <typename T>
std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
  std::vector<T> result(v1.size());

  for (int i{}; i < v1.size(); ++i) {
    result[i] = v1[i] - v2[i];
  }

  return result;
}

// Overload product of vector with constant
template <typename T, std::convertible_to<T> E>
std::vector<T> operator*(E constant, std::vector<T> vec) {
  std::vector<T> result{vec};

  for (auto& element : result) {
    element *= constant;
  }

  return result;
}

// Overload of operator-= with matrices
template <typename T>
void operator-=(std::vector<T>& vec, const Matrix<T>& m) {
  for (int i{}; i < vec.size(); ++i) {
    vec[i] -= m.data()[i];
  }
}

// Ostream operator for vectors
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
