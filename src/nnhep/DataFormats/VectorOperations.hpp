
#pragma once

#include <algorithm>
#include <concepts>
#include <ostream>
#include <vector>

#include "Matrix.hpp"

namespace nnhep {

  template <typename T>
  std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> result(v1.size());

    for (size_t i{}; i < v1.size(); ++i) {
      result[i] = v1[i] + v2[i];
    }

    return result;
  }

  template <typename T>
  std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> result(v1.size());

    for (int i{}; i < v1.size(); ++i) {
      result[i] = v1[i] - v2[i];
    }

    return result;
  }

  template <typename T, std::convertible_to<T> E>
  std::vector<T> operator*(E constant, std::vector<T> vec) {
    std::vector<T> result{vec};

    std::transform(result.cbegin(), result.cend(), result.begin(), [constant](auto x) {
      x *= constant;
      return x;
    });

    return result;
  }

  template <typename T>
  void operator-=(std::vector<T>& vec, const Matrix<T>& m) {
    for (size_t i{}; i < vec.size(); ++i) {
      vec[i] -= m.data()[i];
    }
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    out << *vec.begin();
    std::for_each(vec.begin() + 1, vec.end(), [&out](T x) {
      out << ',';
      out << x;
    });

    return out;
  }

};  // namespace nnhep
