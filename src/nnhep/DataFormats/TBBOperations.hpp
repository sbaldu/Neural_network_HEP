
#ifndef tbb_operations_hpp
#define tbb_operations_hpp

#pragma once

#include <vector>
#include "Matrix.hpp"

#include <oneapi/tbb.h>

namespace nnhep {

  template <typename T>
  std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> result(v1.size());

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, v1.size()),
                              [&](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i{r.begin()}; i != r.end(); ++i) {
                                  result[i] = v1[i] + v2[i];
                                }
                              });
    return result;
  }

  template <typename T>
  std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> result(v1.size());

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, v1.size()),
                              [&](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i{r.begin()}; i != r.end(); ++i) {
                                  result[i] = v1[i] - v2[i];
                                }
                              });

    return result;
  }

  template <typename T, std::convertible_to<T> E>
  std::vector<T> operator*(E constant, std::vector<T> vec) {
    std::vector<T> result{vec};

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, vec.size()),
                              [&vec, constant](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i{r.begin()}; i != r.end(); ++i) {
                                  result[i] = vec[i] * constant;
                                }
                              });

    return result;
  }

  template <typename T>
  void operator-=(std::vector<T>& vec, const Matrix<T>& m) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, vec.size()),
                              [&](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i{r.begin()}; i != r.end(); ++i) {
                                  vec[i] -= m[i];
                                }
                              });
  }
};  // namespace nnhep

#endif
