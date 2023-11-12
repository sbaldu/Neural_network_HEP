
#ifndef random_matrix_hpp
#define random_matrix_hpp

#include <random>

#include "Matrix.hpp"

template <typename T>
void random_matrix(Matrix<T>& mat) {
  for (int i{}; i < mat.size(); ++i) {
    mat[i] = std::rand();
  }
}

#endif
