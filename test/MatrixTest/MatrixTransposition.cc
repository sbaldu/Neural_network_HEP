
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "DataFormats/Matrix.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test the transposition of a vector matrix") {
  std::vector<int> vec(10, 1);

  Matrix<int> m1(vec);

  // Check the dimensions before transposition
  // The vector should be a column vector
  CHECK(m1.ncols() == 1);
  CHECK(m1.nrows() == 10);

  Matrix<int> m2 = m1.transpose();

  // Check the dimensions after transposition
  // The vector should be a row vector
  CHECK(m2.ncols() == 10);
  CHECK(m2.nrows() == 1);
}

TEST_CASE("Test the transposition of a square matrix") {
  Matrix<int> m1(4, 4);

  int index{};
  for (int i : std::views::iota(1, 17)) {
    m1.set_data(index, i);
  }

  Matrix<int> m2 = m1.transpose();

  // Check the new dimensions of the matrix and its content
  CHECK(m2.nrows() == 4);
  CHECK(m2.ncols() == 4);
  for (int i{}; i < m2.nrows(); ++i) {
    for (int j{}; j < m2.ncols(); ++j) {
      CHECK(m2.get(i, j) == m1.get(j, i));
    }
  }
}

TEST_CASE("Test the transposition of a rectangular matrix") {
  Matrix<int> m1(2, 4);

  int index{};
  for (int i : std::views::iota(1, 9)) {
    m1.set_data(index, i);
  }

  Matrix<int> m2 = m1.transpose();

  // Check the new dimensions of the matrix and its content
  CHECK(m2.nrows() == 4);
  CHECK(m2.ncols() == 2);
  for (int i{}; i < m2.nrows(); ++i) {
    for (int j{}; j < m2.ncols(); ++j) {
      CHECK(m2.get(i, j) == m1.get(j, i));
    }
  }
}
