
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "nnhep/DataFormats/Matrix.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test sum of two vector matrices") {
  int len{10};
  std::vector<int> v1(10), v2(10);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), 10);

  Matrix<int> m1(v1);
  Matrix<int> m2(v2);

  Matrix<int> sum_vector{m1 + m2};

  for (int i{}; i < len; ++i) {
    CHECK(sum_vector[i] == 10 + 2 * i);
  }
}

TEST_CASE("Test matrix product between two 2x2 matrices") {
  Matrix<int> m1(2, 2);
  Matrix<int> m2(2, 2);

  int index{};
  for (int i : std::views::iota(1, 5)) {
    m1.set_data(index, i);
    m2.set_data(index, 2 * i);
    ++index;
  }

  Matrix<int> sum_matrix(2, 2);
  sum_matrix = m1 + m2;

  CHECK(sum_matrix.get(0, 0) == 3);
  CHECK(sum_matrix.get(1, 0) == 9);
  CHECK(sum_matrix.get(0, 1) == 6);
  CHECK(sum_matrix.get(1, 1) == 12);
  CHECK(sum_matrix[0] == 3);
  CHECK(sum_matrix[1] == 6);
  CHECK(sum_matrix[2] == 9);
  CHECK(sum_matrix[3] == 12);
}
