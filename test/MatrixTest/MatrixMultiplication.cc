
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "../../src/serial/DataFormats/Matrix.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

TEST_CASE("Test scalar product between two vector matrices") {
  int len{10};
  std::vector<int> v1(10), v2(10);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), 10);

  Matrix<int> m1(v1);
  Matrix<int> m2(v2);

  Matrix<int> product{ m1*m2 };

  CHECK(product.get(0, 0) == 735);
}

TEST_CASE("Test matrix product between two 2x2 matrices") {
  Matrix<int> m1(2, 2);
  Matrix<int> m2(2, 2);
  
  int index{};
  for (int i : std::views::iota(1,5)) {
	m1.set_data(index, i);
	m2.set_data(index, 2*i);
	++index;
  }

  Matrix<int> product(2,2);
  product = m1*m2;

  CHECK(product.get(0, 0) == 14);
  CHECK(product.get(1, 0) == 30);
  CHECK(product.get(0, 1) == 20);
  CHECK(product.get(1, 1) == 44);
  CHECK(product[0] == 14);
  CHECK(product[1] == 20);
  CHECK(product[2] == 30);
  CHECK(product[3] == 44);
}

TEST_CASE("Test multiplication of a column vector with a matrix") {
  Matrix<int> m(5,5);

  int index{};
  for (int i : std::views::iota(1,26)) {
	m.set_data(index, i);
	++index;
  }

  std::vector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m2(vec);

  Matrix<int> product_1{ m*m2 };
  std::vector<int> product_2{ m*vec };

  for (int i{}; i < 5; ++i) {
	CHECK(product_1[i] == product_2[i]);
	CHECK(product_1[i] == 55 + 5*15*i);
	CHECK(product_2[i] == 55 + 5*15*i);
  }
}

TEST_CASE("Test multiplication of a row vector with a matrix") {
  Matrix<int> m(5,5);

  int index{};
  for (int i : std::views::iota(1,26)) {
	m.set_data(index, i);
	++index;
  }

  std::vector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m2(vec);

  Matrix<int> product_1{ m*m2 };
  std::vector<int> product_2{ m*vec };

  for (int i{}; i < 5; ++i) {
	CHECK(product_1[i] == product_2[i]);
	CHECK(product_1[i] == 55 + 5*15*i);
	CHECK(product_2[i] == 55 + 5*15*i);
  }
}
