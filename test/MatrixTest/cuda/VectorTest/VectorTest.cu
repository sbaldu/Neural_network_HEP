
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "DataFormats/VectorOperations.h"
#include "DataFormats/Matrix.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test sum of two small vectors") {
  const int len{1 << 3};
  std::vector<int> v1(len), v2(len);

  // Initialize data
  std::iota(v1.begin(), v1.end(), 10);
  std::iota(v2.begin(), v2.end(), len);

  // Sum the vectors
  std::vector<int> sum{v1 + v2};

  for (int i{}; i < len; ++i) {
	CHECK(sum[i] == v1[i] + v2[i]);
  }
}

TEST_CASE("Test sum of two large vectors") {
  const int len{1 << 12};
  std::vector<int> v1(len), v2(len);

  // Initialize data
  std::iota(v1.begin(), v1.end(), 10);
  std::iota(v2.begin(), v2.end(), len);

  // Sum the vectors
  std::vector<int> sum{v1 + v2};

  for (int i{}; i < len; ++i) {
	CHECK(sum[i] == v1[i] + v2[i]);
  }
}

TEST_CASE("Test subtraction of two small vectors") {
  const int len{1 << 3};
  std::vector<int> v1(len), v2(len);

  // Initialize data
  std::iota(v1.begin(), v1.end(), 10);
  std::iota(v2.begin(), v2.end(), len);

  // Sum the vectors
  std::vector<int> diff{v1 - v2};

  for (int i{}; i < len; ++i) {
	CHECK(diff[i] == v1[i] - v2[i]);
  }
}

TEST_CASE("Test subtraction of two large vectors") {
  const int len{1 << 12};
  std::vector<int> v1(len), v2(len);

  // Initialize data
  std::iota(v1.begin(), v1.end(), 10);
  std::iota(v2.begin(), v2.end(), len);

  // Sum the vectors
  std::vector<int> diff{v1 - v2};

  for (int i{}; i < len; ++i) {
	CHECK(diff[i] == v1[i] - v2[i]);
  }
}

TEST_CASE("Test multiplication of vector with constant of the same type") {
  const int len{1 << 12};
  std::vector<int> v(len);
  int constant{5};

  // Initialize data
  std::iota(v.begin(), v.end(), 10);

  std::vector<int> multiplied{constant * v};

  for (int i{}; i < len; ++i) {
	CHECK(multiplied[i] == constant * v[i]);
  }
}

TEST_CASE("Test multiplication of vector with constant of a different type") {
  const int len{1 << 12};
  std::vector<int> v(len);
  float constant{8.5f};

  // Initialize data
  std::iota(v.begin(), v.end(), 10);

  std::vector<int> multiplied{constant * v};

  for (int i{}; i < len; ++i) {
	CHECK(multiplied[i] == static_cast<int>(constant * v[i]));
  }
}

TEST_CASE("Test division of vector with constant of the same type") {
  const int len{1 << 12};
  std::vector<int> v(len);
  int constant{5};

  // Initialize data
  std::iota(v.begin(), v.end(), 10);

  std::vector<int> divided{v / constant };

  for (int i{}; i < len; ++i) {
	CHECK(divided[i] == v[i] / constant);
  }
}

TEST_CASE("Test division of vector with constant of a different type") {
  const int len{1 << 12};
  std::vector<int> v(len);
  float constant{8.5f};

  // Initialize data
  std::iota(v.begin(), v.end(), 10);

  std::vector<int> divided{v / constant };

  for (int i{}; i < len; ++i) {
	CHECK(divided[i] == static_cast<int>(v[i] / constant));
  }
}

TEST_CASE("Increment vector with a vector matrix") {
  const int len{1 << 12};
  std::vector<int> vec(len), matrix_vec(len);

  std::iota(vec.begin(), vec.end(), 10);
  std::iota(matrix_vec.begin(), matrix_vec.end(), len/2);

  Matrix<int> mat(len, 1, matrix_vec);

  std::vector<int> vec_copy(vec);
  vec += mat;

  for (int i{}; i < len; ++i) {
	CHECK(vec[i] == vec_copy[i] + mat[i]);
  }
}

TEST_CASE("Decrement vector with a vector matrix") {
  const int len{1 << 12};
  std::vector<int> vec(len), matrix_vec(len);

  std::iota(vec.begin(), vec.end(), 10);
  std::iota(matrix_vec.begin(), matrix_vec.end(), len/2);

  Matrix<int> mat(len, 1, matrix_vec);

  std::vector<int> vec_copy(vec);
  vec -= mat;

  for (int i{}; i < len; ++i) {
	CHECK(vec[i] == vec_copy[i] - mat[i]);
  }
}
