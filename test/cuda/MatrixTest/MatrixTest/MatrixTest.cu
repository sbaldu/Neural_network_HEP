
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "DataFormats/Matrix.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test sum of two vector matrices") {
  const int len{1 << 10};
  std::vector<int> v1(len), v2(len);
  std::iota(v1.begin(), v1.end(), 1);
  std::iota(v2.begin(), v2.end(), len);

  Matrix<int> m1(v1);
  Matrix<int> m2(v2);

  Matrix<int> sum{m1 + m2};
  sum.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(sum[i] == m1[i] + m2[i]);
  }
}

TEST_CASE("Test increment of a matrix with another") {
  const int len{1 << 10};
  std::vector<int> v1(len), v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), len);

  Matrix<int> m1(v1);
  Matrix<int> m2(v2);
  Matrix<int> m1_copy(m1);

  m1 += m2;
  m1.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(m1[i] == m1_copy[i] + m2[i]);
  }
}

TEST_CASE("Test overload increment of a matrix with another") {
  const int len{1 << 10};
  std::vector<int> v1(len);
  std::vector<double> v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), static_cast<double>(len));

  Matrix<int> m1(v1);
  Matrix<double> m2(v2);
  Matrix<int> m1_copy(m1);

  m1 += m2;
  m1.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(m1[i] == m1_copy[i] + m2[i]);
  }
}
TEST_CASE("Test overload increment with swapped types") {
  const int len{1 << 10};
  std::vector<int> v1(len);
  std::vector<double> v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), static_cast<double>(len));

  Matrix<int> m1(v1);
  Matrix<double> m2(v2);
  Matrix<double> m2_copy(m2);

  m2 += m1;
  m2.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(m2[i] == m2_copy[i] + m1[i]);
  }
}

TEST_CASE("Test decrement of a matrix with another") {
  const int len{1 << 10};
  std::vector<int> v1(len), v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), len);

  Matrix<int> m1(v1);
  Matrix<int> m2(v2);
  Matrix<int> m2_copy(m2);

  m2 -= m1;
  m2.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(m2[i] == len);
    CHECK(m2[i] == m2_copy[i] - m1[i]);
  }
}

TEST_CASE("Test overload decrement of a matrix with another") {
  const int len{1 << 10};
  std::vector<int> v1(len);
  std::vector<double> v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), static_cast<double>(len));

  Matrix<int> m1(v1);
  Matrix<double> m2(v2);
  Matrix<double> m2_copy(m2);

  m2 -= m1;
  m2.memCpyDtH();

  for (int i{}; i < len; ++i) {
    CHECK(m2[i] == len);
    CHECK(m2[i] == m2_copy[i] - m1[i]);
  }
}

TEST_CASE("Test multiplication of matrix by a constant") {
  // Define dimensions
  const int N{1 << 5};
  const int M{1 << 6};
  std::vector<int> vec(N * M);

  // Initialize data
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m(N, M, vec);
  Matrix<int> m_copy(m);

  // Multiply by constant
  const double c{3.5};
  m *= c;
  m.memCpyDtH();

  for (int i{}; i < N * M; ++i) {
    CHECK(m[i] == static_cast<int>(m_copy[i] * c));
  }
}

TEST_CASE("Test division of matrix by a constant") {
  // Define dimensions
  const int N{1 << 5};
  const int M{1 << 6};
  std::vector<int> vec(N * M);

  // Initialize data
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m(N, M, vec);
  Matrix<int> m_copy(m);

  // Multiply by constant
  const double c{3.5};
  m /= c;
  m.memCpyDtH();

  for (int i{}; i < N * M; ++i) {
    CHECK(m[i] == static_cast<int>(m_copy[i] / c));
  }
}

TEST_CASE("Test scalar product between two vector matrices") {
  const int len{1 << 10};
  std::vector<int> v1(len), v2(len);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), len);

  Matrix<int> m1(1, len, v1);
  Matrix<int> m2(len, 1, v2);

  Matrix<int> product{m1 * m2};
  product.memCpyDtH();

  CHECK(product.size() == 1);

  int tmp{};
  for (int i{}; i < len; ++i) {
    tmp += m1[i] * m2[i];
  }
  CHECK(product[0] == tmp);
}

TEST_CASE("Test 'ket-bra' product between two vector matrices") {
  const int N{1 << 10};
  std::vector<int> v1(N), v2(N);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), N);

  Matrix<int> m1(N, 1, v1);
  Matrix<int> m2(1, N, v2);

  Matrix<int> product{m1 * m2};
  product.memCpyDtH();

  CHECK(product.size() == N * N);

  for (int i{}; i < N; ++i) {
    for (int j{}; j < N; ++j) {
      CHECK(product[i * N + j] == m1[i] * m2[j]);
    }
  }
}

TEST_CASE("Test matrix product between two 32x32 matrices") {
  const int N{1 << 5};
  Matrix<int> m1(N, N);
  Matrix<int> m2(N, N);

  int index{};
  for (int i : std::views::iota(1, N * N + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, 2 * i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, N);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < N; ++j) {
      int tmp{};
      for (int k{}; k < N; ++k) {
        // Accumulate the partial results
        tmp += m1[i * N + k] * m2[k * N + j];
      }

      CHECK(tmp == product[i * N + j]);
    }
  }
}

TEST_CASE("Test matrix product between two 5x5 matrices") {
  const int N{5};
  Matrix<int> m1(N, N);
  Matrix<int> m2(N, N);

  int index{};
  for (int i : std::views::iota(1, N * N + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, 2 * i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, N);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < N; ++j) {
      int tmp{};
      for (int k{}; k < N; ++k) {
        // Accumulate the partial results
        tmp += m1[i * N + k] * m2[k * N + j];
      }

      CHECK(tmp == product[i * N + j]);
    }
  }
}

TEST_CASE("Test matrix product between two 4x8 and 8x4 matrices") {
  const int N{1 << 2};
  const int K{1 << 3};
  const int M{1 << 2};
  Matrix<int> m1(N, K);
  Matrix<int> m2(K, M);

  int index{};
  for (int i : std::views::iota(1, N * K + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, M);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < M; ++j) {
      int tmp{};
      for (int k{}; k < K; ++k) {
        // Accumulate the partial results
        tmp += m1[i * K + k] * m2[k * M + j];
      }

      CHECK(tmp == product[i * M + j]);
    }
  }
}

TEST_CASE("Test matrix product between two 512x1024 and 1024x512 matrices") {
  const int N{1 << 9};
  const int K{1 << 10};
  const int M{1 << 9};
  Matrix<int> m1(N, K);
  Matrix<int> m2(K, M);

  int index{};
  for (int i : std::views::iota(1, N * K + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, M);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < M; ++j) {
      int tmp{};
      for (int k{}; k < K; ++k) {
        // Accumulate the partial results
        tmp += m1[i * K + k] * m2[k * M + j];
      }

      CHECK(tmp == product[i * M + j]);
    }
  }
}

TEST_CASE("Test matrix product between two 8x4 and 4x8 matrices") {
  const int N{1 << 3};
  const int K{1 << 2};
  const int M{1 << 3};
  Matrix<int> m1(N, K);
  Matrix<int> m2(K, M);

  int index{};
  for (int i : std::views::iota(1, N * K + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, M);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < M; ++j) {
      int tmp{};
      for (int k{}; k < K; ++k) {
        // Accumulate the partial results
        tmp += m1[i * K + k] * m2[k * M + j];
      }

      CHECK(tmp == product[i * M + j]);
    }
  }
}

TEST_CASE("Test matrix product between two 1024x512 and 512x1024 matrices") {
  const int N{1 << 10};
  const int K{1 << 9};
  const int M{1 << 10};
  Matrix<int> m1(N, K);
  Matrix<int> m2(K, M);

  int index{};
  for (int i : std::views::iota(1, N * K + 1)) {
    m1.set_data(index, i);
    m2.set_data(index, i);
    ++index;
  }
  m1.memCpyHtD();
  m2.memCpyHtD();

  Matrix<int> product(N, M);
  product = m1 * m2;
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    for (int j{}; j < M; ++j) {
      int tmp{};
      for (int k{}; k < K; ++k) {
        // Accumulate the partial results
        tmp += m1[i * K + k] * m2[k * M + j];
      }

      CHECK(tmp == product[i * M + j]);
    }
  }
}

TEST_CASE("Test multiplication of a small column vector with a matrix") {
  const int N{1 << 2};
  Matrix<int> m(N, N);

  int index{};
  for (int i : std::views::iota(1, N * N + 1)) {
    m.set_data(index, i);
    ++index;
  }
  m.memCpyHtD();

  std::vector<int> vec(N);
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m2(vec);

  Matrix<int> product{m * m2};
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    int tmp{};
    for (int k{}; k < N; ++k) {
      // Accumulate the partial results
      tmp += m[i * N + k] * vec[k];
    }

    CHECK(tmp == product[i]);
  }
}

TEST_CASE("Test multiplication of a large column vector with a matrix") {
  const int N{1 << 9};
  Matrix<int> m(N, N);

  int index{};
  for (int i : std::views::iota(1, N * N + 1)) {
    m.set_data(index, i);
    ++index;
  }
  m.memCpyHtD();

  std::vector<int> vec(N);
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m2(vec);

  Matrix<int> product{m * m2};
  product.memCpyDtH();

  for (int i{}; i < N; ++i) {
    int tmp{};
    for (int k{}; k < N; ++k) {
      // Accumulate the partial results
      tmp += m[i * N + k] * vec[k];
    }

    CHECK(tmp == product[i]);
  }
}

/*
TEST_CASE("Test multiplication of a row vector with a matrix") {
  Matrix<int> m(5, 5);

  int index{};
  for (int i : std::views::iota(1, 26)) {
    m.set_data(index, i);
    ++index;
  }

  std::vector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1);
  Matrix<int> m2(vec);

  Matrix<int> product_1{m * m2};
  std::vector<int> product_2{m * vec};

  for (int i{}; i < 5; ++i) {
    CHECK(product_1[i] == product_2[i]);
    CHECK(product_1[i] == 55 + 5 * 15 * i);
    CHECK(product_2[i] == 55 + 5 * 15 * i);
  }
}
*/
