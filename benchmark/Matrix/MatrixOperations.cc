
#include "Matrix.hpp"

#include "Bench.hpp"
#include "RandomMatrix.hpp"


using sb::Bench;

int main() {
  const int n_rows{1 << 8};
  const int n_cols{1 << 8};
  Matrix<int> m1(n_rows, n_cols);
  Matrix<int> m2(n_rows, n_cols);
  random_matrix(m1);
  random_matrix(m2);

  Bench<long long int> b(100);
  std::cout << "Benchmarking Matrix sum\n";
  b.benchmark([](const Matrix<int>& m1, const Matrix<int>& m2) -> Matrix<int> { return m1 + m2; }, m1, m2);
  b.print();
  std::cout << "Benchmarking Matrix multiplication\n";
  b.benchmark([](const Matrix<int>& m1, const Matrix<int>& m2) -> Matrix<int> { return m1 * m2; }, m1, m2);
  b.print<sb::microseconds>();
  std::cout << "Benchmarking Matrix scalar multiplication\n";
  b.benchmark([](int constant, const Matrix<int>& m1) -> Matrix<int> { return constant * m1; }, 100, m1);
  b.print();
  std::cout << "Benchmarking Matrix increment\n";
  b.benchmark([&m1](const Matrix<int>& m2) -> void { m1 += m2; }, m2);
  b.print();
  std::cout << "Benchmarking Matrix decrement\n";
  b.benchmark([&m1](const Matrix<int>& m2) -> void { m1 -= m2; }, m2);
  b.print();
  std::cout << "Benchmarking Matrix scaling multiplication\n";
  b.benchmark([&m1](int constant) -> void { m1 *= constant; }, 100);
  b.print();
  std::cout << "Benchmarking Matrix scaling division\n";
  b.benchmark([&m1](int constant) -> void { m1 /= constant; }, 100);
  b.print();
  std::cout << "Benchmarking Matrix transposition\n";
  b.benchmark([&m1]() -> void { m1.transpose(); });
  b.print();
}
