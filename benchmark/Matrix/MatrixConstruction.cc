
#include <algorithm>
#include <random>

#include "Matrix.hpp"

#include "Bench.hpp"
#include "RandomMatrix.hpp"

using sb::Bench;

int main() {
  const int n_rows{1 << 9};
  const int n_cols{1 << 9};

  // create a vector of n_rows * n_cols random elements
  std::vector<int> v(n_rows * n_cols);
  std::ranges::for_each(v, [](int& x) { x = std::rand(); });

  Bench<long long int> b(100);
  std::cout << "Benchmarking first Matrix constructor\n";
  b.benchmark([](const int rows, const int cols) -> void { Matrix<int> mat(rows, cols); }, n_rows, n_cols);
  b.print();
  std::cout << "Benchmarking second Matrix constructor\n";
  b.benchmark([](const int rows, const int cols, const std::vector<int>& vec) -> void { Matrix<int> mat(rows, cols, vec); }, n_rows, n_cols, v);
  b.print();
  std::cout << "Benchmarking third Matrix constructor\n";
  b.benchmark([](const std::vector<int>& vec) -> void { Matrix<int> mat(vec); }, v);
  b.print();
}
