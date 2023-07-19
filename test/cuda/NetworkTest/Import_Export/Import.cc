
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "include/Network.h"
#include "include/ErrorFunction.h"

TEST_CASE("Testing the import of a neural network") {
  Network<int, double, Sigmoid, MeanSquaredError> net({2, 1});
  std::vector<std::vector<int>> inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<int>> targets{{0}, {1}, {1}, {1}};

  net.import_network("./output_file.txt");

  CHECK(net.weight_matrix(0)->size() == 2);
  CHECK(net.weight_matrix(1) == nullptr);

  // Check the content of the weights
  CHECK((*net.weight_matrix(0))[0] == -0.164523);
  CHECK((*net.weight_matrix(0))[1] == 0.535009);
};
