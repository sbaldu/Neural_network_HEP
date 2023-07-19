
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../../src/serial/include/Network.hpp"
#include "../../../src/serial/include/ErrorFunction.hpp"

TEST_CASE("Testing the export of a neural network") {
  double eta{0.1};
  Network<int, double, Sigmoid, MeanSquaredError> net(2, {2, 1});
  std::vector<std::vector<int>> inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<int>> targets{{0}, {1}, {1}, {1}};

  for (int i{}; i < inputs.size(); ++i) {
	net.load_input_layer(inputs[i]);
	net.train(targets[i], eta);
  }
  net.export_network("./output_file.nn");
};
