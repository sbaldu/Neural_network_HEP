
#include <iostream>

#include "include/Network.hpp"
#include "include/ErrorFunction.hpp"

int main() {
  double eta{0.05};
  Network<int, double, Sigmoid, MeanSquaredError> net(2, {2, 1});
  std::vector<std::vector<int>> inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<int>> targets{{0}, {1}, {1}, {0}};

  int n_epochs{1000};
  for (int epoch{}; epoch < n_epochs; ++epoch) {
	for (int i{}; i < inputs.size(); ++i) {
	  net.load_input_layer(inputs[i]);
	  net.train(targets[i], eta);
	  std::cout << net.output_layer()[0] << std::endl;
	}
	std::cout << "-------------------------------------\n";
  }
}
