
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "nnhep.hpp"

int main() {
  double eta{0.975};
  Network<double, double, Sigmoid, MeanSquaredError> net({30, 300, 1});
  const std::string training_file{"../data/training_data.csv"};
  const std::string test_file{"../data/test_data.csv"};

  const int n_epochs{22};
  const int test_data_size{22358};
  for (int epoch{}; epoch < n_epochs; ++epoch) {
    // Training
    std::ifstream file_stream(training_file);
    std::string file_row;
    std::string target;
    std::string input;

    // Get rid of the header row
    getline(file_stream, file_row);
    while (getline(file_stream, file_row)) {
      std::stringstream row_stream(file_row);

      // Isolate the target, which is the first value
      getline(row_stream, target, ',');
      const int target_value{std::stoi(target)};

      // Get the remainder of the line, which is the input to the network
      getline(row_stream, input);
      std::stringstream input_stream(input);
      net.load_input_layer(input_stream);
      net.train(target_value, eta);
    }

    file_stream.close();
  }

  // Test
  int n_correct_guesses{};

  std::ifstream file_stream(test_file);
  std::string file_row;
  std::string target;
  std::string input;

  // Get rid of the header row
  getline(file_stream, file_row);
  while (getline(file_stream, file_row)) {
    std::stringstream row_stream(file_row);

    // Isolate the target, which is the first value
    getline(row_stream, target, ',');
    const int target_value{std::stoi(target)};

    // Get the remainder of the line, which is the input to the network
    getline(row_stream, input);
    std::stringstream input_stream(input);
    net.load_input_layer(input_stream);
    net.forward_propatation();

    int guess = *net.output_layer().begin();
    if (guess == target_value) {
      ++n_correct_guesses;
    }
  }

  std::cout << "The accuracy on the test dataset is = "
            << n_correct_guesses / static_cast<float>(test_data_size) << std::endl;
}
