
#include <algorithm>
#include <fstream>
#include <string>

#include "include/Network.h"
#include "include/ErrorFunction.h"

int main() {
  double eta{0.01};
  Network<double, double, Sigmoid, MeanSquaredError> net({784, 50, 10});
  const std::string path_to_trainfile{"../../../../data/mnist/mnist_train_norm.csv"};

  int n_epochs{1};
  int data_size{60000};
  double previous_accuracy = 0.;
  for (int epoch{}; epoch < n_epochs; ++epoch) {
    int n_correct_guesses{};

    std::ifstream file_stream(path_to_trainfile);
    std::string file_row;
    std::string target;
    std::string input;
    while (getline(file_stream, file_row)) {
      std::stringstream row_stream(file_row);

      // Isolate the target, which is the first value
      getline(row_stream, target, ',');
      int target_value{std::stoi(target)};
      std::vector<int> target_vector(10, 0);
      target_vector[target_value] = 1;

      // Get the remainder of the line, which is the input to the network
      getline(row_stream, input);
      std::stringstream input_stream(input);
      net.load_input_layer(input_stream);
      net.train(target_vector, eta);
      auto max_it{std::max_element(net.output_layer().begin(), net.output_layer().end())};

      int guess = std::distance(net.output_layer().begin(), max_it);
      if (guess == target_value) {
        ++n_correct_guesses;
      }
    }

    double accuracy{static_cast<double>(n_correct_guesses) * 100 / data_size};
    std::cout << "Accuracy = " << accuracy << " %" << std::endl;
    std::cout << "-------------------------------------\n";

    if (accuracy - previous_accuracy > 0.1) {
      previous_accuracy = accuracy;
    } else {
      break;
    }

    file_stream.close();
  }
}
