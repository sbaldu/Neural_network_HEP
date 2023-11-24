
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "nnhep.hpp"

void train_and_validate(double eta, int hidden_layer_size) {
  Network<double, double, Sigmoid, MeanSquaredError> net({30, hidden_layer_size, 1});
  const std::string training_file{"../data/training_data.csv"};
  const std::string validation_file{"../data/validation_data.csv"};

  std::vector<double> training_loss_values{};
  std::vector<double> validation_loss_values{};

  const int n_epochs{35};
  const int training_data_size{178860};
  const int validation_data_size{22358};
  for (int epoch{}; epoch < n_epochs; ++epoch) {
    // Training
    int n_correct_guesses{};
    double training_loss{};

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

      int guess = *net.output_layer().begin();
      if (guess == target_value) {
        ++n_correct_guesses;
      }

      // We also want to keep track of the value of the loss function
      std::vector<int> target_vec{target_value};
      training_loss += net.get_loss_value(target_vec);
    }

    training_loss_values.push_back(training_loss);

    double training_accuracy{static_cast<double>(n_correct_guesses) * 100 / training_data_size};

    // Validation
    n_correct_guesses = 0;
    double validation_loss{};

    file_stream = std::ifstream(validation_file);
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

      // We also want to keep track of the value of the loss function
      std::vector<int> target_vec{target_value};
      validation_loss += net.get_loss_value(target_vec);
    }

    validation_loss_values.push_back(validation_loss);

    double validation_accuracy{static_cast<double>(n_correct_guesses) * 100 / validation_data_size};
    std::cout << training_accuracy << ',' << validation_accuracy << std::endl;

    file_stream.close();
  }

  for (int i{}; i < n_epochs; ++i) {
    std::cout << training_loss_values[i] << ',' << validation_loss_values[i] << std::endl;
  }
}

int main(int argc, char** argv) { train_and_validate(0.975, 300); }
