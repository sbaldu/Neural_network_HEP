
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "nnhep.hpp"

int main(int argc, char** argv) {
  double eta{0.975};
  /* double eta{atof(argv[1])}; */
  Network<double, double, Sigmoid, MeanSquaredError> net({30, std::strtol(argv[1], NULL, 10), 1});
  /* Network<double, double, Sigmoid, MeanSquaredError> net({30, 1000, 1}); */
  const std::string path_to_trainfile{"../data/training_data.csv"};
  /* const std::string path_to_trainfile{"../data/training_processed.csv"}; */

  std::vector<double> training_loss_values{};

  const int n_epochs{35};
  /* const int data_size{68113}; */
  const int data_size{178860};
  for (int epoch{}; epoch < n_epochs; ++epoch) {
    int n_correct_guesses{};
    double training_loss{};

    std::ifstream file_stream(path_to_trainfile);
    std::string file_row;
    std::string target;
    std::string input;

    // Get rid of the header row
    /* int count{}; */
    getline(file_stream, file_row);
    while (getline(file_stream, file_row)) {
      /* std::cout << count << '\n'; */
      /* ++count; */
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
      /* std::cout << target_value << ' ' << guess << std::endl; */
      if (guess == target_value) {
        ++n_correct_guesses;
      }

      // We also want to keep track of the value of the loss function
      std::vector<int> target_vec{target_value};
      training_loss += net.get_loss_value(target_vec);
    }

    training_loss_values.push_back(training_loss);

    double accuracy{static_cast<double>(n_correct_guesses) * 100 / data_size};
    std::cout << accuracy << std::endl;

    file_stream.close();
  }

  for (auto x : training_loss_values) {
    std::cout << x << std::endl;
  }
}
