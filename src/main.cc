
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "serial/include/Activators.hpp"
#include "serial/include/ErrorFunction.hpp"
#include "serial/include/Network.hpp"

int main() {
  double eta{11.};
  /* Network<double, double, Sigmoid, MeanSquaredError> net(3, {30, 50, 1}); */
  /* const std::string path_to_trainfile{"../data/training_processed.csv"}; */
  Network<double, double, Sigmoid, MeanSquaredError> net({10, 10, 10, 1});
  const std::string path_to_trainfile{"../data/training_processed_reduced.csv"};

  int n_epochs{10};
  int data_size{250000};
  double previous_accuracy;
  for (int epoch{}; epoch < n_epochs; ++epoch) {
    int n_correct_guesses{};

    std::ifstream file_stream(path_to_trainfile);
    std::string file_row;
    std::string target;
    std::string input;

	// Get rid of the header row
	getline(file_stream, file_row);
    while (getline(file_stream, file_row)) {
      std::stringstream row_stream(file_row);

      // Isolate the target, which is the first value
      getline(row_stream, target, ',');
      int target_value{std::stoi(target)};

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
    }

    double accuracy{static_cast<double>(n_correct_guesses) * 100 / data_size};
    std::cout << "Accuracy = " << accuracy << " %" << std::endl;
    std::cout << "-------------------------------------\n";

    /* if (accuracy - previous_accuracy > 0.1) { */
    /*   previous_accuracy = accuracy; */
    /* } else { */
    /*   break; */
    /* } */

    file_stream.close();
  }
}
