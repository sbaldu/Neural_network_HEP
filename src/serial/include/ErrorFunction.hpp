
#ifndef ErrorFunction_h
#define ErrorFunction_h

#include <cmath>
#include <memory>
#include <vector>

#include "../DataFormats/Matrix.hpp"
#include "Layer.hpp"

template <typename T>
using shared = std::shared_ptr<T>;

template <typename T, template <typename E> typename Activator>
struct MeanSquaredError {
  double operator()(const std::vector<T>& node_values, const std::vector<T>& expected_values) {
    double error{};
    int N{node_values.size()};
    for (int node_index{}; node_index < N; ++node_index) {
      error += pow(node_values[node_index] - expected_values[node_index], 2);
    }
    error /= N;

    return error;
  }

  // Derivative of the error function with respect to the activated node values
  std::vector<double> grad(const std::vector<T>& expected_values,
                           shared<Layer<T>> current_layer,
                           shared<Layer<T>> previous_layer,
                           shared<Matrix<T>> weight_matrix) {
    if (weight_matrix == nullptr) {
      int N{current_layer->size()};
      std::vector<double> delta(N);
      for (int node_index{}; node_index < N; ++node_index) {
        delta[node_index] = current_layer->nodes()[node_index] - expected_values[node_index];
      }

      return delta;
    } else {
      Activator<T> act;
      int N{current_layer->size()};
      std::vector<double> delta(N);

      for (int node_index{}; node_index < N; ++node_index) {
        delta[node_index] =
            act.grad(current_layer->nodes()[node_index]) * (*weight_matrix * previous_layer->nodes())[node_index];
      }

      return delta;
    }
  }
};

/* struct MeanAbsoluteError { */
/*   double operator()() {} */
/*   double grad() {} */
/* }; */

/* struct LogLoss { */
/*   double operator()() {} */
/*   double grad() {} */
/* }; */

#endif
