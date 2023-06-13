
#ifndef ErrorFunction_h
#define ErrorFunction_h

#include <cmath>
#include <memory>
#include <vector>

#include "../DataFormats/Matrix.hpp"
#include "Layer.hpp"

template <typename T>
using shared = std::shared_ptr<T>;

template <typename T, typename W, template <typename E> typename Activator>
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
  template <typename U>
  std::vector<double> grad(const std::vector<U>& expected_values,
                           shared<Layer<T>> current_layer,
                           shared<Layer<T>> next_layer,
                           shared<Matrix<W>> next_layer_matrix) {
    if (next_layer == nullptr) {
      int N{current_layer->size()};
      std::vector<double> delta(N);
      for (int node_index{}; node_index < N; ++node_index) {
        delta[node_index] = (*current_layer)[node_index] - static_cast<T>(expected_values[node_index]);
      }

      return delta;
    } else {
      Activator<T> act;
      int N{current_layer->size()};
      std::vector<double> delta(N);

      for (int node_index{}; node_index < N; ++node_index) {
        delta[node_index] =
            act.grad((*current_layer)[node_index]) * (*next_layer_matrix * next_layer->nodes())[node_index];
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
