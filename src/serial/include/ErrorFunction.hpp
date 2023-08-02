
#ifndef ErrorFunction_h
#define ErrorFunction_h

#include <cmath>
#include <memory>
#include <vector>

#include "DataFormats/Matrix.hpp"
#include "Layer.hpp"

template <typename T>
using shared = std::shared_ptr<T>;

template <typename T, typename W, template <typename E> typename Activator>
struct MeanSquaredError {
  template <typename U>
  double operator()(const std::vector<T>& node_values, const std::vector<U>& expected_values) {
    double error{};
    const size_t N{node_values.size()};
    for (size_t node_index{}; node_index < N; ++node_index) {
      error += pow(node_values[node_index] - expected_values[node_index], 2);
    }
    error /= N;

    return error;
  }

  // Derivative of the error function with respect to the activated node values
  template <typename U>
  std::vector<double> grad(const std::vector<U>& expected_values,
                           int layer_id,
                           const std::vector<shared<Layer<T>>>& layers,
                           const std::vector<shared<Matrix<W>>>& weights) {
    if (layers[layer_id + 1] == nullptr) {
      int N{layers[layer_id]->size()};
      std::vector<double> delta(N);
      for (int node_index{}; node_index < N; ++node_index) {
        delta[node_index] = (*layers[layer_id])[node_index] - static_cast<T>(expected_values[node_index]);
      }

      return delta;
    } else {
      Activator<T> act;
      int N{layers[layer_id]->size()};
      std::vector<double> delta(N);

      for (int node_index{}; node_index < N; ++node_index) {
        std::vector<double> previous_delta{grad(expected_values, layer_id + 1, layers, weights)};
        delta[node_index] = act.grad((*layers[layer_id])[node_index]) *
                            (weights[layer_id]->transpose() * previous_delta)[node_index];
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
