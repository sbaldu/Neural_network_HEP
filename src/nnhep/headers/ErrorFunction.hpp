/// @file ErrorFunction.hpp
/// @brief Error functions for neural networks
///
/// @details Error functions are used to determine the error of a neural network
/// by comparing the output of the network to the expected output. The error
/// function is used to determine the gradient of the error with respect to the
/// weights and biases of the network. The gradient is then used to update the
/// weights and biases of the network using gradient descent.

#ifndef ErrorFunction_h
#define ErrorFunction_h

#include <cmath>
#include <memory>
#include <vector>

#include "../DataFormats/Matrix.hpp"
#include "Layer.hpp"

namespace nnhep {

  template <typename T>
  using shared = std::shared_ptr<T>;

  /// @brief Mean squared error function
  /// @tparam T The type of the node values
  /// @tparam W The type of the weights
  /// @tparam Activator The activation function used by the network
  template <typename T, typename W, template <typename E> typename Activator>
  struct MeanSquaredError {
    /// @brief Calculate the error of the network
    /// @tparam U The type of the expected values
    /// @param node_values The values of the nodes in the network
    /// @param expected_values The expected values of the nodes in the network
    /// @return The error of the network
    ///
    /// @details The error is calculated by taking the mean of the squared
    /// difference between the node values and the expected values.
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

    /// @brief Calculate the gradient of the error with respect to the weights and
    /// biases of the network
    /// @tparam U The type of the expected values
    /// @param expected_values The expected values of the nodes in the network
    /// @param layer_id The id of the layer to calculate the gradient for
    /// @param layers The layers of the network
    /// @param weights The weights of the network
    /// @return The gradient of the error with respect to the weights and biases
    /// of the network
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

  // TODO: Implement other error functions
  /* struct MeanAbsoluteError { */
  /*   double operator()() {} */
  /*   double grad() {} */
  /* }; */

  /* struct LogLoss { */
  /*   double operator()() {} */
  /*   double grad() {} */
  /* }; */

};  // namespace nnhep

#endif
