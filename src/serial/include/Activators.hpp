/// @file Activators.hpp
/// @brief Contains the declaration of the activation functions and their
/// derivatives.
///
/// @details The activation functions are implemented as structs with a
/// templated operator() that can be used to activate a single value or a vector
/// of values.

#ifndef Activators_h
#define Activators_h

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "Layer.hpp"

template <typename T>
using shared = std::shared_ptr<T>;

/// @brief The identity function.
/// @tparam T The type of the input.
template <typename T>
struct Step {
  /// @brief The identity function.
  /// @param x The input value.
  /// @return The input value.
  short operator()(double x) {
    if (x < 0) {
      return 0;
    } else {
      return 1;
    }
  }
  /// @brief The identity function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<short> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Step<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The identity function.
  /// @param vec The input vector.
  /// @return The input vector.
  /// @details This is an overload of the operator() for vectors of a different
  /// type than the one specified in the template.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Step<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the identity function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the identity function.
  ///
  /// @details The derivative of the identity function is always 0.
  double grad(double activated_value) { return 0.; }
  /// @brief The derivative of the identity function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the identity function.
  ///
  /// @details The derivative of the identity function is always 0.
  std::vector<double> grad(shared<Layer<T>> layer) { return std::vector<double>(layer->size(), 0.); }
  /// @brief The derivative of the identity function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the identity function.
  ///
  /// @details The derivative of the identity function is always 0.
  std::vector<double> grad(std::vector<T> node_values) {
    return std::vector<double>(node_values.size(), 0);
  }
};

/// @brief The linear activation function.
/// @tparam T The type of the input.
template <typename T>
struct Linear {
  /// @brief The linear activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return x; }
  /// @brief The linear activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Linear<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The linear activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  ///
  /// @details This is an overload of the operator() for vectors of a different
  /// type than the one specified in the template.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Linear<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the linear activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the linear activation function.
  double grad(double activated_value) { return 1; }
  /// @brief The derivative of the linear activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the linear activation function.
  std::vector<double> grad(shared<Layer<T>> layer) { return std::vector<double>(layer->size(), 1); }
  /// @brief The derivative of the linear activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the linear activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    return std::vector<double>(node_values.size(), 1);
  }
};

/// @brief The sigmoid activation function.
/// @tparam T The type of the input.
/// @details The sigmoid activation function is defined as:
/// \f[
/// f(x) = \frac{1}{1 + e^{-x}}
/// \f]
/// and its derivative is:
/// \f[
/// f'(x) = f(x)(1 - f(x))
/// \f]
template <typename T>
struct Sigmoid {
  /// @brief The sigmoid activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return 1. / (1 + std::exp(-x)); }
  /// @brief The sigmoid activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (size_t i{}; i < N; ++i) {
      activated[i] = Sigmoid<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The sigmoid activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  ///
  /// @details This is an overload of the operator() for vectors of a different
  /// type than the one specified in the template.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Sigmoid<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the sigmoid activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the sigmoid activation function.
  double grad(double activated_value) { return activated_value * (1 - activated_value); }
  /// @brief The derivative of the sigmoid activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the sigmoid activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad((*layer)[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the sigmoid activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the sigmoid activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

/// @brief The hyperbolic tangent activation function.
/// @tparam T The type of the input.
template <typename T>
struct Tanh {
  /// @brief The hyperbolic tangent activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return std::tanh(x); }
  /// @brief The hyperbolic tangent activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Tanh<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The hyperbolic tangent activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  ///
  /// @details This is an overload of the operator() for vectors of a different
  /// type than the one specified in the template.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Tanh<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the hyperbolic tangent activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the hyperbolic tangent activation function.
  double grad(double activated_value) { return 1 + pow(activated_value, 2); }
  /// @brief The derivative of the hyperbolic tangent activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the hyperbolic tangent activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the hyperbolic tangent activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the hyperbolic tangent activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

/// @brief The Elu activation function.
/// @tparam T The type of the input.
/// @details The Elu activation function is defined as:
/// \f[
/// f(x) = \begin{cases}
/// x & \text{if } x \geq 0 \\
/// A(e^x - 1) & \text{if } x < 0
/// \end{cases}
/// \f]
/// and its derivative is:
/// \f[
/// f'(x) = \begin{cases}
/// 1 & \text{if } x \geq 0 \\
/// f(x) + A & \text{if } x < 0
/// \end{cases}
/// \f]
/// where \f$A\f$ is a constant.
template <typename T>
struct Elu {
  double A;

  /// @brief The Elu activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) {
    if (x >= 0) {
      return x;
    } else {
      return A * (std::expm1(x));
    }
  }
  /// @brief The Elu activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Elu<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The Elu activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Elu<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the Elu activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the Elu activation function.
  double grad(double activated_value) {
    if (activated_value >= 0) {
      return 1;
    } else {
      return activated_value + A;
    }
  }
  /// @brief The derivative of the Elu activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the Elu activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the Elu activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the Elu activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

/// @brief The Leaky ReLU activation function.
/// @tparam T The type of the input.
/// @details The Leaky ReLU activation function is defined as:
/// \f[
/// f(x) = \max(0.1x, x)
/// \f]
/// and its derivative is:
/// \f[
/// f'(x) = \begin{cases}
/// 1 & \text{if } x \geq 0 \\
/// 0.1 & \text{if } x < 0
/// \end{cases}
/// \f]
template <typename T>
struct Leaky_ReLU {
  /// @brief The Leaky ReLU activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return std::max(0.1 * x, x); }
  /// @brief The Leaky ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Leaky_ReLU<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The Leaky ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Leaky_ReLU<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the Leaky ReLU activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the Leaky ReLU activation function.
  double grad(double activated_value) {
    if (activated_value >= 0.) {
      return 1.;
    } else {
      return 0.1;
    }
  }
  /// @brief The derivative of the Leaky ReLU activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the Leaky ReLU activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the Leaky ReLU activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the Leaky ReLU activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

/// @brief The ReLU activation function.
/// @tparam T The type of the input.
/// @details The ReLU activation function is defined as:
/// \f[
///  f(x) = \max(0, x)
///  \f]
///  and its derivative is:
///  \f[
///  f'(x) = \begin{cases}
///  1 & \text{if } x \geq 0 \\
///  0 & \text{if } x < 0
///  \end{cases}
///  \f]
template <typename T>
struct Parametric_ReLU {
  double A;

  /// @brief The ReLU activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return std::max(A * x, x); }
  /// @brief The ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Parametric_ReLU<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Parametric_ReLU<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the ReLU activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the ReLU activation function.
  double grad(double activated_value) {
    if (activated_value >= 0.) {
      return 1.;
    } else {
      return A;
    }
  }
  /// @brief The derivative of the ReLU activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the ReLU activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the ReLU activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the ReLU activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

/// @brief The ReLU activation function.
/// @tparam T The type of the input.
template <typename T>
struct Swish {
  /// @brief The ReLU activation function.
  /// @param x The input value.
  /// @return The input value.
  double operator()(double x) { return x * Sigmoid<T>()(x); }
  /// @brief The ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Swish<T>()(vec[i]);
    }

    return activated;
  }
  /// @brief The ReLU activation function.
  /// @param vec The input vector.
  /// @return The input vector.
  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Swish<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  /// @brief The derivative of the ReLU activation function.
  /// @param activated_value The value of the activated node.
  /// @return The derivative of the ReLU activation function.
  double grad(double x) { return Sigmoid<T>()(x) * (1 + x) * (1 - Sigmoid<T>()(x)); }
  /// @brief The derivative of the ReLU activation function.
  /// @param layer The layer of nodes.
  /// @return The derivative of the ReLU activation function.
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  /// @brief The derivative of the ReLU activation function.
  /// @param node_values The values of the nodes.
  /// @return The derivative of the ReLU activation function.
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

#endif
