
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

template <typename T>
struct Step {
  short operator()(double x) {
    if (x < 0) {
      return 0;
    } else {
      return 1;
    }
  }

  std::vector<short> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Step<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Step<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) { return 0.; }
  std::vector<double> grad(shared<Layer<T>> layer) { return std::vector<double>(layer->size(), 0.); }

  std::vector<double> grad(std::vector<T> node_values) {
    return std::vector<double>(node_values.size(), 0);
  }
};

template <typename T>
struct Linear {
  double operator()(double x) { return x; }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Linear<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Linear<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) { return 1; }
  std::vector<double> grad(shared<Layer<T>> layer) { return std::vector<double>(layer->size(), 1); }
  std::vector<double> grad(std::vector<T> node_values) {
    return std::vector<double>(node_values.size(), 1);
  }
};

template <typename T>
struct Sigmoid {
  double operator()(double x) { return 1. / (1 + std::exp(-x)); }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (size_t i{}; i < N; ++i) {
      activated[i] = Sigmoid<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Sigmoid<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) { return activated_value * (1 - activated_value); }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad((*layer)[i]);
    }

    return gradient_values;
  }
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

template <typename T>
struct Tanh {
  double operator()(double x) { return std::tanh(x); }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Tanh<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Tanh<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) { return 1 + pow(activated_value, 2); }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

template <typename T>
struct Elu {
  double A;

  double operator()(double x) {
    if (x >= 0) {
      return x;
    } else {
      return A * (std::expm1(x));
    }
  }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Elu<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Elu<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) {
    if (activated_value >= 0) {
      return 1;
    } else {
      return activated_value + A;
    }
  }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

template <typename T>
struct Leaky_ReLU {
  double operator()(double x) { return std::max(0.1 * x, x); }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Leaky_ReLU<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Leaky_ReLU<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) {
    if (activated_value >= 0.) {
      return 1.;
    } else {
      return 0.1;
    }
  }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

template <typename T>
struct Parametric_ReLU {
  double A;

  double operator()(double x) { return std::max(A * x, x); }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Parametric_ReLU<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Parametric_ReLU<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double activated_value) {
    if (activated_value >= 0.) {
      return 1.;
    } else {
      return A;
    }
  }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
  std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(node_values[i]);
    }

    return gradient_values;
  }
};

template <typename T>
struct Swish {
  double operator()(double x) { return x * Sigmoid<T>()(x); }

  std::vector<T> operator()(const std::vector<T>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Swish<T>()(vec[i]);
    }

    return activated;
  }

  template <typename E>
  std::vector<T> operator()(const std::vector<E>& vec) {
    size_t N{vec.size()};
    std::vector<T> activated(N);
    for (int i{}; i < N; ++i) {
      activated[i] = Swish<T>()(static_cast<T>(vec[i]));
    }

    return activated;
  }

  // Derivative of the activation function
  double grad(double x) { return Sigmoid<T>()(x) * (1 + x) * (1 - Sigmoid<T>()(x)); }
  std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);
    for (int i{}; i < N; ++i) {
      gradient_values[i] = grad(layer->nodes()[i]);
    }

    return gradient_values;
  }
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
