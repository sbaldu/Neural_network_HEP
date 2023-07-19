
#ifndef Activators_h
#define Activators_h

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "Layer.h"

template <typename T>
using shared = std::shared_ptr<T>;

/*
template <typename T>
struct Step {
  __host__ __device__ short operator()(double x) {
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
  double grad(double activated_value) { return 0; }
  std::vector<double> grad(shared<Layer<T>> layer) { 
	return std::vector<double>(layer->size(), 0);
  }

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
  std::vector<double> grad(shared<Layer<T>> layer) {
	return std::vector<double>(layer->size(), 1);
  }
  std::vector<double> grad(std::vector<T> node_values) {
	return std::vector<double>(node_values.size(), 1);
  }
};
*/

template <typename T, typename E, template <typename U> typename Act>
__global__ void apply(T* v1, const E* v2, int n) {
  unsigned int index{threadIdx.x + blockDim.x * blockIdx.x};

  if (index < n) {
    v1[index] = Act<T>()(v2[index]);
  }
}

template <typename T, template <typename E> typename Act>
__global__ void calculate_grad(double* grad, const T* node_values, int n) {
  unsigned int index{threadIdx.x + blockDim.x * blockIdx.x};

  if (index < n) {
    grad[index] = Act<T>().grad(node_values[index]);
  }
}

template <typename T>
struct Sigmoid {
  __host__ __device__ double operator()(double x) { return 1. / (1 + exp(-x)); }

  template <typename E>
  __host__ std::vector<T> operator()(const std::vector<E>& vec) {
    const size_t N{vec.size()};

    std::vector<T> activated(N);

    // Allocate on device
    T* d_act;
    E* d_vec;
    const size_t size_act{N * sizeof(T)};
    const size_t size_vec{N * sizeof(E)};
    cudaMalloc(&d_act, size_act);
    cudaMalloc(&d_vec, size_vec);

    // Create working division
    const int block_size{32};
    const int grid_size{(int)(std::ceil(N / (float)(block_size)))};

    // Launch kernel
    cudaMemcpy(d_vec, vec.data(), size_vec, cudaMemcpyHostToDevice);
    apply<T, E, Sigmoid><<<grid_size, block_size>>>(d_act, d_vec, N);
    cudaMemcpy(activated.data(), d_act, size_act, cudaMemcpyDeviceToHost);

    cudaFree(d_act);
    cudaFree(d_vec);

    return activated;
  }

  // Derivative of the activation function
  __host__ __device__ double grad(double activated_value) {
    return activated_value * (1 - activated_value);
  }

  __host__ std::vector<double> grad(shared<Layer<T>> layer) {
    int N{layer->size()};
    std::vector<double> gradient_values(N);

    // Allocate on device
    T* d_lay;
    double* d_grad;
    const int size_lay{N * sizeof(T)};
    const int size_grad{N * sizeof(double)};

    // Create working division
    const int block_size{32};
    const int grid_size{(int)(std::ceil(N / (float)(block_size)))};

    // Launch kernel
    cudaMemcpy(d_lay, layer->nodes().data(), size_lay, cudaMemcpyHostToDevice);
    calculate_grad<T, Sigmoid><<<grid_size, block_size>>>(d_grad, d_lay, N);
    cudaMemcpy(gradient_values.data(), d_grad, size_grad, cudaMemcpyDeviceToHost);

    cudaFree(d_lay);
    cudaFree(d_grad);

    return gradient_values;
  }

  __host__ std::vector<double> grad(std::vector<T> node_values) {
    int N{node_values.size()};
    std::vector<double> gradient_values(N);

    // Allocate on device
    T* d_lay;
    double* d_grad;
    const int size_lay{N * sizeof(T)};
    const int size_grad{N * sizeof(double)};

    // Create working division
    const int block_size{32};
    const int grid_size{(int)(std::ceil(N / (float)(block_size)))};

    // Launch kernel
    cudaMemcpy(d_lay, node_values.data(), size_lay, cudaMemcpyHostToDevice);
    calculate_grad<<<grid_size, block_size>>>(d_grad, d_lay, N);
    cudaMemcpy(gradient_values.data(), d_grad, size_grad, cudaMemcpyDeviceToHost);

    cudaFree(d_lay);
    cudaFree(d_grad);

    return gradient_values;
  }
};

/*
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
  double grad(double activated_value) {
	return 1 + pow(activated_value, 2);
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
struct Elu {
  double A;

  double operator()(double x) {
    if (x >= 0) {
      return x;
    } else {
      return A * (std::exp(x) - 1);
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
	if (activated_value >= 0) {
	  return 1;
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
	if (activated_value >= 0) {
	  return 1;
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
  double operator()(double x) {
    return x * Sigmoid<T>()(x);
  }

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
  double grad(double x) {
	return Sigmoid<T>()(x) * (1 + x) * (1 - Sigmoid<T>()(x));
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
*/
#endif
