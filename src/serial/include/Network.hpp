
#ifndef Network_h
#define Network_h

#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "Activators.hpp"
#include "Layer.hpp"
#include "../DataFormats/Matrix.hpp"
#include "../DataFormats/VectorOperations.hpp"

template <typename T>
using shared = std::shared_ptr<T>;

template <typename W>
void random_matrix(shared<Matrix<W>> matrix);

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
class Network {
private:
  std::vector<shared<Layer<T>>> m_layers;
  std::vector<shared<Matrix<W>>> m_weights;
  std::vector<shared<std::vector<W>>> m_bias;
  int n_layers;

  inline void normalize();

public:
  Network() = delete;
  Network(int n_layers, std::vector<int> nodes_per_layer);

  void load_input_layer(std::stringstream& stream);
  void load_input_layer(const std::vector<T>& vec);

  void load_from_file(const std::string& path_to_file);

  // Getters
  const shared<Matrix<W>> weight_matrix(int layer_id) const;
  const std::vector<T>& output_layer() const;

  // Setters for the weight matrices
  void set_matrix_data(int layer_id, Matrix<W> weight_matrix);
  void set_matrix_data(int layer_id, shared<Matrix<W>> weight_matrix_ptr);

  // Setters for the bias vectors
  void set_bias_data(int layer_id, std::vector<W> bias_vector);
  void set_bias_data(int layer_id, shared<std::vector<W>> bias_vector_ptr);

  std::vector<T> forward_propatation(shared<Layer<T>>, shared<Matrix<W>>, shared<std::vector<W>>);
  void forward_propatation();

  template <typename U>
  void back_propagation(const std::vector<U>& target,
						int layer_id,
                        double eta);
  template <typename U>
  void back_propagation(double eta, const std::vector<U>& target);

  template <typename U>
  void train(const std::vector<U>& target, double eta);

  template <typename U>
  void train(U target, double eta);

  template <typename U,
            typename P,
            template <typename Q>
            typename A,
            template <typename F, typename LP, template <typename Y> typename Ac>
            typename L>
  friend std::ostream& operator<<(std::ostream& out, const Network<U, P, A, L>& Net);
};

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
Network<T, W, Activator, Loss>::Network(int n_layers, std::vector<int> nodes_per_layer)
    : n_layers{n_layers},
      m_layers(n_layers + 1),
      m_weights(n_layers + 1, std::make_shared<Matrix<W>>()),
      m_bias(n_layers, std::make_shared<std::vector<W>>()) {
  for (int i{}; i < n_layers - 1; ++i) {
    m_layers[i] = std::make_shared<Layer<T>>(nodes_per_layer[i]);
    m_weights[i] = std::make_shared<Matrix<W>>(nodes_per_layer[i + 1], nodes_per_layer[i]);
    m_bias[i]->resize(nodes_per_layer[i]);

    // Generate random weight matrices
    random_matrix(m_weights[i]);
  }
  m_layers[n_layers - 1] = std::make_shared<Layer<T>>(nodes_per_layer.back());
  m_layers[n_layers] = nullptr;
  m_weights[n_layers - 1] = nullptr;
  m_weights[n_layers] = nullptr;
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
inline void Network<T, W, Activator, Loss>::normalize() {
  m_layers[0]->normalize();
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::load_input_layer(std::stringstream& stream) {
  m_layers[0]->load(stream);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::load_input_layer(const std::vector<T>& vec) {
  m_layers[0]->set_node_data(vec);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::load_from_file(const std::string& path_to_file) {}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
const shared<Matrix<W>> Network<T, W, Activator, Loss>::weight_matrix(int layer_id) const {
  return m_weights[layer_id];
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
const std::vector<T>& Network<T, W, Activator, Loss>::output_layer() const {
  return m_layers[n_layers-1]->nodes();
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::set_matrix_data(int layer_id, Matrix<W> weight_matrix) {
  m_weights[layer_id] = std::make_shared<Matrix<W>>(weight_matrix);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::set_matrix_data(int layer_id, shared<Matrix<W>> weight_matrix_ptr) {
  m_weights[layer_id] = weight_matrix_ptr;
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::set_bias_data(int layer_id, std::vector<W> bias_vector) {
  m_bias[layer_id] = std::make_shared<std::vector<W>>(bias_vector);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::set_bias_data(int layer_id, shared<std::vector<W>> bias_vector_ptr) {
  m_bias[layer_id] = bias_vector_ptr;
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
std::vector<T> Network<T, W, Activator, Loss>::forward_propatation(shared<Layer<T>> layer,
                                                                   shared<Matrix<W>> weight_matrix,
                                                                   shared<std::vector<W>> bias_vector) {
  std::vector<W> next_layer_nodes{*weight_matrix * layer->nodes() + *bias_vector};

  return Activator<T>()(next_layer_nodes);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
void Network<T, W, Activator, Loss>::forward_propatation() {
  for (int i{}; i < n_layers - 1; ++i) {
    std::vector<T> new_layer_data{forward_propatation(m_layers[i], m_weights[i], m_bias[i])};
    m_layers[i + 1]->set_node_data(new_layer_data);
  }
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
template <typename U>
void Network<T, W, Activator, Loss>::back_propagation(const std::vector<U>& target,
													  int layer_id,
                                                      double eta) {
  Loss<T, W, Activator> loss_function;
  Matrix<W> loss_grad(loss_function.grad(target, m_layers[layer_id+1], m_layers[layer_id+2], m_weights[layer_id+1]));
  Matrix<T> activated_values_grad(m_layers[layer_id]->nodes());
  *m_weights[layer_id] -= eta * (loss_grad * activated_values_grad.transpose());
  *m_bias[layer_id] -= eta * loss_grad;
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
template <typename U>
void Network<T, W, Activator, Loss>::back_propagation(double eta, const std::vector<U>& target) {
  for (int layer_id{n_layers - 2}; layer_id >= 0; --layer_id) {
    back_propagation(target,
					 layer_id,
                     eta);
  }
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
template <typename U>
void Network<T, W, Activator, Loss>::train(const std::vector<U>& target, double eta) {
  // Standardization of the input
  if (!std::is_integral<T>::value) {
	normalize();
  }

  forward_propatation();
  back_propagation(eta, target);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
template <typename U>
void Network<T, W, Activator, Loss>::train(U target, double eta) {
  // Standardization of the input
  if (!std::is_integral<T>::value) {
	normalize();
  }

  std::vector<U> target_vector{target};
  forward_propatation();
  back_propagation(eta, target_vector);
}

template <typename T,
          typename W,
          template <typename F>
          typename Activator,
          template <typename E, typename LW, template <typename K> typename Act>
          typename Loss>
std::ostream& operator<<(std::ostream& out, const Network<T, W, Activator, Loss>& Net) {
  for (int i{}; i < Net.n_layers; ++i) {
    out << "Layer_" << i << ' ';
    out << Net.m_layers[i];
  }

  return out;
}

template <typename W>
inline void random_matrix(shared<Matrix<W>> matrix) {
  std::mt19937 gen;
  std::uniform_real_distribution<W> dis(-0.5, 0.5);

  for (int i{}; i < matrix->size(); ++i) {
    matrix->set_data(i, dis(gen));
  }
}

#endif
