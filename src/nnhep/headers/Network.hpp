
#pragma once

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

namespace nnhep {

  template <typename W>
  void random_matrix(Matrix<W>& matrix);

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, typename LW, template <typename K> typename Act>
            typename Loss>
  class Network {
  private:
    int n_layers;
    std::vector<std::shared_ptr<Layer<T>>> m_layers;
    std::vector<Matrix<T>> m_weights;
    std::vector<std::vector<T>> m_bias;

  public:
    explicit Network(const std::vector<T>& nodes_per_layer);

    void load_input_layer(std::stringstream& stream);
    void load_input_layer(const std::vector<T>& vec);
    void load_from_file(const std::string& path_to_file);

    const Matrix<T>& weight_matrix(int layer_id) const;
    const std::vector<T>& output_layer() const;

    void set_matrix_data(int layer_id, Matrix<T> weight_matrix);
    void set_bias_data(int layer_id, std::vector<T> bias_vector);

    std::vector<T> forward_propatation(std::shared_ptr<Layer<T>>,
                                       const Matrix<T>&,
                                       const std::vector<T>&);
    void forward_propatation();

    template <typename U>
    void back_propagation(const std::vector<U>& target, int layer_id, double eta);
    template <typename U>
    void back_propagation(double eta, const std::vector<U>& target);

    template <typename U>
    void train(const std::vector<U>& target, double eta);
    template <typename U>
    void train(U target, double eta);

    void import_network(const std::string& filepath);
    void export_network(const std::string& filepath);

    template <typename U>
    double get_loss_value(const std::vector<U>& target);

    template <typename U,
              typename P,
              template <typename Q>
              typename A,
              template <typename F, typename LP, template <typename Y> typename Ac>
              typename L>
    friend std::ostream& operator<<(std::ostream& out, const Network<U, P, A, L>& Net);
  };

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  Network<T, Activator, Loss>::Network(const std::vector<T>& nodes_per_layer)
      : n_layers{static_cast<int>(nodes_per_layer.size())},
        m_layers(n_layers + 1),
        m_weights(n_layers + 1),
        m_bias(n_layers - 1) {
    for (int i{}; i < n_layers - 1; ++i) {
      m_layers[i] = std::shared_ptr<Layer<T>>(nodes_per_layer[i]);
      m_weights[i] = std::shared_ptr<Matrix<T>>(nodes_per_layer[i + 1],
                                                          nodes_per_layer[i]);
      m_bias[i] = std::shared_ptr<std::vector<T>>(nodes_per_layer[i + 1]);

      // Generate random weight matrices
      random_matrix(m_weights[i]);
    }
    m_layers[n_layers - 1] = std::shared_ptr<Layer<T>>(nodes_per_layer.back());
    m_layers[n_layers] = nullptr;
    m_weights[n_layers - 1] = nullptr;
    m_weights[n_layers] = nullptr;
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, Activator, Loss>::load_input_layer(std::stringstream& stream) {
    m_layers[0]->load(stream);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, Activator, Loss>::load_input_layer(const std::vector<T>& vec) {
    m_layers[0]->set_node_data(vec);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, Activator, Loss>::load_from_file(const std::string& path_to_file) {}

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  const Matrix<T>& Network<T, Activator, Loss>::weight_matrix(
      int layer_id) const {
    return m_weights[layer_id];
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  const std::vector<T>& Network<T, Activator, Loss>::output_layer() const {
    return m_layers[n_layers - 1]->nodes();
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, T, Activator, Loss>::set_matrix_data(int layer_id,
                                                       Matrix<T> weight_matrix) {
    m_weights[layer_id] = std::shared_ptr<Matrix<T>>(weight_matrix);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, T, Activator, Loss>::set_bias_data(int layer_id,
                                                     std::vector<T> bias_vector) {
    m_bias[layer_id] = std::shared_ptr<std::vector<T>>(bias_vector);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  std::vector<T> Network<T, Activator, Loss>::forward_propatation(
      std::shared_ptr<Layer<T>> layer,
      const Matrix<T>& weight_matrix,
      const std::vector<T>& bias_vector) {
    std::vector<T> next_layer_nodes{*weight_matrix * layer->nodes() + *bias_vector};

    return Activator<T>()(next_layer_nodes);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, T, Activator, Loss>::forward_propatation() {
    for (int i{}; i < n_layers - 1; ++i) {
      std::vector<T> new_layer_data{
          forward_propatation(m_layers[i], m_weights[i], m_bias[i])};
      m_layers[i + 1]->set_node_data(new_layer_data);
    }
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  void Network<T, Activator, Loss>::back_propagation(const std::vector<U>& target,
                                                        int layer_id,
                                                        double eta) {
    Loss<T, Activator> loss_function;
    Matrix<T> loss_grad(loss_function.grad(target, layer_id + 1, m_layers, m_weights));
    Matrix<T> activated_values_grad(m_layers[layer_id]->nodes());
    *m_weights[layer_id] -= eta * (loss_grad * activated_values_grad.transpose());
    *m_bias[layer_id] -= eta * loss_grad;
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  void Network<T, Activator, Loss>::back_propagation(double eta,
                                                        const std::vector<U>& target) {
    for (int layer_id{n_layers - 2}; layer_id >= 0; --layer_id) {
      back_propagation(target, layer_id, eta);
    }
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  void Network<T, Activator, Loss>::train(const std::vector<U>& target, double eta) {
    forward_propatation();
    back_propagation(eta, target);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  void Network<T, Activator, Loss>::train(U target, double eta) {
    std::vector<U> target_vector{target};
    forward_propatation();
    back_propagation(eta, target_vector);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, Activator, Loss>::import_network(const std::string& filepath) {
    std::ifstream file_stream(filepath);

    std::string file_row;
    int i{};
    while (getline(file_stream, file_row)) {
      std::vector<T> weights;
      std::vector<T> bias;

      std::stringstream row_stream(file_row);
      std::string value;
      // First we fill the weight matrix
      while (getline(row_stream, value, ',')) {
        weights.push_back(std::stod(value));
      }
      // Then we get the next line and fill the bias vector
      getline(file_stream, file_row);
      row_stream = std::stringstream(file_row);
      while (getline(row_stream, value, ',')) {
        bias.push_back(std::stod(value));
      }

      m_weights[i] = std::shared_ptr<Matrix<T>>(
          m_weights[i]->nrows(), m_weights[i]->ncols(), weights);
      m_bias[i] = std::shared_ptr<std::vector<T>>(bias);
    }
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  void Network<T, Activator, Loss>::export_network(const std::string& filepath) {
    std::ofstream ofile;
    ofile.open(filepath);

    if (!ofile.is_open()) {
      std::cout << "The file is not open\n";
      return;
    }

    for (int i{}; i < n_layers - 1; ++i) {
      ofile << *m_weights[i] << '\n';
      ofile << *m_bias[i] << '\n';
    }

    ofile.close();
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  double Network<T, Activator, Loss>::get_loss_value(const std::vector<U>& target) {
    return Loss<T, Activator>()(m_layers[n_layers - 1]->nodes(), target);
  }

  template <typename T,
            template <typename F>
            typename Activator,
            template <typename E, template <typename K> typename Act>
            typename Loss>
  std::ostream& operator<<(std::ostream& out, const Network<T, Activator, Loss>& Net) {
    for (int i{}; i < Net.n_layers; ++i) {
      out << "Layer_" << i << ' ';
      out << Net.m_layers[i];
    }

    return out;
  }

  template <typename T>
  inline void random_matrix(Matrix<T>& matrix) {
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis(-0.5, 0.5);

    for (int i{}; i < matrix->size(); ++i) {
      matrix->set_data(i, dis(gen));
    }
  }

};  // namespace nnhep
