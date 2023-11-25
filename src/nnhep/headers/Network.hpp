/// @file Network.hpp
/// @brief Neural network
///
/// @details A neural network is a collection of layers. The network contains
/// the weights and biases of the nodes in the network. The network can be
/// trained using gradient descent to minimize the error of the network.

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

namespace nnhep {

  template <typename T>
  using shared = std::shared_ptr<T>;

  /// @brief Generate a random matrix
  /// @tparam W The type of the matrix elements
  /// @param matrix The matrix to fill with random values
  ///
  /// @details The random values are generated using a uniform distribution
  /// between -0.5 and 0.5.
  template <typename W>
  void random_matrix(shared<Matrix<W>> matrix);

  /// @brief Neural network
  /// @tparam T The type of the node values
  /// @tparam W The type of the weights
  /// @tparam Activator The activation function used by the network
  /// @tparam Loss The error function used by the network
  template <typename T,
            typename W,
            template <typename F>
            typename Activator,
            template <typename E, typename LW, template <typename K> typename Act>
            typename Loss>
  class Network {
  private:
    int n_layers;
    std::vector<shared<Layer<T>>> m_layers;
    std::vector<shared<Matrix<W>>> m_weights;
    std::vector<shared<std::vector<W>>> m_bias;

  public:
    /// @brief Default constructor
    Network() = delete;
    /// @brief Constructor
    /// @param nodes_per_layer The number of nodes in each layer of the network
    explicit Network(const std::vector<T>& nodes_per_layer);

    /// @brief Load the values of the input layer from a stream
    /// @param stream The stream to read the values from
    /// @details The values are expected to be provided in csv format
    void load_input_layer(std::stringstream& stream);
    /// @brief Load the values of the input layer from a vector
    /// @param vec The vector to read the values from
    void load_input_layer(const std::vector<T>& vec);
    /// @brief Load the values of the network from a file
    /// @param path_to_file The path to the file to read the values from
    /// @details The values are expected to be provided in csv format
    void load_from_file(const std::string& path_to_file);

    /// @brief Get the weight matrix of a layer
    /// @param layer_id The id of the layer
    /// @return The weight matrix of the layer
    const shared<Matrix<W>> weight_matrix(int layer_id) const;
    /// @brief Get the values of the output layer
    /// @return The values of the output layer
    const std::vector<T>& output_layer() const;

    /// @brief Setters for the weight matrices
    /// @param layer_id The id of the layer
    /// @param weight_matrix The weight matrix to set
    void set_matrix_data(int layer_id, Matrix<W> weight_matrix);
    /// @brief Setters for the weight matrices
    /// @param layer_id The id of the layer
    /// @param weight_matrix_ptr The weight matrix to set
    void set_matrix_data(int layer_id, shared<Matrix<W>> weight_matrix_ptr);

    /// @brief Setters for the bias vectors
    /// @param layer_id The id of the layer
    /// @param bias_vector The bias vector to set
    void set_bias_data(int layer_id, std::vector<W> bias_vector);
    /// @brief Setters for the bias vectors
    /// @param layer_id The id of the layer
    /// @param bias_vector_ptr The bias vector to set
    void set_bias_data(int layer_id, shared<std::vector<W>> bias_vector_ptr);

    /// @brief Forward propagate the values of the network
    /// @param layer The layer to forward propagate
    /// @param weight_matrix The weight matrix to use
    /// @param bias_vector The bias vector to use
    /// @return The values of the next layer
    std::vector<T> forward_propatation(shared<Layer<T>>, shared<Matrix<W>>, shared<std::vector<W>>);
    /// @brief Forward propagate the values of the network
    ///
    /// @details This function is used to forward propagate the values of the
    /// network. The values of the input layer are used as input to the network.
    /// The values of the output layer are stored in the output layer of the
    /// network.
    void forward_propatation();

    /// @brief Back propagate the error of the network
    /// @tparam U The type of the expected values
    /// @param target The expected values of the output layer
    /// @param layer_id The id of the layer to back propagate
    /// @param eta The learning rate of the network
    template <typename U>
    void back_propagation(const std::vector<U>& target, int layer_id, double eta);
    /// @brief Back propagate the error of the network
    /// @tparam U The type of the expected values
    /// @param target The expected values of the output layer
    /// @param eta The learning rate of the network
    template <typename U>
    void back_propagation(double eta, const std::vector<U>& target);

    /// @brief Train the network
    /// @tparam U The type of the expected values
    /// @param target The expected values of the output layer
    /// @param eta The learning rate of the network
    template <typename U>
    void train(const std::vector<U>& target, double eta);
    /// @brief Train the network
    /// @tparam U The type of the expected values
    /// @param target The expected values of the output layer
    /// @param eta The learning rate of the network
    template <typename U>
    void train(U target, double eta);

    // Import and export functions
    /// @brief Import the values of the network from a file
    /// @param filepath The path to the file to read the values from
    /// @details The values are expected to be provided in csv format
    void import_network(const std::string& filepath);
    /// @brief Export the values of the network to a file
    /// @param filepath The path to the file to write the values to
    /// @details The values are provided in csv format
    void export_network(const std::string& filepath);

    /// @brief Get the value of the error for the output values with respect to the target
    /// @tparam U The type of the expected values
    /// @param target The expected values of the output layer
    /// @return The error value of the network
    template <typename U>
    double get_loss_value(const std::vector<U>& target);

    /// @brief Print the network
    /// @tparam U The type of the node values
    /// @tparam P The type of the weights
    /// @tparam A The activation function used by the network
    /// @tparam L The error function used by the network
    /// @param out The stream to print to
    /// @param Net The network to print
    /// @return The stream
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
  Network<T, W, Activator, Loss>::Network(const std::vector<T>& nodes_per_layer)
      : n_layers{static_cast<int>(nodes_per_layer.size())},
        m_layers(n_layers + 1),
        m_weights(n_layers + 1),
        m_bias(n_layers - 1) {
    for (int i{}; i < n_layers - 1; ++i) {
      m_layers[i] = std::make_shared<Layer<T>>(nodes_per_layer[i]);
      m_weights[i] = std::make_shared<Matrix<W>>(nodes_per_layer[i + 1], nodes_per_layer[i]);
      m_bias[i] = std::make_shared<std::vector<W>>(nodes_per_layer[i + 1]);

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
    return m_layers[n_layers - 1]->nodes();
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
    Matrix<W> loss_grad(loss_function.grad(target, layer_id + 1, m_layers, m_weights));
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
      back_propagation(target, layer_id, eta);
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
  void Network<T, W, Activator, Loss>::import_network(const std::string& filepath) {
    std::ifstream file_stream(filepath);

    std::string file_row;
    int i{};
    while (getline(file_stream, file_row)) {
      std::vector<W> weights;
      std::vector<W> bias;

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

      m_weights[i] = std::make_shared<Matrix<W>>(m_weights[i]->nrows(), m_weights[i]->ncols(), weights);
      m_bias[i] = std::make_shared<std::vector<W>>(bias);
    }
  }

  template <typename T,
            typename W,
            template <typename F>
            typename Activator,
            template <typename E, typename LW, template <typename K> typename Act>
            typename Loss>
  void Network<T, W, Activator, Loss>::export_network(const std::string& filepath) {
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
            typename W,
            template <typename F>
            typename Activator,
            template <typename E, typename LW, template <typename K> typename Act>
            typename Loss>
  template <typename U>
  double Network<T, W, Activator, Loss>::get_loss_value(const std::vector<U>& target) {
    return Loss<T, W, Activator>()(m_layers[n_layers - 1]->nodes(), target);
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

};  // namespace nnhep

#endif
