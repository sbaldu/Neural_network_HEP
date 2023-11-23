/// @file Layer.hpp
/// @brief Layer of a neural network
///
/// @details A layer of a neural network is a collection of nodes. The layer
/// contains the values of the nodes and the weights and biases of the nodes.

#ifndef Layer_h
#define Layer_h

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "DataFormats/Matrix.hpp"

/// @brief Layer of a neural network
/// @tparam T The type of the node values
template <typename T>
class Layer {
private:
  std::vector<T> m_nodes;
  int n_nodes;

public:
  /// @brief Default constructor
  Layer() = default;
  /// @brief Constructor
  /// @param n_nodes The number of nodes in the layer
  explicit Layer(int n_nodes);
  /// @brief Constructor
  /// @param nodes The values of the nodes in the layer
  explicit Layer(std::vector<T> nodes);
  /// @brief Constructor
  /// @param stream The stream to read the node values from
  explicit Layer(std::stringstream& stream);

  /// @brief Load the node values from a stream
  /// @param stream The stream to read the node values from
  /// @details The node values are expected to be provided in csv format
  void load(std::stringstream& stream);

  /// @brief Get the node values
  /// @return The node values
  const std::vector<T>& nodes() const;
  /// @brief Get the number of nodes in the layer
  /// @return The number of nodes in the layer
  int size() const;

  /// @brief Set the values of the nodes
  /// @param values The values of the nodes
  void set_node_data(int i, T value);
  /// @brief Set the values of the nodes
  /// @param values The values of the nodes
  void set_node_data(std::vector<T> values);

  /// @brief Get the value of a node
  /// @param i The index of the node
  /// @return The value of the node
  T& operator[](int i);
  /// @brief Get the value of a node
  /// @param i The index of the node
  /// @return The value of the node
  const T& operator[](int i) const;

  /// @brief Print the layer
  /// @param out The stream to print to
  /// @param layer The layer to print
  /// @return The stream
  template <typename E>
  friend std::ostream& operator<<(std::ostream& out, const Layer<E>& layer);
};

template <typename T>
Layer<T>::Layer(int n_nodes) : m_nodes(n_nodes), n_nodes{n_nodes} {}

template <typename T>
Layer<T>::Layer(std::vector<T> nodes) : m_nodes{std::move(nodes)}, n_nodes{m_nodes.size()} {}

template <typename T>
Layer<T>::Layer(std::stringstream& stream) {
  // I assume that the data is provided in csv format
  std::string value;
  int node_index{};
  while (getline(stream, value, ',')) {
    double numeric_value{std::stod(value)};
    m_nodes.push_back(static_cast<T>(numeric_value));
  }

  n_nodes = node_index;
}

template <typename T>
void Layer<T>::load(std::stringstream& stream) {
  // I assume that the data is provided in csv format
  std::string value;
  int node_index{};
  while (getline(stream, value, ',')) {
    double numeric_value{std::stod(value)};
    m_nodes[node_index] = static_cast<T>(numeric_value);

    try {
      if (node_index + 1 > n_nodes) {
        throw(node_index);
      }
      ++node_index;
    } catch (int num) {
      std::cout << "The data provided exceedes the number of nodes expected for the layer\n";
    }
  }
}

template <typename T>
void Layer<T>::set_node_data(int i, T value) {
  try {
    if (i >= n_nodes) {
      throw(i);
    }
    m_nodes[i] = value;
  } catch (...) {
    std::cout << "The index " << i << " is larger than the number of nodes in the layer\n";
  }
}

template <typename T>
const std::vector<T>& Layer<T>::nodes() const {
  return m_nodes;
}

template <typename T>
int Layer<T>::size() const {
  return m_nodes.size();
}

template <typename T>
void Layer<T>::set_node_data(std::vector<T> values) {
  m_nodes = std::move(values);
}

template <typename T>
T& Layer<T>::operator[](int i) {
  return m_nodes[i];
}

template <typename T>
const T& Layer<T>::operator[](int i) const {
  return m_nodes[i];
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Layer<T>& layer) {
  for (auto node : layer.m_nodes) {
    out << node << ',';
  }

  return out;
}

#endif
