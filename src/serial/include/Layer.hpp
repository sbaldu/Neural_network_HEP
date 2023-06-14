
#ifndef Layer_h
#define Layer_h

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "../DataFormats/Matrix.hpp"

template <typename T>
class Layer{
  private:
	std::vector<T> m_nodes;
	int n_nodes;

  public:
	Layer() = default;
	Layer(int n_nodes);
	Layer(std::vector<T> nodes);
	Layer(std::stringstream& stream);

	void load(std::stringstream& stream);

	inline void normalize();

	// Getters
	const std::vector<T>& nodes() const;
	const int size() const;

	// Setters for the node data
	void set_node_data(int i, T value);
	void set_node_data(std::vector<T> values);

	// Maybe overload operator[]
	T& operator[](int i);
	const T& operator[](int i) const;

	// Overload ostream operator for output layer
	template <typename E>
	friend std::ostream& operator<<(std::ostream& out, const Layer<E>& layer);
};

template <typename T>
Layer<T>::Layer(int n_nodes) : m_nodes(n_nodes), n_nodes{n_nodes} {}

template <typename T>
Layer<T>::Layer(std::vector<T> nodes)
  : m_nodes{std::move(nodes)}, n_nodes{m_nodes.size()} {}

template <typename T>
Layer<T>::Layer(std::stringstream& stream) {
  // I assume that the data is provided in csv format
  std::string value;
  int node_index{};
  while (getline(stream, value, ',')) {
	double numeric_value{ std::stod(value) };
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
	double numeric_value{ std::stod(value) };
	m_nodes[node_index] = static_cast<T>(numeric_value);

	try {
	  if (node_index + 1 >= n_nodes) {
		throw (node_index);
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
	  throw (i);
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
const int Layer<T>::size() const {
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

template <typename T>
inline void Layer<T>::normalize() {
  T min{*std::min_element(m_nodes.begin(), m_nodes.end())};
  T max{*std::max_element(m_nodes.begin(), m_nodes.end())};

  for (auto& x : m_nodes) {
	x /= (max - min);
  }
}

#endif
