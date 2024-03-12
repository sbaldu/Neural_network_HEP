/// @file Matrix.hpp
/// @brief This file contains the declaration of the Matrix class.
/// @author Simone Balducci
///
///
/// @details The Matrix class is a class that represents a matrix of any size.

#ifndef Matrix_hpp
#define Matrix_hpp

#pragma once

#include <algorithm>
#include <concepts>
#include <iostream>
#include <iterator>
#include <vector>

#include "../../HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "../../HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "../../HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include "alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename T>
  class Matrix {
  private:
    int m_nrows;
    int m_ncols;
    int m_size;
    std::vector<T> m_data;
    cms::alpakatools::device_buffer<Device, T[]> m_dev;

  public:
    Matrix() = delete;
    Matrix(Queue queue, int n_rows, int n_cols);
    template <typename E>
    Matrix(Queue queue, int n_rows, int n_cols, std::vector<E> vec);
    template <typename E>
    Matrix(Queue queue, std::vector<E> vec);

    // host and device buffers
    std::vector<T>& hostBuffer() { return m_data; }
    const std::vector<T>& hostBuffer() const { return m_data; }
    cms::alpakatools::device_buffer<Device, T[]>& deviceBuffer() { return m_dev; }
    const cms::alpakatools::device_buffer<Device, T[]>& deviceBuffer() const {
      return m_dev;
    }

    // host and device views
    T* hostView() { return m_data.data(); }
    const T* hostView() const { return m_data.data(); }
    T* deviceView() { return m_dev.data(); }
    const T* deviceView() const { return m_dev.data(); }

    void updateHost(Queue queue) {
      alpaka::memcpy(queue, m_data, m_dev);
      alpaka::wait(queue);
    }

    inline int rows() const { return m_nrows; }
    inline int cols() const { return m_ncols; }
    inline std::size_t size() const { return m_size; }

    inline void set_data(int i, int j, T data);
    inline void set_data(int index, T data);
    inline void set_data(std::vector<T> data_vec);

    inline T& operator()(int i, int j) { return m_data[j + m_ncols * i]; }
    inline const T& operator()(int i, int j) const { return m_data[j + m_ncols * i]; }

    T& operator[](int index) { return m_data[index]; }
    const T& operator[](int index) const { return m_data[index]; }

    inline Matrix transpose();

    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, const Matrix<U>& m);

    template <typename E>
    void add(Queue queue, const Matrix<E>& other);
    template <typename E>
    void subtract(Queue queue, const Matrix<E>& other);
    template <typename E>
    void multiply(Queue queue, const Matrix<E>& other);
    template <typename E>
    void multiply(Queue queue, E scalar);
    template <typename E>
    void divide(Queue queue, E scalar);
  };

  template <typename T>
  Matrix<T>::Matrix(Queue queue, int n_rows, int n_cols)
      : m_nrows{n_rows},
        m_ncols{n_cols},
        m_size{n_rows * n_cols},
        m_data(m_size),
        m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, m_size)} {}

  template <typename T>
  template <typename E>
  Matrix<T>::Matrix(Queue queue, int n_rows, int n_cols, std::vector<E> vec)
      : m_nrows{n_rows},
        m_ncols{n_cols},
        m_size{n_rows * n_cols},
        m_data{std::move(vec)},
        m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, m_size)} {
    alpaka::memcpy(
        queue, m_dev, cms::alpakatools::make_host_view(m_data.data(), m_data.size()));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  Matrix<T>::Matrix(Queue queue, std::vector<E> vec)
      : m_nrows{vec.size()},
        m_ncols{1},
        m_size{m_nrows},
        m_data{std::move(vec)},
        m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, m_size)} {
    alpaka::memcpy(queue, m_dev, m_data);
    alpaka::wait(queue);
  }

  template <typename T>
  void Matrix<T>::set_data(int i, int j, T data) {
    int index{j + m_ncols * i};
    try {
      if (index >= m_ncols * m_nrows) {
        throw(index);
      }
      m_data[index] = data;
    } catch (...) {
      std::cout << "The index " << index << " is larger that the size of the matrix\n";
    }
  }

  template <typename T>
  void Matrix<T>::set_data(int index, T data) {
    m_data[index] = data;
    try {
      if (index >= m_ncols * m_nrows) {
        throw(index);
      }
      m_data[index] = data;
    } catch (...) {
      std::cout << "The index " << index << " is larger that the size of the matrix\n";
    }
  }

  template <typename T>
  void Matrix<T>::set_data(std::vector<T> data_vec) {
    m_data = std::move(data_vec);
  }

  template <typename T>
  Matrix<T> Matrix<T>::transpose() {
    Matrix<T> matrix(this->m_ncols, this->m_nrows);

    for (int i{}; i < this->m_nrows; ++i) {
      for (int j{}; j < this->m_ncols; ++j) {
        matrix.set_data(j, i, (*this)(i, j));
      }
    }

    return matrix;
  }

  template <typename U>
  std::ostream& operator<<(std::ostream& out, const Matrix<U>& m) {
    out << m.m_data[0];
    std::for_each(m.m_data.begin() + 1, m.m_data.end(), [&out](U x) {
      out << ',';
      out << x;
    });

    return out;
  }

  template <typename T>
  template <typename E>
  void Matrix<T>::add(Queue queue, const Matrix<E>& other) {
    const size_t n_points{m_data.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelIncrement<T>(),
                                                    other.m_dev.data(),
                                                    this->m_dev.data(),
                                                    n_points));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  void Matrix<T>::subtract(Queue queue, const Matrix<E>& other) {
    const size_t n_points{m_data.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelDecrement<T>(),
                                                    other.m_dev.data(),
                                                    this->m_dev.data(),
                                                    n_points));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  void Matrix<T>::multiply(Queue queue, const Matrix<E>& other) {
    const Idx block_size{32};
    const Idx grid_size = cms::alpakatools::divide_up_by(m_ncols, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc2D>({grid_size, grid_size},
                                                          {block_size, block_size});

    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc2D>(work_div,
                                        KernelMatrixMultiplication_inplace<T>(),
                                        this->m_dev.data(),
                                        other.m_dev.data(),
                                        this->m_nrows,
                                        other.m_ncols,
                                        this->m_ncols));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  void Matrix<T>::multiply(Queue queue, E scalar) {
    const size_t n_points{this->m_data.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            work_div, KernelMultiply_inplace<T>(), this->m_dev.data(), scalar, n_points));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  void Matrix<T>::divide(Queue queue, E scalar) {
    const size_t n_points{this->m_data.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            work_div, KernelDivide_inplace<T>(), this->m_dev.data(), scalar, n_points));
    alpaka::wait(queue);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
