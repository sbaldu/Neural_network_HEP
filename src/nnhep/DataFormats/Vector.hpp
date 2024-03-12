
#ifndef vector_hpp
#define vector_hpp

#pragma once

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include <vector>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include "alpaka/Kernels.h"

namespace nnhep {

template <typename T, typename TDev> class Vector {
private:
  std::vector<T> m_data;
  cms::alpakatools::device_buffer<TDev, T[]> m_dev;

public:
  Vector() = delete;
  template <typename TQueue>
  Vector(TQueue queue, size_t size)
      : m_data(size),
        m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, size)} {}
  template <typename TQueue, typename E>
  Vector(TQueue queue, const std::vector<E> &other);

  std::size_t size() const { return m_data.size(); }

  // host and device buffers
  std::vector<T> &hostBuffer() { return m_data; }
  const std::vector<T> &hostBuffer() const { return m_data; }
  cms::alpakatools::device_buffer<TDev, T[]> &deviceBuffer() { return m_dev; }
  const cms::alpakatools::device_buffer<TDev, T[]> &deviceBuffer() const {
    return m_dev;
  }

  // host and device views
  T *hostView() { return m_data.data(); }
  const T *hostView() const { return m_data.data(); }
  T *deviceView() { return m_dev.data(); }
  const T *deviceView() const { return m_dev.data(); }

  T &operator[](size_t i) { return m_data[i]; }
  const T &operator[](size_t i) const { return m_data[i]; }

  template <typename TQueue> void updateHost(TQueue queue) {
    alpaka::memcpy(
        /* queue, cms::alpakatools::make_host_view(m_data.data(),
           m_data.size()), m_dev); */
        queue,
        m_data, m_dev);
    alpaka::wait(queue);
  }
  template <typename TQueue> void updateDevice(TQueue queue) {
    alpaka::memcpy(queue, m_dev, m_data.data());
    alpaka::wait(queue);
  }

  template <typename TQueue, typename E>
  void add(TQueue queue, const Vector<E> &other);
  template <typename TQueue, typename E>
  void subtract(TQueue queue, const Vector<E> &other);
  template <typename TQueue, typename E> void multiply(TQueue queue, E scalar);
  template <typename TQueue, typename E> void divide(TQueue queue, E scalar);

  // scalar product
  template <typename TQueue, typename E>
  T multiply(TQueue queue, const Vector<E> &other);
};

// not 100% sure until I test it on gpu
template <typename T>
template <typename E>
Vector<T>::Vector(Queue queue, const std::vector<E> &other)
    : m_data{other},
      m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, other.size())} {
  alpaka::memcpy(queue, m_dev,
                 cms::alpakatools::make_host_view(other.data(), other.size()));
  alpaka::wait(queue);
}

template <typename T>
template <typename E>
void Vector<T>::add(Queue queue, const Vector<E> &other) {
  const size_t n_points{m_data.size()};
  const Idx block_size{256};
  const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
  auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(
                             work_div, KernelIncrement<T>(), other.m_dev.data(),
                             this->m_dev.data(), n_points));
  alpaka::wait(queue);
}

template <typename T>
template <typename E>
void Vector<T>::subtract(Queue queue, const Vector<E> &other) {
  const size_t n_points{m_data.size()};
  const Idx block_size{256};
  const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
  auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(
                             work_div, KernelDecrement<T>(), other.m_dev.data(),
                             this->m_dev.data(), n_points));
  alpaka::wait(queue);
}

template <typename T>
template <typename E>
void Vector<T>::multiply(Queue queue, E scalar) {
  const size_t n_points{this->m_data.size()};
  const Idx block_size{256};
  const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
  auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(
                             work_div, KernelMultiply_inplace<T>(),
                             this->m_dev.data(), scalar, n_points));
  alpaka::wait(queue);
}

template <typename T>
template <typename E>
void Vector<T>::divide(Queue queue, E scalar) {
  const size_t n_points{this->m_data.size()};
  const Idx block_size{256};
  const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
  auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(
                             work_div, KernelDivide_inplace<T>(),
                             this->m_dev.data(), scalar, n_points));
  alpaka::wait(queue);
}

template <typename T>
template <typename E>
T Vector<T>::multiply(Queue queue, const Vector<E> &other) {
  const size_t n_points{this->m_data.size()};
  const Idx block_size{256};
  const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
  auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

  auto temp_buffer = cms::alpakatools::make_device_buffer<T[]>(queue, n_points);
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(
                      work_div, KernelScalarProduct<T>(), this->m_dev.data(),
                      other.m_dev.data(), temp_buffer.data(), n_points));
  alpaka::wait(queue);

  auto host_buffer = cms::alpakatools::make_host_buffer<T[]>(n_points);
  alpaka::memcpy(queue, host_buffer, temp_buffer);
  alpaka::wait(queue);

  return std::accumulate(host_buffer.data(), host_buffer.data() + n_points,
                         0.0);
}

}; // namespace nnhep

#endif
