
#ifndef vector_hpp
#define vector_hpp

#pragma once

#include <vector>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include "alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename T>
  class Vector {
  private:
    std::vector<T> m_data;
    cms::alpakatools::device_buffer<Device, T[]> m_dev;

  public:
    Vector() = delete;
    Vector(Queue queue, size_t size)
        : m_data(size), m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, size)} {}
    template <typename E>
    Vector(Queue queue, const std::vector<E>& other);

	std::size_t size() const { return m_data.size(); }

	// host and device buffers
    std::vector<T>& hostBuffer() { return m_data; }
    const std::vector<T>& hostBuffer() const { return m_data; }
    cms::alpakatools::device_buffer<Device, T[]>& deviceBuffer() { return m_dev; }
    const cms::alpakatools::device_buffer<Device, T[]>& deviceBuffer() const { return m_dev; }

	// host and device views
	T* hostView() { return m_data.data(); }
	const T* hostView() const { return m_data.data(); }
	T* deviceView() { return m_dev.data(); }
	const T* deviceView() const { return m_dev.data(); }

    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }

    void updateHost(Queue queue) {
      alpaka::memcpy(
          /* queue, cms::alpakatools::make_host_view(m_data.data(), m_data.size()), m_dev); */
          queue, m_data, m_dev);
      alpaka::wait(queue);
    }

    template <typename E>
    void add(Queue queue, const Vector<E>& other);
    template <typename E>
    void subtract(Queue queue, const Vector<E>& other);
    template <typename E>
    void multiply(Queue queue, E scalar);
    template <typename E>
    void divide(Queue queue, E scalar);

    // scalar product
    template <typename E>
    T multiply(Queue queue, const Vector<E>& other);
  };

  // not 100% sure until I test it on gpu
  template <typename T>
  template <typename E>
  Vector<T>::Vector(Queue queue, const std::vector<E>& other)
      : m_data{other},
        m_dev{cms::alpakatools::make_device_buffer<T[]>(queue, other.size())} {
    alpaka::memcpy(
        queue, m_dev, cms::alpakatools::make_host_view(other.data(), other.size()));
    alpaka::wait(queue);
  }

  template <typename T>
  template <typename E>
  void Vector<T>::add(Queue queue, const Vector<E>& other) {
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
  void Vector<T>::subtract(Queue queue, const Vector<E>& other) {
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
  void Vector<T>::multiply(Queue queue, E scalar) {
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
  void Vector<T>::divide(Queue queue, E scalar) {
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

  template <typename T>
  template <typename E>
  T Vector<T>::multiply(Queue queue, const Vector<E>& other) {
    const size_t n_points{this->m_data.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    auto temp_buffer = cms::alpakatools::make_device_buffer<T[]>(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelScalarProduct<T>(),
                                                    this->m_dev.data(),
                                                    other.m_dev.data(),
                                                    temp_buffer.data(),
                                                    n_points));
    alpaka::wait(queue);

    auto host_buffer = cms::alpakatools::make_host_buffer<T[]>(n_points);
    alpaka::memcpy(queue, host_buffer, temp_buffer);
    alpaka::wait(queue);

    return std::accumulate(host_buffer.data(), host_buffer.data() + n_points, 0.0);
  }

  // define arithmetic operations as outside functions
  //
  // NOTE: the arithmetic operations cannot be implemented with operator overloading
  // because in order to call the alpaka kernels we need to pass the queue as an
  // argument, and this is not possible with operator overloading.

  template <typename T, typename E>
  Vector<T> add(Queue queue, const Vector<T>& v1, const Vector<E>& v2) {
    const size_t n_points{v1.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Vector<T> result(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelAdd<T>(),
                                                    v1.deviceView(),
                                                    v2.deviceView(),
                                                    result.deviceView(),
                                                    n_points));
    alpaka::wait(queue);
    return result;
  }

  template <typename T, typename E>
  Vector<T> subtract(Queue queue, const Vector<T>& v1, const Vector<E>& v2) {
    const size_t n_points{v1.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Vector<T> result(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelSubtract<T>(),
                                                    v1.deviceView(),
                                                    v2.deviceView(), result.deviceView(), n_points));
    alpaka::wait(queue);

	return result;
  }

  template <typename T, typename E>
  Vector<T> multiply(Queue queue, const Vector<T>& vec, E scalar) {
    const size_t n_points{vec.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Vector<T> result(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelMultiply<T>(),
                                                    vec.deviceView(),
                                                    result.deviceView(),
                                                    scalar,
                                                    n_points));
    alpaka::wait(queue);

	return result;
  }

  template <typename T, typename E>
  Vector<T> divide(Queue queue, const Vector<T>& vec, E scalar) {
    const size_t n_points{vec.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Vector<T> result(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelDivide<T>(),
                                                    vec.deviceView(),
                                                    result.deviceView(),
                                                    scalar,
                                                    n_points));
    alpaka::wait(queue);

	return result;
  }

  template <typename T, typename E>
  T multiply(Queue queue, const Vector<T>& v1, const Vector<E>& v2) {
    const size_t n_points{v1.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    auto temp_buffer = cms::alpakatools::make_device_buffer<T[]>(queue, n_points);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelScalarProduct<T>(),
                                                    v1.deviceView(),
                                                    v2.deviceView(),
                                                    temp_buffer.data(),
                                                    n_points));
    alpaka::wait(queue);

    auto host_buffer = cms::alpakatools::make_host_buffer<T[]>(n_points);
    alpaka::memcpy(queue, host_buffer, temp_buffer);
    alpaka::wait(queue);

    return std::accumulate(host_buffer.data(), host_buffer.data() + n_points, 0.0);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
