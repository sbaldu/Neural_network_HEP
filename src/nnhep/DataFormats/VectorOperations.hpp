/// @file VectorOperations.hpp
/// @brief File containing the definition of the vector operations

#ifndef vec_operations_h
#define vec_operations_h

#pragma once

#include <algorithm>
#include <concepts>
#include <ostream>
#include <vector>

#include "Vector.hpp"

namespace nnhep {

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
                                                    v2.deviceView(),
                                                    result.deviceView(),
                                                    n_points));
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
                                                    v2.deviceView(),
                                                    result.deviceView(),
                                                    n_points));
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

};  // namespace nnhep

#endif
