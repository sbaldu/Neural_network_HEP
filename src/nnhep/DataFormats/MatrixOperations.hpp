
#ifndef matrix_operations_hpp
#define matrix_operations_hpp

#include "Matrix.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename T, typename E>
  Matrix<T> add(Queue queue, const Matrix<T>& m1, const Matrix<E>& m2) {
    const size_t n_points{m1.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Matrix<T> result(queue, m1.rows(), m1.cols());
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelAdd<T>(),
                                                    m1.deviceView(),
                                                    m2.deviceView(),
                                                    result.deviceView(),
                                                    n_points));
    alpaka::wait(queue);

    result.updateHost(queue);
    return result;
  }

  template <typename T, typename E>
  Matrix<T> subtract(Queue queue, const Matrix<T>& m1, const Matrix<E>& m2) {
    const size_t n_points{m1.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Matrix<T> result(queue, m1.rows(), m1.cols());
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelSubtract<T>(),
                                                    m1.deviceView(),
                                                    m2.deviceView(),
                                                    result.deviceView(),
                                                    n_points));
    alpaka::wait(queue);

    result.updateHost(queue);
    return result;
  }

  template <typename T, typename E>
  Matrix<T> multiply(Queue queue, const Matrix<T>& m1, const Matrix<E>& m2) {
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(m1.cols(), block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc2D>({grid_size, grid_size},
                                                          {block_size, block_size});

    Matrix<T> result(queue, m1.rows(), m2.cols());
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(work_div,
                                                    KernelMatrixMultiplication<T>(),
                                                    m1.deviceView(),
                                                    m2.deviceView(),
                                                    result.deviceView(),
                                                    m1.rows(),
                                                    m2.cols(),
                                                    m1.cols()));
    alpaka::wait(queue);

    result.updateHost(queue);
    return result;
  }

  template <typename T, typename E>
  Matrix<T> multiply(Queue queue, const Matrix<T>& mat, E scalar) {
    const size_t n_points{mat.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Matrix<T> result(queue, mat.rows(), mat.cols());
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelMultiply<T>(),
                                                    mat.deviceView(),
                                                    result.deviceView(),
                                                    scalar,
                                                    n_points));
    alpaka::wait(queue);

    result.updateHost(queue);
    return result;
  }

  template <typename T, typename E>
  Matrix<T> divide(Queue queue, const Matrix<T>& mat, E scalar) {
    const size_t n_points{mat.size()};
    const Idx block_size{256};
    const Idx grid_size = cms::alpakatools::divide_up_by(n_points, block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);

    Matrix<T> result(queue, mat.rows(), mat.cols());
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(work_div,
                                                    KernelDivide<T>(),
                                                    mat.deviceView(),
                                                    result.deviceView(),
                                                    scalar,
                                                    n_points));
    alpaka::wait(queue);

    result.updateHost(queue);
    return result;
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // matrix_operations_hpp
