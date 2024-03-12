
#include "../../../HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename T>
  struct KernelAdd {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, const T* b, T* c, size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        c[i] = a[i] + b[i];
      }
    }
  };

  template <typename T>
  struct KernelIncrement {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const T* a,
                                  T* b,
                                  size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        b[i] += a[i];
      }
    }
  };

  template <typename T>
  struct KernelSubtract {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, const T* b, T* c, size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        c[i] = a[i] - b[i];
      }
    }
  };

  template <typename T>
  struct KernelDecrement {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const T* a,
                                  T* b,
                                  size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        b[i] -= a[i];
      }
    }
  };

  template <typename T>
  struct KernelMultiply {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, T* b, T constant, size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        b[i] = a[i] * constant;
      }
    }
  };

  template <typename T>
  struct KernelMultiply_inplace {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  T* a,
                                  T constant,
                                  size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        a[i] *= constant;
      }
    }
  };

  template <typename T>
  struct KernelDivide {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, T* b, T constant, size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        b[i] = a[i] / constant;
      }
    }
  };

  template <typename T>
  struct KernelDivide_inplace {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  T* a,
                                  T constant,
                                  size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        a[i] /= constant;
      }
    }
  };

  template <typename T>
  struct KernelScalarProduct {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, const T* b, T* c, size_t n_points) const {
      for (auto i : cms::alpakatools::elements_with_stride(acc, n_points)) {
        c[i] = a[i] * b[i];
      }
    }
  };

  // C = A * B
  template <typename T>
  struct KernelMatrixMultiplication {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const T* a,
                                  const T* b,
                                  T* c,
                                  size_t a_rows,
                                  size_t b_cols,
                                  size_t n) const {
      for (auto nd_index :
           cms::alpakatools::elements_with_stride_nd(acc, Vec2D{a_rows, b_cols})) {
        for (size_t k = 0; k < n; ++k) {
          c[nd_index[0] * b_cols + nd_index[1]] +=
              a[nd_index[0] * n + k] * b[k * b_cols + nd_index[1]];
        }
      }
    }
  };

  // A = A * B
  template <typename T>
  struct KernelMatrixMultiplication_inplace {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, T* a, const T* b, size_t a_rows, size_t b_cols, size_t n) const {
      for (auto nd_index :
           cms::alpakatools::elements_with_stride_nd(acc, Vec2D{a_rows, b_cols})) {
        for (size_t k = 0; k < n; ++k) {
          a[nd_index[0] * b_cols + nd_index[1]] +=
              a[nd_index[0] * n + k] * b[k * b_cols + nd_index[1]];
        }
      }
    }
  };

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE
