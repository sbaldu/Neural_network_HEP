
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
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, const T* a, T* b, size_t n_points) const {
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

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE
