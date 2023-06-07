
#ifndef Activators_h
#define Activators_h

#pragma once

#include <algorithm>
#include <cmath>

struct Step {
  short operator()(double x) {
    if (x < 0) {
      return 0;
    } else {
      return 1;
    }
  }
};

struct Linear {
  double operator()(double x) { return x; }
};

struct Sigmoid {
  double operator()(double x) { return 1. / (1 + std::exp(-x)); }
};

struct Tanh {
  double operator()(double x) { return std::tanh(x); }
};

struct Elu {
  double A;

  double operator()(double x) {
    if (x >= 0) {
      return x;
    } else {
      return A * (std::exp(x) - 1);
    }
  }
};

struct Leaky_Elu {
  double operator()(double x) { return std::max(0.1 * x, x); }
};

struct Parametric_Elu {
  double A;

  double operator()(double x) { return std::max(A * x, x); }
};

struct Swish {
  double operator()(double x) {
    Sigmoid sig;
    return x * sig(x);
  }
};

#endif
