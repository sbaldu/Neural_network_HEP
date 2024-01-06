
#include <cmath>
#include <limits>

#include "nnhep.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test the step functor") {
  nnhep::Step<double> step;

  CHECK(step(0) == 1);
  CHECK(step(1) == 1);
  CHECK(step(-1) == 0);
  CHECK(step(std::numeric_limits<double>::max()) == 1.);
  CHECK(step(-std::numeric_limits<double>::max()) == 0.);
}

TEST_CASE("Test the sigmoid functor") {
  nnhep::Sigmoid<double> sig;

  CHECK(sig(0) == 0.5);
  CHECK(sig(1) == 1 / (1 + std::exp(-1.)));
  CHECK(sig(0.5) == 1 / (1 + std::exp(-0.5)));
  CHECK(sig(std::numeric_limits<double>::max()) == 1.);
  CHECK(sig(-std::numeric_limits<double>::max()) == 0.);
}

TEST_CASE("Test the Elu functor") {
  nnhep::Elu<double> e{0.5};

  CHECK(e(1.) == 1.);
  CHECK(e(0.) == 0.);
  CHECK(e(std::numeric_limits<double>::max()) == std::numeric_limits<double>::max());
  CHECK(e(-1) == 0.5 * (std::expm1(-1)));
  CHECK(e(-std::numeric_limits<double>::max()) ==
        0.5 * (std::expm1(-std::numeric_limits<double>::max())));
}

TEST_CASE("Test the Leaky Elu functor") {
  nnhep::Leaky_ReLU<double> le;

  CHECK(le(0.) == 0.);
  CHECK(le(1.) == 1.);
  CHECK(le(std::numeric_limits<double>::max()) == std::numeric_limits<double>::max());
  CHECK(le(-std::numeric_limits<double>::max()) ==
        -0.1 * std::numeric_limits<double>::max());
  CHECK(le(-1) == -0.1);
}
