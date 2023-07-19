
#include <cmath>
#include <limits>

#include "include/Activators.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test the sigmoid functor") {
  Sigmoid<double> sig;

  CHECK(sig(0) == 0.5);
  CHECK(sig(1) == 1 / (1 + std::exp(-1.)));
  CHECK(sig(0.5) == 1 / (1 + std::exp(-0.5)));
  CHECK(sig(std::numeric_limits<double>::max()) == 1.);
  CHECK(sig(-std::numeric_limits<double>::max()) == 0.);
}
