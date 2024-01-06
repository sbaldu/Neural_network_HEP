
#include <numeric>
#include <vector>

#include "Vector.hpp"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  std::vector<float> v(10);
  std::iota(v.begin(), v.end(), 0.0f);

  const auto device = alpaka::getDevByIdx<Acc1D>(0u);
  Queue queue{device};
  Vector<float> a(queue, v);
  Vector<float> b(queue, v);
  a.add(queue, b);
  a.updateHost(queue);
  for (auto x : a.hostBuffer()) {
    std::cout << x << std::endl;
  }

  std::cout << '\n';

  a.subtract(queue, b);
  a.updateHost(queue);
  for (auto x : a.hostBuffer()) {
    std::cout << x << std::endl;
  }

  std::cout << '\n';

  a.multiply(queue, 10.0f);
  a.updateHost(queue);
  for (auto x : a.hostBuffer()) {
    std::cout << x << std::endl;
  }

  std::cout << '\n';

  a.divide(queue, 4.0f);
  a.updateHost(queue);
  for (auto x : a.hostBuffer()) {
    std::cout << x << std::endl;
  }

  std::cout << '\n';

  std::cout << a.multiply(queue, b) << std::endl;

  Vector<float> c = add(queue, a, b);
  c.updateHost(queue);
  Vector<float> d = subtract(queue, a, b);
  d.updateHost(queue);

  std::cout << '\n';

  for (auto x : c.hostBuffer()) {
    std::cout << x << std::endl;
  }
  std::cout << '\n';
  for (auto x : d.hostBuffer()) {
    std::cout << x << std::endl;
  }

  std::cout << '\n';

  Vector<float> e = multiply(queue, a, 2.0f);
  e.updateHost(queue);
  Vector<float> f = divide(queue, a, 5.0f);
  f.updateHost(queue);

  for (auto x : e.hostBuffer()) {
    std::cout << x << std::endl;
  }
  std::cout << '\n';
  for (auto x : f.hostBuffer()) {
    std::cout << x << std::endl;
  }
}
