#ifndef VECTOR_HH
#define VECTOR_HH

#include <functional>
#include <cmath>
#include <cstddef>

struct AVX {};
struct SSE {};
struct Scalar {};

template <class T>
class Vector {};

template<>
struct Vector<Scalar> {
  typedef float TYPE;
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 1;

  static void erff(float *a) {
    *a = std::erf(*a);
  }

  static void expf(float *a) {
    *a = std::exp(*a);
  }
};

#endif // VECTOR_HH
