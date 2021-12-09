#ifndef VECTOR_HH
#define VECTOR_HH

#include <functional>
#include <cmath>
#include <cstddef>

struct AVX {};
struct AVX512 {};
struct SSE {};
struct Scalar {};

template <class T>
struct Vector {
  typedef Scalar LOWER_TYPE;
};

template<>
struct Vector<Scalar> {
  typedef float TYPE;
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 1;

  static void erff(float *a) noexcept {
    *a = std::erf(*a);
  }

  static void expf(float *a) noexcept {
    *a = std::exp(*a);
  }
};

#endif // VECTOR_HH
