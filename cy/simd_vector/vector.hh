#ifndef VECTOR_HH
#define VECTOR_HH

#include <functional>
#include <cmath>
#include <cstddef>

#include <sleef.h>

struct AVX {};
struct AVX512 {};
struct NEON {};
struct SSE {};
struct Scalar {};

template <class T>
struct Vector {
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 1;
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

  static void tanhf(float *a) noexcept {
    *a = Sleef_tanhf_u10(*a);
  }
};

#endif // VECTOR_HH
