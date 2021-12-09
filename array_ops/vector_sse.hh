#ifndef VECTOR_SSE_HH
#define VECTOR_SSE_HH

#include <sleef.h>

#include <functional>
#include <cstddef>

#include "vector.hh"

template <>
struct Vector<SSE> {
  typedef __m128 TYPE;
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 4;

  static void erff(float *a) {
    with_load_store(Sleef_erff4_u10, a);
  }

  static void expf(float *a) {
    with_load_store(Sleef_expf4_u10, a);
  }

private:
  static void with_load_store(std::function<__m128(__m128)> f, float *a) {
    TYPE val = _mm_loadu_ps(a);
    val = f(val);
    _mm_storeu_ps(a, val);
  }
};

#endif // VECTOR_SSE_HH
