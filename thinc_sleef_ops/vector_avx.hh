#ifndef VECTOR_AVX_HH
#define VECTOR_AVX_HH

#include <functional>
#include <cstddef>

#include "../sleef/sleef.h"
#include "vector.hh"

template <>
struct Vector<AVX> {
  typedef __m256 TYPE;
  typedef SSE LOWER_TYPE;
  static size_t const N_FLOAT = 8;

  static void erff(float *a) {
    with_load_store(Sleef_erff8_u10, a);
  }

  static void expf(float *a) {
    with_load_store(Sleef_expf8_u10, a);
  }

private:
  static void with_load_store(std::function<__m256(__m256)> f, float *a) {
    TYPE val = _mm256_loadu_ps(a);
    val = f(val);
    _mm256_storeu_ps(a, val);
  }
};

#endif // VECTOR_AVX_HH
