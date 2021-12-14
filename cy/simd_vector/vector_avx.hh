#ifndef VECTOR_AVX_HH
#define VECTOR_AVX_HH

#include <sleef.h>

#include <functional>
#include <cstddef>

#include "vector.hh"

template <>
struct Vector<AVX> {
  typedef __m256d DOUBLE_TYPE;
  static size_t const N_DOUBLE = 4;

  typedef __m256 FLOAT_TYPE;
  static size_t const N_FLOAT = 8;

  typedef SSE LOWER_TYPE;

  static void erf(double *a) {
    with_load_store(Sleef_erfd4_u10, a);
  }

  static void erff(float *a) {
    with_load_store(Sleef_erff8_u10, a);
  }

  static void expf(float *a) {
    with_load_store(Sleef_expf8_u10, a);
  }

  static void exp(double *a) {
    with_load_store(Sleef_expd4_u10, a);
  }

  static void tanh(double *a) {
    with_load_store(Sleef_tanhd4_u10, a);
  }

  static void tanhf(float *a) {
    with_load_store(Sleef_tanhf8_u10, a);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a) {
    FLOAT_TYPE val = _mm256_loadu_ps(a);
    val = f(val);
    _mm256_storeu_ps(a, val);
  }

  template <class F>
  static void with_load_store(F f, double *a) {
    DOUBLE_TYPE val = _mm256_loadu_pd(a);
    val = f(val);
    _mm256_storeu_pd(a, val);
  }
};

#endif // VECTOR_AVX_HH
