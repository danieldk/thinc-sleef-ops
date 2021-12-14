#ifndef VECTOR_AVX512_HH
#define VECTOR_AVX512_HH

#include <sleef.h>

#include <functional>
#include <cstddef>

#include "vector.hh"

template <>
struct Vector<AVX512> {
  typedef __m512d DOUBLE_TYPE;
  static size_t const N_DOUBLE = 8;

  typedef __m512 FLOAT_TYPE;
  static size_t const N_FLOAT = 16;

  typedef AVX LOWER_TYPE;

  static void erf(double *a) {
    with_load_store(Sleef_erfd8_u10, a);
  }

  static void erff(float *a) {
    with_load_store(Sleef_erff16_u10, a);
  }

  static void exp(double *a) {
    with_load_store(Sleef_expd8_u10, a);
  }

  static void expf(float *a) {
    with_load_store(Sleef_expf16_u10, a);
  }

  static void tanh(double *a) {
    with_load_store(Sleef_tanhd8_u10, a);
  }

  static void tanhf(float *a) {
    with_load_store(Sleef_tanhf16_u10, a);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a) {
    FLOAT_TYPE val = _mm512_loadu_ps(a);
    val = f(val);
    _mm512_storeu_ps(a, val);
  }

  template <class F>
  static void with_load_store(F f, double *a) {
    DOUBLE_TYPE val = _mm512_loadu_pd(a);
    val = f(val);
    _mm512_storeu_pd(a, val);
  }
};

#endif // VECTOR_AVX512_HH
