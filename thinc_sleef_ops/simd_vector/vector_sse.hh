#ifndef VECTOR_SSE_HH
#define VECTOR_SSE_HH

#include <sleef.h>

#include <cstddef>
#include <functional>

#include "vector.hh"

template <>
struct Vector<SSE> {
  typedef __m128d DOUBLE_TYPE;
  static size_t const N_DOUBLE = 2;

  typedef __m128 FLOAT_TYPE;
  static size_t const N_FLOAT = 4;

  typedef Scalar LOWER_TYPE;

  static void erf(double *a) noexcept {
    with_load_store(Sleef_erfd2_u10, a);
  }

  static void erff(float *a) noexcept {
    with_load_store(Sleef_erff4_u10, a);
  }

  static void exp(double *a) noexcept {
    with_load_store(Sleef_expd2_u10, a);
  }

  static void expf(float *a) noexcept {
    with_load_store(Sleef_expf4_u10, a);
  }

  static void tanh(double *a) noexcept {
    with_load_store(Sleef_tanhd2_u10, a);
  }

  static void tanhf(float *a) noexcept {
    with_load_store(Sleef_tanhf4_u10, a);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a) noexcept {
    FLOAT_TYPE val = _mm_loadu_ps(a);
    val = f(val);
    _mm_storeu_ps(a, val);
  }

  template <class F>
  static void with_load_store(F f, double *a) noexcept {
    DOUBLE_TYPE val = _mm_loadu_pd(a);
    val = f(val);
    _mm_storeu_pd(a, val);
  }
};

#endif // VECTOR_SSE_HH
