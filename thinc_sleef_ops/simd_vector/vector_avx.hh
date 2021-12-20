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

  static void add(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = _mm256_set1_pd(v);
      return _mm256_add_pd(a, v_simd);
    }, a);
  }

  static void addf(float *a, float v) noexcept {
    with_load_store([=](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = _mm256_set1_ps(v);
      return _mm256_add_ps(a, v_simd);
    }, a);
  }

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

  static void neg(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      DOUBLE_TYPE minus_zero = _mm256_set1_pd(-0.0);
      return _mm256_xor_pd(v, minus_zero);
    }, a);
  }

  static void negf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      FLOAT_TYPE minus_zero = _mm256_set1_ps(-0.0);
      return _mm256_xor_ps(v, minus_zero);
    }, a);
  }

  static void recip(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      DOUBLE_TYPE one = _mm256_set1_pd(1.0);
      return _mm256_div_pd(one, v);
    }, a);
  }

  static void recipf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      FLOAT_TYPE one = _mm256_set1_ps(1.0);
      return _mm256_div_ps(one, v);
    }, a);
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
