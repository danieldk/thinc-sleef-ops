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

  static void add(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = _mm_set1_pd(v);
      return _mm_add_pd(a, v_simd);
    }, a);
  }

  static void add(double *a, double *v) noexcept {
    with_load_load_store(_mm_add_pd, a, v);
  }

  static void addf(float *a, float v) noexcept {
    with_load_store([=](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = _mm_set1_ps(v);
      return _mm_add_ps(a, v_simd);
    }, a);
  }

  static void addf(float *a, float *v) noexcept {
    with_load_load_store(_mm_add_ps, a, v);
  }

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

  static void mul(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = _mm_set1_pd(v);
      return _mm_mul_pd(a, v_simd);
    }, a);
  }

  static void mul(double *a, double *v) noexcept {
    with_load_load_store(_mm_mul_pd, a, v);
  }

  static void mulf(float *a, float v) noexcept {
    with_load_store([v](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = _mm_set1_ps(v);
      return _mm_mul_ps(a, v_simd);
    }, a);
  }

  static void mulf(float *a, float *v) noexcept {
    with_load_load_store(_mm_mul_ps, a, v);
  }

  static void neg(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      DOUBLE_TYPE minus_zero = _mm_set1_pd(-0.0);
      return _mm_xor_pd(v, minus_zero);
    }, a);
  }

  static void negf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      FLOAT_TYPE minus_zero = _mm_set1_ps(-0.0);
      return _mm_xor_ps(v, minus_zero);
    }, a);
  }

  static void recip(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      DOUBLE_TYPE one = _mm_set1_pd(1.0);
      return _mm_div_pd(one, v);
    }, a);
  }

  static void recipf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      FLOAT_TYPE one = _mm_set1_ps(1.0);
      return _mm_div_ps(one, v);
    }, a);
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

  template <class F>
  static void with_load_load_store(F f, double *a, double *b) noexcept {
    DOUBLE_TYPE val_a = _mm_loadu_pd(a);
    DOUBLE_TYPE val_b = _mm_loadu_pd(b);
    val_a = f(val_a, val_b);
    _mm_storeu_pd(a, val_a);
  }

  template <class F>
  static void with_load_load_store(F f, float *a, float *b) noexcept {
    FLOAT_TYPE val_a = _mm_loadu_ps(a);
    FLOAT_TYPE val_b = _mm_loadu_ps(b);
    val_a = f(val_a, val_b);
    _mm_storeu_ps(a, val_a);
  }
};

#endif // VECTOR_SSE_HH
