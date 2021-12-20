#ifndef VECTOR_AVX512_HH
#define VECTOR_AVX512_HH

#include <sleef.h>

#include <functional>
#include <cstddef>

#include "vector.hh"

// The following intrinsics are only defined in AVX512DQ.
// However, since these are bitwise xor, we can cast the
// FP numbers to integers first and use integer xor, which
// is available in AVX512F.
#define _mm512_xor_ps(a, b) \
  (__m512) _mm512_xor_si512(_mm512_castps_si512(a), \
                            _mm512_castps_si512(b))

#define _mm512_xor_pd(a, b) \
  (__m512d) _mm512_xor_si512(_mm512_castpd_si512(a), \
                            _mm512_castpd_si512(b))

template <>
struct Vector<AVX512> {
  typedef __m512d DOUBLE_TYPE;
  static size_t const N_DOUBLE = 8;

  typedef __m512 FLOAT_TYPE;
  static size_t const N_FLOAT = 16;

  typedef AVX LOWER_TYPE;

  static void add(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = _mm512_set1_pd(v);
      return _mm512_add_pd(a, v_simd);
    }, a);
  }

  static void add(double *a, double *v) noexcept {
    with_load_load_store(_mm512_add_pd, a, v);
  }

  static void addf(float *a, float v) noexcept {
    with_load_store([=](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = _mm512_set1_ps(v);
      return _mm512_add_ps(a, v_simd);
    }, a);
  }

  static void addf(float *a, float *v) noexcept {
    with_load_load_store(_mm512_add_ps, a, v);
  }

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

  static void mul(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = _mm512_set1_pd(v);
      return _mm512_mul_pd(a, v_simd);
    }, a);
  }

  static void mul(double *a, double *v) noexcept {
    with_load_load_store(_mm512_mul_pd, a, v);
  }

  static void mulf(float *a, float v) noexcept {
    with_load_store([v](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = _mm512_set1_ps(v);
      return _mm512_mul_ps(a, v_simd);
    }, a);
  }

  static void mulf(float *a, float *v) noexcept {
    with_load_load_store(_mm512_mul_ps, a, v);
  }

  static void neg(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      DOUBLE_TYPE minus_zero = _mm512_set1_pd(-0.0);
      return _mm512_xor_pd(v, minus_zero);
    }, a);
  }

  static void negf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      FLOAT_TYPE minus_zero = _mm512_set1_ps(-0.0);
      return _mm512_xor_ps(v, minus_zero);
    }, a);
  }

  static void recip(double *a) noexcept {
    with_load_store([](DOUBLE_TYPE v) {
      // Use division rather than reciprocal instruction, for
      // higher precision. Not sure if we care?
      DOUBLE_TYPE one = _mm512_set1_pd(1.0);
      return _mm512_div_pd(one, v);
    }, a);
  }

  static void recipf(float *a) noexcept {
    with_load_store([](FLOAT_TYPE v) {
      // Use division rather than reciprocal instruction, for
      // higher precision. Not sure if we care?
      FLOAT_TYPE one = _mm512_set1_ps(1.0);
      return _mm512_div_ps(one, v);
    }, a);
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

  template <class F>
  static void with_load_load_store(F f, double *a, double *b) noexcept {
    DOUBLE_TYPE val_a = _mm512_loadu_pd(a);
    DOUBLE_TYPE val_b = _mm512_loadu_pd(b);
    val_a = f(val_a, val_b);
    _mm512_storeu_pd(a, val_a);
  }

  template <class F>
  static void with_load_load_store(F f, float *a, float *b) noexcept {
    FLOAT_TYPE val_a = _mm512_loadu_ps(a);
    FLOAT_TYPE val_b = _mm512_loadu_ps(b);
    val_a = f(val_a, val_b);
    _mm512_storeu_ps(a, val_a);
  }
};

#endif // VECTOR_AVX512_HH
