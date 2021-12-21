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

  static DOUBLE_TYPE add(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm_add_pd(a, b);
  }

  static DOUBLE_TYPE add_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm_set1_pd(b);
    return _mm_add_pd(a, b_simd);
  }

  static FLOAT_TYPE addf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm_add_ps(a, b);
  }

  static FLOAT_TYPE addf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm_set1_ps(b);
    return _mm_add_ps(a, b_simd);
  }

  static DOUBLE_TYPE cdf(DOUBLE_TYPE a) {
    return generic_cdf<SSE>(a);
  }

  static FLOAT_TYPE cdff(FLOAT_TYPE a) {
    return generic_cdff<SSE>(a);
  }

  static DOUBLE_TYPE erf(DOUBLE_TYPE a) noexcept {
    return Sleef_erfd2_u10(a);
  }

  static FLOAT_TYPE erff(FLOAT_TYPE a) noexcept {
    return Sleef_erff4_u10(a);
  }

  static DOUBLE_TYPE exp(DOUBLE_TYPE a) noexcept {
    return Sleef_expd2_u10(a);
  }

  static FLOAT_TYPE expf(FLOAT_TYPE a) noexcept {
    return Sleef_expf4_u10(a);
  }

  static DOUBLE_TYPE mul(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm_mul_pd(a, b);
  }

  static DOUBLE_TYPE mul_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm_set1_pd(b);
    return _mm_mul_pd(a, b_simd);
  }

  static FLOAT_TYPE mulf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm_mul_ps(a, b);
  }

  static FLOAT_TYPE mulf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm_set1_ps(b);
    return _mm_mul_ps(a, b_simd);
  }

  static DOUBLE_TYPE neg(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE minus_zero = _mm_set1_pd(-0.0);
    return _mm_xor_pd(a, minus_zero);
  }

  static FLOAT_TYPE negf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE minus_zero = _mm_set1_ps(-0.0);
    return _mm_xor_ps(a, minus_zero);
  }

  static DOUBLE_TYPE pdf(DOUBLE_TYPE a) {
    return generic_pdf<SSE>(a);
  }

  static FLOAT_TYPE pdff(FLOAT_TYPE a) {
    return generic_pdff<SSE>(a);
  }

  static DOUBLE_TYPE recip(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE one = _mm_set1_pd(1.0);
    return _mm_div_pd(one, a);
  }

  static FLOAT_TYPE recipf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE one = _mm_set1_ps(1.0);
    return _mm_div_ps(one, a);
  }

  static DOUBLE_TYPE tanh(DOUBLE_TYPE a) noexcept {
    return Sleef_tanhd2_u10(a);
  }

  static FLOAT_TYPE tanhf(FLOAT_TYPE a) noexcept {
    return Sleef_tanhf4_u10(a);
  }

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
