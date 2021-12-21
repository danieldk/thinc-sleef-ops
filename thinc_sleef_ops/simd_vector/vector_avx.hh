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

  static DOUBLE_TYPE add(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm256_add_pd(a, b);
  }

  static DOUBLE_TYPE add_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm256_set1_pd(b);
    return _mm256_add_pd(a, b_simd);
  }

  static FLOAT_TYPE addf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm256_add_ps(a, b);
  }

  static FLOAT_TYPE addf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm256_set1_ps(b);
    return _mm256_add_ps(a, b_simd);
  }

  static DOUBLE_TYPE div(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm256_div_pd(a, b);
  }

  static FLOAT_TYPE divf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm256_div_ps(a, b);
  }

  static DOUBLE_TYPE erf(DOUBLE_TYPE a) {
    return Sleef_erfd4_u10(a);
  }

  static FLOAT_TYPE erff(FLOAT_TYPE a) {
    return Sleef_erff8_u10(a);
  }

  static DOUBLE_TYPE exp(DOUBLE_TYPE a) {
    return Sleef_expd4_u10(a);
  }

  static FLOAT_TYPE expf(FLOAT_TYPE a) {
    return Sleef_expf8_u10(a);
  }

  static DOUBLE_TYPE logistic_cdf(DOUBLE_TYPE a) {
    return generic_logistic_cdf<AVX>(a);
  }

  static FLOAT_TYPE logistic_cdff(FLOAT_TYPE a) {
    return generic_logistic_cdff<AVX>(a);
  }

  static DOUBLE_TYPE logistic_pdf(DOUBLE_TYPE a) {
    return generic_logistic_pdf<AVX>(a);
  }

  static FLOAT_TYPE logistic_pdff(FLOAT_TYPE a) {
    return generic_logistic_pdff<AVX>(a);
  }

  static DOUBLE_TYPE mul(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm256_mul_pd(a, b);
  }

  static DOUBLE_TYPE mul_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm256_set1_pd(b);
    return _mm256_mul_pd(a, b_simd);
  }

  static FLOAT_TYPE mulf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm256_mul_ps(a, b);
  }

  static FLOAT_TYPE mulf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm256_set1_ps(b);
    return _mm256_mul_ps(a, b_simd);
  }

  static DOUBLE_TYPE neg(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE minus_zero = _mm256_set1_pd(-0.0);
    return _mm256_xor_pd(a, minus_zero);
  }

  static FLOAT_TYPE negf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE minus_zero = _mm256_set1_ps(-0.0);
    return _mm256_xor_ps(a, minus_zero);
  }

  static DOUBLE_TYPE normal_cdf(DOUBLE_TYPE a) {
    return generic_normal_cdf<AVX>(a);
  }

  static FLOAT_TYPE normal_cdff(FLOAT_TYPE a) {
    return generic_normal_cdff<AVX>(a);
  }

  static DOUBLE_TYPE normal_pdf(DOUBLE_TYPE a) {
    return generic_normal_pdf<AVX>(a);
  }

  static FLOAT_TYPE normal_pdff(FLOAT_TYPE a) {
    return generic_normal_pdff<AVX>(a);
  }

  static DOUBLE_TYPE recip(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE one = _mm256_set1_pd(1.0);
    return _mm256_div_pd(one, a);
  }

  static FLOAT_TYPE recipf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE one = _mm256_set1_ps(1.0);
    return _mm256_div_ps(one, a);
  }

  static DOUBLE_TYPE tanh(DOUBLE_TYPE a) {
    return Sleef_tanhd4_u10(a);
  }

  static FLOAT_TYPE tanhf(FLOAT_TYPE a) {
    return Sleef_tanhf8_u10(a);
  }

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
