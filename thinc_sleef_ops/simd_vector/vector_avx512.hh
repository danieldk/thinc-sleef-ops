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

  static DOUBLE_TYPE add(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm512_add_pd(a, b);
  }

  static DOUBLE_TYPE add_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm512_set1_pd(b);
    return _mm512_add_pd(a, b_simd);
  }

  static FLOAT_TYPE addf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm512_add_ps(a, b);
  }

  static FLOAT_TYPE addf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm512_set1_ps(b);
    return _mm512_add_ps(a, b_simd);
  }

  static DOUBLE_TYPE cdf(DOUBLE_TYPE a) {
    return generic_cdf<AVX512>(a);
  }

  static FLOAT_TYPE cdff(FLOAT_TYPE a) {
    return generic_cdff<AVX512>(a);
  }

  static DOUBLE_TYPE erf(DOUBLE_TYPE a) {
    return Sleef_erfd8_u10(a);
  }

  static FLOAT_TYPE erff(FLOAT_TYPE a) {
    return Sleef_erff16_u10(a);
  }

  static DOUBLE_TYPE exp(DOUBLE_TYPE a) {
    return Sleef_expd8_u10(a);
  }

  static FLOAT_TYPE expf(FLOAT_TYPE a) {
    return Sleef_expf16_u10(a);
  }

  static DOUBLE_TYPE mul(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return _mm512_mul_pd(a, b);
  }

  static DOUBLE_TYPE mul_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE b_simd = _mm512_set1_pd(b);
    return _mm512_mul_pd(a, b_simd);
  }

  static FLOAT_TYPE mulf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return _mm512_mul_ps(a, b);
  }

  static FLOAT_TYPE mulf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE b_simd = _mm512_set1_ps(b);
    return _mm512_mul_ps(a, b_simd);
  }

  static DOUBLE_TYPE neg(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE minus_zero = _mm512_set1_pd(-0.0);
    return _mm512_xor_pd(a, minus_zero);
  }

  static DOUBLE_TYPE pdf(DOUBLE_TYPE a) {
    return generic_pdf<AVX512>(a);
  }

  static FLOAT_TYPE pdff(FLOAT_TYPE a) {
    return generic_pdff<AVX512>(a);
  }

  static FLOAT_TYPE negf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE minus_zero = _mm512_set1_ps(-0.0);
    return _mm512_xor_ps(a, minus_zero);
  }

  static DOUBLE_TYPE recip(DOUBLE_TYPE a) noexcept {
    // Use division rather than reciprocal instruction, for
    // higher precision. Not sure if we care?
    DOUBLE_TYPE one = _mm512_set1_pd(1.0);
    return _mm512_div_pd(one, a);
  }

  static FLOAT_TYPE recipf(FLOAT_TYPE a) noexcept {
    // Use division rather than reciprocal instruction, for
    // higher precision. Not sure if we care?
    FLOAT_TYPE one = _mm512_set1_ps(1.0);
    return _mm512_div_ps(one, a);
  }

  static DOUBLE_TYPE tanh(DOUBLE_TYPE a) {
    return Sleef_tanhd8_u10(a);
  }

  static FLOAT_TYPE tanhf(FLOAT_TYPE a) {
    return Sleef_tanhf16_u10(a);
  }

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
