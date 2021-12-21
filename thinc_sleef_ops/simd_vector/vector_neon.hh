#ifndef VECTOR_NEON_HH
#define VECTOR_NEON_HH

#include <sleef.h>

#include <cstddef>
#include <functional>

#include "vector.hh"

template <>
struct Vector<NEON> {
  typedef float64x2_t DOUBLE_TYPE;
  static size_t const N_DOUBLE = 2;

  typedef float32x4_t FLOAT_TYPE;
  static size_t const N_FLOAT = 4;

  typedef Scalar LOWER_TYPE;

  static DOUBLE_TYPE add_scalar(DOUBLE_TYPE a, double b) noexcept {
    DOUBLE_TYPE v_simd = vdupq_n_f64(b);
    return vaddq_f64(a, v_simd);
  }

  static DOUBLE_TYPE add(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return vaddq_f64(a, b);
  }

  static FLOAT_TYPE addf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE v_simd = vdupq_n_f32(b);
    return vaddq_f32(a, v_simd);
  }

  static FLOAT_TYPE addf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return vaddq_f32(a, b);
  }

  static DOUBLE_TYPE div(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return vdivq_f64(a, b);
  }

  static FLOAT_TYPE divf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return vdivq_f32(a, b);
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

  static DOUBLE_TYPE logistic_cdf(DOUBLE_TYPE a) {
    return generic_logistic_cdf<NEON>(a);
  }

  static FLOAT_TYPE logistic_cdff(FLOAT_TYPE a) {
    return generic_logistic_cdff<NEON>(a);
  }

  static DOUBLE_TYPE logistic_pdf(DOUBLE_TYPE a) {
    return generic_logistic_pdf<NEON>(a);
  }

  static FLOAT_TYPE logistic_pdff(FLOAT_TYPE a) {
    return generic_logistic_pdff<NEON>(a);
  }

  static DOUBLE_TYPE mul(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return vmulq_f64(a, b);
  }

  static DOUBLE_TYPE mul_scalar(FLOAT_TYPE a, double b) noexcept {
    DOUBLE_TYPE v_simd = vdupq_n_f64(b);
    return vmulq_f64(a, v_simd);
  }

  static FLOAT_TYPE mulf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return vmulq_f32(a, b);
  }

  static FLOAT_TYPE mulf_scalar(FLOAT_TYPE a, float b) noexcept {
    FLOAT_TYPE v_simd = vdupq_n_f32(b);
    return vmulq_f32(a, v_simd);
  }

  static DOUBLE_TYPE neg(DOUBLE_TYPE a) noexcept {
    return vnegq_f64(a);
  }

  static FLOAT_TYPE negf(FLOAT_TYPE a) noexcept {
    return vnegq_f32(a);
  }

  static DOUBLE_TYPE normal_cdf(DOUBLE_TYPE a) {
    return generic_normal_cdf<NEON>(a);
  }

  static FLOAT_TYPE normal_cdff(FLOAT_TYPE a) {
    return generic_normal_cdff<NEON>(a);
  }

  static DOUBLE_TYPE normal_pdf(DOUBLE_TYPE a) {
    return generic_normal_pdf<NEON>(a);
  }

  static FLOAT_TYPE normal_pdff(FLOAT_TYPE a) {
    return generic_normal_pdff<NEON>(a);
  }

  static DOUBLE_TYPE recip(DOUBLE_TYPE a) noexcept {
    DOUBLE_TYPE one = vdupq_n_f64(1);
    return vdivq_f64(one, a);
  }

  static FLOAT_TYPE recipf(FLOAT_TYPE a) noexcept {
    FLOAT_TYPE one = vdupq_n_f32(1);
    return vdivq_f32(one, a);
  }

  static DOUBLE_TYPE tanh(DOUBLE_TYPE a) noexcept {
    return Sleef_tanhd2_u10(a);
  }

  static FLOAT_TYPE tanhf(FLOAT_TYPE a) noexcept {
    return Sleef_tanhf4_u10(a);
  }

  template <class F>
  static void with_load_store(F f, float *a) noexcept {
    FLOAT_TYPE val = vld1q_f32(a);
    val = f(val);
    vst1q_f32(a, val);
  }

  template <class F>
  static void with_load_store(F f, double *a) noexcept {
    DOUBLE_TYPE val = vld1q_f64(a);
    val = f(val);
    vst1q_f64(a, val);
  }
};

#endif // VECTOR_NEON_HH
