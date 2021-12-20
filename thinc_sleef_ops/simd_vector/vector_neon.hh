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

  static void add(double *a, double v) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE v_simd = vdupq_n_f64(v);
      return vaddq_f64(a, v_simd);
    }, a);
  }

  static void addf(float *a, float v) noexcept {
    with_load_store([=](FLOAT_TYPE a){
      FLOAT_TYPE v_simd = vdupq_n_f32(v);
      return vaddq_f32(a, v_simd);
    }, a);
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

  static void neg(double *a) noexcept {
    with_load_store(vnegq_f64, a);
  }

  static void negf(float *a) noexcept {
    with_load_store(vnegq_f32, a);
  }

  static void recip(double *a) noexcept {
    with_load_store([=](DOUBLE_TYPE a){
      DOUBLE_TYPE one = vdupq_n_f64(1);
      return vdivq_f64(one, a);
    }, a);
  }

  static void recipf(float *a) noexcept {
    with_load_store([=](FLOAT_TYPE a){
      FLOAT_TYPE one = vdupq_n_f32(1);
      return vdivq_f32(one, a);
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
