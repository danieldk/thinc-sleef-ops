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
