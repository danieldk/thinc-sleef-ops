#ifndef VECTOR_NEON_HH
#define VECTOR_NEON_HH

#include <sleef.h>

#include <cstddef>
#include <functional>

#include "vector.hh"

template <>
struct Vector<NEON> {
  typedef float32x4_t TYPE;
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 4;

  static void erff(float *a) noexcept {
    with_load_store(Sleef_erff4_u10, a);
  }

  static void expf(float *a) noexcept {
    with_load_store(Sleef_expf4_u10, a);
  }

  static void tanhf(float *a) noexcept {
    with_load_store(Sleef_tanhf4_u10, a);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a) noexcept {
    TYPE val = vld1q_f32(a);
    val = f(val);
    vst1q_f32(a, val);
  }
};

#endif // VECTOR_NEON_HH
