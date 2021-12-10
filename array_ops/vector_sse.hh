#ifndef VECTOR_SSE_HH
#define VECTOR_SSE_HH

#include <sleef.h>

#include <cstddef>
#include <cstring>
#include <functional>

#include "vector.hh"

template <>
struct Vector<SSE> {
  typedef __m128 TYPE;
  typedef Scalar LOWER_TYPE;
  static size_t const N_FLOAT = 4;

  static void erff(float *a, size_t n) noexcept {
    with_load_store(Sleef_erff4_u10, a, n);
  }

  static void expf(float *a, size_t n) noexcept {
    with_load_store(Sleef_expf4_u10, a, n);
  }

  static void tanhf(float *a, size_t n) noexcept {
    with_load_store(Sleef_tanhf4_u10, a, n);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a, size_t n) noexcept {
    float buf[N_FLOAT - 1];
    float *b = a;
    if (n < N_FLOAT) {
      b = reinterpret_cast<float *>(&buf);
      memcpy(b, a, n * sizeof(float));
    }

    TYPE val = _mm_loadu_ps(a);
    val = f(val);
    _mm_storeu_ps(a, val);

    if (n < N_FLOAT) {
      memcpy(a, b, n * sizeof(float));
    }
  }
};

#endif // VECTOR_SSE_HH
