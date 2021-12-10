#ifndef VECTOR_AVX512_HH
#define VECTOR_AVX512_HH

#include <sleef.h>

#include <functional>
#include <cstddef>
#include <cstring>

#include "vector.hh"

template <>
struct Vector<AVX512> {
  typedef __m512 TYPE;
  typedef AVX LOWER_TYPE;
  static size_t const N_FLOAT = 8;

  static void erff(float *a, size_t n) {
    with_load_store(Sleef_erff16_u10, a, n);
  }

  static void expf(float *a, size_t n) {
    with_load_store(Sleef_expf16_u10, a, n);
  }

  static void tanhf(float *a, size_t n) {
    with_load_store(Sleef_tanhf16_u10, a, n);
  }

private:
  template <class F>
  static void with_load_store(F f, float *a, size_t n) {
    float buf[N_FLOAT - 1];
    float *b = a;
    if (n < N_FLOAT) {
      b = reinterpret_cast<float *>(&buf);
      memcpy(b, a, n * sizeof(float));
    }

    TYPE val = _mm512_loadu_ps(b);
    val = f(val);
    _mm512_storeu_ps(b, val);

    if (n < N_FLOAT) {
      memcpy(a, b, n * sizeof(float));
    }

  }
};

#endif // VECTOR_AVX512_HH
