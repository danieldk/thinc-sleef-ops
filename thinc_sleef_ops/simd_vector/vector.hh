#ifndef VECTOR_HH
#define VECTOR_HH

#include <functional>
#include <cmath>
#include <cstddef>

#include <sleef.h>

struct AVX {};
struct AVX512 {};
struct NEON {};
struct SSE {};
struct Scalar {};

template <class T>
struct Vector {
  typedef Scalar LOWER_TYPE;
  static size_t const N_DOUBLE = 1;
  static size_t const N_FLOAT = 1;
};

template<>
struct Vector<Scalar> {
  typedef double DOUBLE_TYPE;
  static size_t const N_DOUBLE = 1;

  typedef float FLOAT_TYPE;
  static size_t const N_FLOAT = 1;

  typedef Scalar LOWER_TYPE;

  static void add(double *a, double v) noexcept {
    *a += v;
  }

  static void add(double *a, double *v) noexcept {
    *a += *v;
  }

  static void addf(float *a, float v) noexcept {
    *a += v;
  }

  static void addf(float *a, float *v) noexcept {
    *a += *v;
  }

  static void erf(double *a) noexcept {
    *a = std::erf(*a);
  }

  static void erff(float *a) noexcept {
    *a = std::erf(*a);
  }

  static void exp(double *a) noexcept {
    *a = std::exp(*a);
  }

  static void expf(float *a) noexcept {
    *a = std::exp(*a);
  }

  static void mulf(float *a, float v) noexcept {
    *a *= v;
  }

  static void mulf(float *a, float *v) noexcept {
    *a *= *v;
  }

  static void mul(double *a, double v) noexcept {
    *a *= v;
  }

  static void mul(double *a, double *v) noexcept {
    *a *= *v;
  }

  static void neg(double *a) noexcept {
    *a = -*a;
  }

  static void negf(float *a) noexcept {
    *a = -*a;
  }

  static void recip(double *a) noexcept {
    *a = 1.0 / *a;
  }

  static void recipf(float *a) noexcept {
    *a = 1.0 / *a;
  }

  static void tanh(double *a) noexcept {
    *a = Sleef_tanh_u10(*a);
  }

  static void tanhf(float *a) noexcept {
    *a = Sleef_tanhf_u10(*a);
  }
};

#endif // VECTOR_HH
