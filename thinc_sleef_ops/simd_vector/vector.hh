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
  typedef double DOUBLE_TYPE;
  static size_t const N_DOUBLE = 1;
  typedef float FLOAT_TYPE;
  static size_t const N_FLOAT = 1;
};

#define M_1_SQRT_2PI 0.398942280401432677939946059934

template <class T>
static typename Vector<T>::DOUBLE_TYPE generic_cdf(typename Vector<T>::DOUBLE_TYPE a) {
  auto r = Vector<T>::mul_scalar(a, M_SQRT1_2);
  r = Vector<T>::erf(r);
  r = Vector<T>::add_scalar(r, 1.0);
  return Vector<T>::mul_scalar(r, 0.5);
}

template <class T>
static typename Vector<T>::FLOAT_TYPE generic_cdff(typename Vector<T>::FLOAT_TYPE a) {
  auto r = Vector<T>::mulf_scalar(a, M_SQRT1_2);
  r = Vector<T>::erff(r);
  r = Vector<T>::addf_scalar(r, 1.0);
  return Vector<T>::mulf_scalar(r, 0.5);
}

template <class T>
static typename Vector<T>::DOUBLE_TYPE generic_pdf(typename Vector<T>::DOUBLE_TYPE a) {
  auto r = Vector<T>::mul(a, a);
  r = Vector<T>::mul_scalar(r, -0.5);
  r = Vector<T>::exp(r);
  return Vector<T>::mul_scalar(r, M_1_SQRT_2PI);
}

template <class T>
static typename Vector<T>::FLOAT_TYPE generic_pdff(typename Vector<T>::FLOAT_TYPE a) {
  auto r = Vector<T>::mulf(a, a);
  r = Vector<T>::mulf_scalar(r, -0.5);
  r = Vector<T>::expf(r);
  return Vector<T>::mulf_scalar(r, M_1_SQRT_2PI);
}

template<>
struct Vector<Scalar> {
  typedef double DOUBLE_TYPE;
  static size_t const N_DOUBLE = 1;

  typedef float FLOAT_TYPE;
  static size_t const N_FLOAT = 1;

  typedef Scalar LOWER_TYPE;

  static DOUBLE_TYPE add(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return a + b;
  }

  static DOUBLE_TYPE add_scalar(DOUBLE_TYPE a, double b) noexcept {
    return a + b;
  }

  static FLOAT_TYPE addf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return a + b;
  }

  static FLOAT_TYPE addf_scalar(FLOAT_TYPE a, float b) noexcept {
    return a + b;
  }

  static DOUBLE_TYPE cdf(DOUBLE_TYPE a) {
    return generic_cdf<Scalar>(a);
  }

  static FLOAT_TYPE cdff(FLOAT_TYPE a) {
    return generic_cdff<Scalar>(a);
  }

  static DOUBLE_TYPE erf(DOUBLE_TYPE a) noexcept {
    return std::erf(a);
  }

  static FLOAT_TYPE erff(FLOAT_TYPE a) noexcept {
    return std::erf(a);
  }

  static DOUBLE_TYPE exp(DOUBLE_TYPE a) noexcept {
    return std::exp(a);
  }

  static FLOAT_TYPE expf(FLOAT_TYPE a) noexcept {
    return std::exp(a);
  }

  static DOUBLE_TYPE mul(DOUBLE_TYPE a, DOUBLE_TYPE b) noexcept {
    return a * b;
  }

  static DOUBLE_TYPE mul_scalar(DOUBLE_TYPE a, double b) noexcept {
    return a * b;
  }

  static FLOAT_TYPE mulf(FLOAT_TYPE a, FLOAT_TYPE b) noexcept {
    return a * b;
  }

  static FLOAT_TYPE mulf_scalar(FLOAT_TYPE a, float b) noexcept {
    return a * b;
  }

  static DOUBLE_TYPE neg(DOUBLE_TYPE a) noexcept {
    return -a;
  }

  static FLOAT_TYPE negf(FLOAT_TYPE a) noexcept {
    return -a;
  }

  static DOUBLE_TYPE pdf(DOUBLE_TYPE a) {
    return generic_pdf<Scalar>(a);
  }

  static FLOAT_TYPE pdff(FLOAT_TYPE a) {
    return generic_pdff<Scalar>(a);
  }

  static DOUBLE_TYPE recip(DOUBLE_TYPE a) noexcept {
    return 1.0 / a;
  }

  static FLOAT_TYPE recipf(FLOAT_TYPE a) noexcept {
    return 1.0 / a;
  }

  static DOUBLE_TYPE tanh(DOUBLE_TYPE a) noexcept {
    return Sleef_tanh_u10(a);
  }

  static FLOAT_TYPE tanhf(FLOAT_TYPE a) noexcept {
    return Sleef_tanhf_u10(a);
  }

  template <class F>
  static void with_load_store(F f, float *a) noexcept {
    FLOAT_TYPE val = *a;
    val = f(val);
    *a = val;
  }

  template <class F>
  static void with_load_store(F f, double *a) noexcept {
    DOUBLE_TYPE val = *a;
    val = f(val);
    *a = val;
  }
};

#endif // VECTOR_HH
