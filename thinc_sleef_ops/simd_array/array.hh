#ifndef ARRAY_HH
#define ARRAY_HH

#include <cstddef>
#include <memory>
#include <string>

#include "../simd_vector/vector.hh"
#include "array_base.hh"

template <class T>
struct Array: ArrayBase {
  static size_t const N_DOUBLE = Vector<T>::N_DOUBLE;
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void add(double *a, size_t n, double v) noexcept;

  void addf(float *a, size_t n, float v) noexcept;

  void erf(double *a, size_t n) noexcept;

  void erff(float *a, size_t n) noexcept;

  void exp(double *a, size_t n) noexcept;

  void expf(float *a, size_t n) noexcept;

  void logisticf(double *a, size_t n) noexcept;

  void logisticff(float *a, size_t n) noexcept;

  void neg(double *a, size_t n) noexcept;

  void negf(float *a, size_t n) noexcept;

  void recip(double *a, size_t n) noexcept;

  void recipf(float *a, size_t n) noexcept;

  void tanh(double *a, size_t n) noexcept;

  void tanhf(float *a, size_t n) noexcept;
};

#endif // ARRAY_HH