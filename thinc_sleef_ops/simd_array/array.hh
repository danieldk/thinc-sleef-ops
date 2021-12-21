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

  void erf(double *a, size_t n) noexcept;

  void erff(float *a, size_t n) noexcept;

  void exp(double *a, size_t n) noexcept;

  void expf(float *a, size_t n) noexcept;

  void gelu(double *a, size_t n) noexcept;

  void gelu_backward(double* a, size_t n) noexcept;

  void geluf(float *a, size_t n) noexcept;

  void geluf_backward(float* a, size_t n) noexcept;

  void logistic_cdf(double *a, size_t n) noexcept;

  void logistic_cdff(float *a, size_t n) noexcept;

  void swish(double *a, size_t n) noexcept;

  void swish_backward(double* a, size_t n) noexcept;

  void swishf(float *a, size_t n) noexcept;

  void swishf_backward(float* a, size_t n) noexcept;

  void tanh(double *a, size_t n) noexcept;

  void tanhf(float *a, size_t n) noexcept;
};

#endif // ARRAY_HH
