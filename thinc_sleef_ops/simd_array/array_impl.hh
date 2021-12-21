#ifndef ARRAY_IMPL_H_
#define ARRAY_IMPL_H_

#include <cmath>
#include <cstddef>
#include <vector>

#include "../simd_vector/vector.hh"

#if defined(__ARM_NEON)
#include "../simd_vector/vector_neon.hh"
#endif

#if defined(__SSE__)
#include "../simd_vector/vector_sse.hh"
#endif

#if defined(__AVX__)
#include "simd_vector/vector_avx.hh"
#endif

#if defined(__AVX512F__)
#include "simd_vector/vector_avx512.hh"
#endif

#include "array_base.hh"

template <class T>
struct Array : ArrayBase {
  static size_t const N_DOUBLE = Vector<T>::N_DOUBLE;
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;

  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void erf(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::erf, &Array<LOWER_TYPE>::erf, a, n);
  }

  void erff(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::erff, &Array<LOWER_TYPE>::erff, a, n);
  }

  void exp(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::exp, &Array<LOWER_TYPE>::exp, a, n);
  }

  void expf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::expf, &Array<LOWER_TYPE>::expf, a, n);
  }

  void gelu(double *a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      // GELU(x) = x · Φ(x)
      auto cdf = Vector<T>::cdf(a);
      return Vector<T>::mul(a, cdf);
    }, &Array<LOWER_TYPE>::gelu, a, n);
  }

  void gelu_backward(double* a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      // GELU'(x) = Φ(x) + x · PDF(x)
      auto cdf = Vector<T>::cdf(a);
      auto pdf = Vector<T>::pdf(a);
      auto x_pdf = Vector<T>::mul(a, pdf);
      return Vector<T>::add(x_pdf, cdf);
    }, &Array<LOWER_TYPE>::gelu_backward, a, n);
  }

  void geluf(float *a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      // GELU(x) = x · Φ(x)
      auto cdf = Vector<T>::cdff(a);
      return Vector<T>::mulf(a, cdf);
    }, &Array<LOWER_TYPE>::geluf, a, n);
  }

  void geluf_backward(float* a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      // GELU'(x) = Φ(x) + x · PDF(x)
      auto cdf = Vector<T>::cdff(a);
      auto pdf = Vector<T>::pdff(a);
      auto x_pdf = Vector<T>::mulf(a, pdf);
      return Vector<T>::addf(x_pdf, cdf);
    }, &Array<LOWER_TYPE>::geluf_backward, a, n);
  }

  void logisticf(double *a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      auto r = Vector<T>::neg(a);
      r = Vector<T>::exp(r);
      r = Vector<T>::add_scalar(r, 1.0);
      return Vector<T>::recip(r);
    }, &Array<LOWER_TYPE>::logisticf, a, n);
  }

  void logisticff(float *a, size_t n) noexcept {
    apply_elementwise([](auto a) {
      auto r = Vector<T>::negf(a);
      r = Vector<T>::expf(r);
      r = Vector<T>::addf_scalar(r, 1.0);
      return Vector<T>::recipf(r);
    }, &Array<LOWER_TYPE>::logisticff, a, n);
  }

  void tanh(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanh, &Array<LOWER_TYPE>::tanh, a, n);
  }

  void tanhf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanhf, &Array<LOWER_TYPE>::tanhf, a, n);
  }

private:
  template <class F, class G>
  static void apply_elementwise(F f, G f_rest, float *a, size_t n) {
    size_t upper = n - (n % N_FLOAT);
    for (float *cur = a; cur != a + upper; cur += N_FLOAT) {
      Vector<T>::with_load_store(f, cur);
    }

    if (upper != n) {
      Array<LOWER_TYPE> array_lower;
      (array_lower.*f_rest)(a + upper, n - upper);
    }
  }

  template <class F, class G>
  static void apply_elementwise(F f, G f_rest, double *a, size_t n) {
    size_t upper = n - (n % N_DOUBLE);
    for (double *cur = a; cur != a + upper; cur += N_DOUBLE) {
      Vector<T>::with_load_store(f, cur);
    }

    if (upper != n) {
      Array<LOWER_TYPE> array_lower;
      (array_lower.*f_rest)(a + upper, n - upper);
    }
  }
};

#endif // ARRAY_IMPL_H_
