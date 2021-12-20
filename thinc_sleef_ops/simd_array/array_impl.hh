#ifndef ARRAY_IMPL_H_
#define ARRAY_IMPL_H_

#include <cstddef>

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

  void add(double *a, size_t n, double v) noexcept {
    apply_elementwise([v](double *a) { Vector<T>::add(a, v); }, [v](double *a, size_t n) { return Array<LOWER_TYPE>().add(a, n, v); }, a, n);
  }

  void addf(float *a, size_t n, float v) noexcept {
    apply_elementwise([v](float *a) { Vector<T>::addf(a, v); }, [v](float *a, size_t n) { return Array<LOWER_TYPE>().addf(a, n, v); }, a, n);
  }

  void erf(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::erf, [](double *a, size_t n) { return Array<LOWER_TYPE>().erf(a, n); }, a, n);
  }

  void erff(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::erff, [](float *a, size_t n) { return Array<LOWER_TYPE>().erff(a, n); }, a, n);
  }

  void exp(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::exp, [](double *a, size_t n) { return Array<LOWER_TYPE>().exp(a, n); }, a, n);
  }

  void expf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::expf, [](float *a, size_t n) { return Array<LOWER_TYPE>().expf(a, n); }, a, n);
  }

  void logisticf(double *a, size_t n) noexcept {
    Array<T>::neg(a, n);
    Array<T>::exp(a, n);
    Array<T>::add(a, n, 1.0);
    Array<T>::recip(a, n);
  }

  void logisticff(float *a, size_t n) noexcept {
    Array<T>::negf(a, n);
    Array<T>::expf(a, n);
    Array<T>::addf(a, n, 1.0);
    Array<T>::recipf(a, n);
  }

  void neg(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::neg, [](double *a, size_t n) { return Array<LOWER_TYPE>().neg(a, n); }, a, n);
  }

  void negf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::negf, [](float *a, size_t n) { return Array<LOWER_TYPE>().negf(a, n); }, a, n);
  }

  void recip(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::recip, [](double *a, size_t n) { return Array<LOWER_TYPE>().recip(a, n); }, a, n);
  }

  void recipf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::recipf, [](float *a, size_t n) { return Array<LOWER_TYPE>().recipf(a, n); }, a, n);
  }

  void tanh(double *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanh, [](double *a, size_t n) { return Array<LOWER_TYPE>().tanh(a, n); }, a, n);
  }

  void tanhf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanhf, [](float *a, size_t n) { return Array<LOWER_TYPE>().tanhf(a, n); }, a, n);
  }

private:
  template <class F, class G>
  static void apply_elementwise(F f, G f_rest, float *a, size_t n) {
    size_t upper = n - (n % N_FLOAT);
    for (float *cur = a; cur != a + upper; cur += N_FLOAT) {
      f(cur);
    }

    if (upper != n) {
      f_rest(a + upper, n - upper);
    }
  }

  template <class F, class G>
  static void apply_elementwise(F f, G f_rest, double *a, size_t n) {
    size_t upper = n - (n % N_DOUBLE);
    for (double *cur = a; cur != a + upper; cur += N_DOUBLE) {
      f(cur);
    }

    if (upper != n) {
      f_rest(a + upper, n - upper);
    }
  }

};

#endif // ARRAY_IMPL_H_
