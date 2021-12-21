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

#define M_1_SQRT_2PI 0.398942280401432677939946059934

template <class T>
struct Array : ArrayBase {
  typedef typename Vector<T>::DOUBLE_TYPE DOUBLE_TYPE;
  static size_t const N_DOUBLE = Vector<T>::N_DOUBLE;

  typedef typename Vector<T>::FLOAT_TYPE FLOAT_TYPE;
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;

  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void cdf(double *a, size_t n) noexcept {
    // Φ(x) = 1/2[1 + erf(x/sqrt(2))]
    apply_elementwise(
      [](DOUBLE_TYPE a) {
        auto r = Vector<T>::mul_scalar(a, M_SQRT1_2);
        r = Vector<T>::erf(r);
        r = Vector<T>::add_scalar(r, 1.0);
        return Vector<T>::mul_scalar(r, 0.5);
      },
      [](double *a, size_t n) { return Array<LOWER_TYPE>().cdf(a, n); },
      a, n
    );
  }

  void cdff(float *a, size_t n) noexcept {
    // Φ(x) = 1/2[1 + erf(x/sqrt(2))]
    apply_elementwise(
      Vector<T>::cdff,
      [](float *a, size_t n) { return Array<LOWER_TYPE>().cdff(a, n); },
      a, n
    );
  }

  void pdf(double *a, size_t n) noexcept {
    apply_elementwise(
      [](DOUBLE_TYPE a) {
        auto r = Vector<T>::mul(a, a);
        r = Vector<T>::mul_scalar(r, -0.5);
        r = Vector<T>::exp(r);
        return Vector<T>::mul_scalar(r, M_1_SQRT_2PI);
      },
      [](double *a, size_t n) { return Array<LOWER_TYPE>().pdf(a, n); },
      a, n
    );
  }

  void pdff(float *a, size_t n) noexcept {
    apply_elementwise(
      Vector<T>::pdff,
      [](float *a, size_t n) { return Array<LOWER_TYPE>().pdff(a, n); },
      a, n
    );
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

  void gelu(double *a, size_t n) noexcept {
    apply_elementwise([](DOUBLE_TYPE a) {
      // GELU(x) = x · Φ(x)
      auto cdf = Vector<T>::cdf(a);
      return Vector<T>::mul(a, cdf);
    }, [](double *a, size_t n) { return Array<LOWER_TYPE>().gelu(a, n); }, a, n);
  }

  void gelu_backward(double* a, size_t n) noexcept {
    apply_elementwise([](DOUBLE_TYPE a) {
      // GELU'(x) = Φ(x) + x · PDF(x)
      auto cdf = Vector<T>::cdf(a);
      auto pdf = Vector<T>::pdf(a);
      auto x_pdf = Vector<T>::mul(a, pdf);
      return Vector<T>::add(x_pdf, cdf);
    }, [](double *a, size_t n) { return Array<LOWER_TYPE>().gelu_backward(a, n); }, a, n);
  }

  void geluf(float *a, size_t n) noexcept {
    apply_elementwise([](FLOAT_TYPE a) {
      // GELU(x) = x · Φ(x)
      auto cdf = Vector<T>::cdff(a);
      return Vector<T>::mulf(a, cdf);
    }, [](float *a, size_t n) { return Array<LOWER_TYPE>().geluf(a, n); }, a, n);
  }

  void geluf_backward(float* a, size_t n) noexcept {
    apply_elementwise([](FLOAT_TYPE a) {
      // GELU'(x) = Φ(x) + x · PDF(x)
      auto cdf = Vector<T>::cdff(a);
      auto pdf = Vector<T>::pdff(a);
      auto x_pdf = Vector<T>::mulf(a, pdf);
      return Vector<T>::addf(x_pdf, cdf);
    }, [](float *a, size_t n) { return Array<LOWER_TYPE>().geluf_backward(a, n); }, a, n);
  }

  void logisticf(double *a, size_t n) noexcept {
    apply_elementwise([](DOUBLE_TYPE a) {
      auto r = Vector<T>::neg(a);
      r = Vector<T>::exp(r);
      r = Vector<T>::add_scalar(r, 1.0);
      return Vector<T>::recip(r);
    }, [](double *a, size_t n) { return Array<LOWER_TYPE>().logisticf(a, n); }, a, n);
  }

  void logisticff(float *a, size_t n) noexcept {
    apply_elementwise([](FLOAT_TYPE a) {
      auto r = Vector<T>::negf(a);
      r = Vector<T>::expf(r);
      r = Vector<T>::addf_scalar(r, 1.0);
      return Vector<T>::recipf(r);
    }, [](float *a, size_t n) { return Array<LOWER_TYPE>().logisticff(a, n); }, a, n);
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
      Vector<T>::with_load_store(f, cur);
    }

    if (upper != n) {
      f_rest(a + upper, n - upper);
    }
  }

  template <class F, class G>
  static void apply_elementwise(F f, G f_rest, double *a, size_t n) {
    size_t upper = n - (n % N_DOUBLE);
    for (double *cur = a; cur != a + upper; cur += N_DOUBLE) {
      Vector<T>::with_load_store(f, cur);
    }

    if (upper != n) {
      f_rest(a + upper, n - upper);
    }
  }
};

#endif // ARRAY_IMPL_H_
