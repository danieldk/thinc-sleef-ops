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
  static size_t const N_DOUBLE = Vector<T>::N_DOUBLE;
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void add(double *a, size_t n, double v) noexcept {
    apply_elementwise([v](double *a) { Vector<T>::add(a, v); }, [v](double *a, size_t n) { return Array<LOWER_TYPE>().add(a, n, v); }, a, n);
  }

  void add(double *a, double *v, size_t n) noexcept {
    apply_pairwise([](double *a, double *b) { return Vector<T>::add(a, b); },
                   [](double *a, double *v, size_t n) { return Array<LOWER_TYPE>().add(a, v, n); },
                   a, v, n);
  }

  void addf(float *a, size_t n, float v) noexcept {
    apply_elementwise([v](float *a) { Vector<T>::addf(a, v); }, [v](float *a, size_t n) { return Array<LOWER_TYPE>().addf(a, n, v); }, a, n);
  }

  void addf(float *a, float *v, size_t n) noexcept {
    apply_pairwise([](float *a, float *b) { return Vector<T>::addf(a, b); },
                   [](float *a, float *v, size_t n) { return Array<LOWER_TYPE>().addf(a, v, n); },
                   a, v, n);
  }

  void cdf(double *a, size_t n) noexcept {
    // Φ(x) = 1/2[1 + erf(x/sqrt(2))]
    Array<T>::mul(a, n, M_SQRT1_2);
    Array<T>::erf(a, n);
    Array<T>::add(a, n, 1.0);
    Array<T>::mul(a, n, 0.5);
  }

  void cdff(float *a, size_t n) noexcept {
    // Φ(x) = 1/2[1 + erf(x/sqrt(2))]
    Array<T>::mulf(a, n, M_SQRT1_2);
    Array<T>::erff(a, n);
    Array<T>::addf(a, n, 1.0);
    Array<T>::mulf(a, n, 0.5);
  }

  void pdf(double *a, size_t n) noexcept {
    Array<T>::mul(a, a, n);
    Array<T>::mul(a, n, -0.5);
    Array<T>::exp(a, n);
    Array<T>::mul(a, n, M_1_SQRT_2PI);
  }

  void pdff(float *a, size_t n) noexcept {
    Array<T>::mulf(a, a, n);
    Array<T>::mulf(a, n, -0.5);
    Array<T>::expf(a, n);
    Array<T>::mulf(a, n, M_1_SQRT_2PI);
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
    std::vector<double> cdf(a, a + n);
    Array<T>::cdf(cdf.data(), n);
    Array<T>::mul(a, cdf.data(), n);
  }

  void gelu_backward(double* a, size_t n) noexcept {
    // Φ(x)
    std::vector<double> cdf(a, a + n);
    Array<T>::cdf(cdf.data(), n);

    // PDF(x)
    std::vector<double> pdf(a, a + n);
    Array<T>::pdf(pdf.data(), n);

    // a = x · PDF(x)
    Array<T>::mul(a, pdf.data(), n);

    // a = Φ(x) + x · PDF(x)
    Array<T>::add(a, cdf.data(), n);
  }

  void geluf(float *a, size_t n) noexcept {
    std::vector<float> cdf(a, a + n);
    Array<T>::cdff(cdf.data(), n);
    Array<T>::mulf(a, cdf.data(), n);
  }

  void geluf_backward(float* a, size_t n) noexcept {
    // Φ(x)
    std::vector<float> cdf(a, a + n);
    Array<T>::cdff(cdf.data(), n);

    // PDF(x)
    std::vector<float> pdf(a, a + n);
    Array<T>::pdff(pdf.data(), n);

    // a = x · PDF(x)
    Array<T>::mulf(a, pdf.data(), n);

    // a = Φ(x) + x · PDF(x)
    Array<T>::addf(a, cdf.data(), n);
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

  void mul(double *a, size_t n, double v) noexcept {
    apply_elementwise([v](double *a) { Vector<T>::mul(a, v); }, [v](double *a, size_t n) { return Array<LOWER_TYPE>().mul(a, n, v); }, a, n);
  }

  void mul(double *a, size_t n, double *v) noexcept {
    apply_elementwise([v](double *a) { Vector<T>::mul(a, v); }, [v](double *a, size_t n) { return Array<LOWER_TYPE>().mul(a, n, v); }, a, n);
  }

  void mul(double *a, double *v, size_t n) noexcept {
    apply_pairwise([](double *a, double *b) { return Vector<T>::mul(a, b); },
                   [](double *a, double *v, size_t n) { return Array<LOWER_TYPE>().mul(a, v, n); },
                   a, v, n);
  }

  void mulf(float *a, size_t n, float v) noexcept {
    apply_elementwise([v](float *a) { Vector<T>::mulf(a, v); }, [v](float *a, size_t n) { return Array<LOWER_TYPE>().mulf(a, n, v); }, a, n);
  }

  void mulf(float *a, float *v, size_t n) noexcept {
    apply_pairwise([](float *a, float *b) { return Vector<T>::mulf(a, b); },
                   [](float *a, float *v, size_t n) { return Array<LOWER_TYPE>().mulf(a, v, n); },
                   a, v, n);
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

  template <class F, class G>
  static void apply_pairwise(F f, G f_rest, double *a, double *b, size_t n) {
    size_t upper = n - (n % N_DOUBLE);
    for (double *cur_a = a, *cur_b = b; cur_a != a + upper; cur_a += N_DOUBLE, cur_b += N_DOUBLE) {
      f(cur_a, cur_b);
    }

    if (upper != n) {
      f_rest(a + upper, b + upper, n - upper);
    }
  }

  template <class F, class G>
  static void apply_pairwise(F f, G f_rest, float *a, float *b, size_t n) {
    size_t upper = n - (n % N_FLOAT);
    for (float *cur_a = a, *cur_b = b; cur_a != a + upper; cur_a += N_FLOAT, cur_b += N_FLOAT) {
      f(cur_a, cur_b);
    }

    if (upper != n) {
      f_rest(a + upper, b + upper, n - upper);
    }
  }
};

#endif // ARRAY_IMPL_H_
