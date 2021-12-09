#ifndef ARRAY_IMPL_H_
#define ARRAY_IMPL_H_

#include <functional>
#include <cstddef>

#include "vector.hh"

#if defined(__SSE__)
#include <vector_sse.hh>
#endif

#if defined(__AVX__)
#include <vector_avx.hh>
#endif

#if defined(__AVX512F__)
#include <vector_avx512.hh>
#endif

#include "arrayi.hh"

template <class T>
struct Array : ArrayI {
  static size_t CONST N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void erff(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::erff, [](float* a, size_t n) { return Array<LOWER_TYPE>().erff(a, n); }, a, n);
  }

  void expf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::expf, [](float* a, size_t n) { return Array<LOWER_TYPE>().expf(a, n); }, a, n);
  }

  void tanhf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanhf, [](float* a, size_t n) { return Array<LOWER_TYPE>().tanhf(a, n); }, a, n);
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
};


#endif // ARRAY_IMPL_H_
