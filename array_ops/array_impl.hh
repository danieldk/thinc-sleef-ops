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
    apply_elementwise(Vector<T>::erff, a, n);
  }

  void expf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::expf, a, n);
  }

  void tanhf(float *a, size_t n) noexcept {
    apply_elementwise(Vector<T>::tanhf, a, n);
  }

private:
  template <class F>
  static void apply_elementwise(F f, float *a, size_t n) {
    for (size_t i = 0; i < n; i += N_FLOAT) {
      f(a + i, n - i);
    }
  }
};


#endif // ARRAY_IMPL_H_
