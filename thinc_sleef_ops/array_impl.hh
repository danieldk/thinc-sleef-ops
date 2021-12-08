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

template <class T>
struct Array {
  static size_t CONST N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  static void erff(float *a, size_t n) {
    apply_elementwise(Vector<T>::erff, Array<LOWER_TYPE>::erff, a, n);
  }

  static void expf(float *a, size_t n) {
    apply_elementwise(Vector<T>::expf, Array<LOWER_TYPE>::expf, a, n);
  }

private:
  static void apply_elementwise(std::function<void(float *)> f, std::function<void(float *, size_t)> f_rest, float *a, size_t n) {
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
