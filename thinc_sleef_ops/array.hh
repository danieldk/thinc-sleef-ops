#ifndef ARRAY_HH
#define ARRAY_HH

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

  static void erff(float *a, size_t n);

  static void expf(float *a, size_t n);
};

#endif // ARRAY_HH
