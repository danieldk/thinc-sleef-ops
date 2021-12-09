#ifndef ARRAY_HH
#define ARRAY_HH

#include <functional>
#include <cstddef>

#include "vector.hh"

template <class T>
struct Array {
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  static void erff(float *a, size_t n);

  static void expf(float *a, size_t n);
};

#endif // ARRAY_HH
