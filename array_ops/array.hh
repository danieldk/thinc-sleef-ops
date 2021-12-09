#ifndef ARRAY_HH
#define ARRAY_HH

#include <functional>
#include <cstddef>

#include "vector.hh"

#include "arrayi.hh"

template <class T>
struct Array: ArrayI {
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void erff(float *a, size_t n) noexcept;

  void expf(float *a, size_t n) noexcept;
};

#endif // ARRAY_HH
