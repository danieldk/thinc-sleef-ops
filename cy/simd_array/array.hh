#ifndef ARRAY_HH
#define ARRAY_HH

#include <functional>
#include <cstddef>

#include "../simd_vector/vector.hh"

#include "array_base.hh"

template <class T>
struct Array: ArrayBase {
  static size_t const N_FLOAT = Vector<T>::N_FLOAT;
  typedef typename Vector<T>::LOWER_TYPE LOWER_TYPE;

  void erff(float *a, size_t n) noexcept;

  void expf(float *a, size_t n) noexcept;

  void tanhf(float *a, size_t n) noexcept;
};

#endif // ARRAY_HH
