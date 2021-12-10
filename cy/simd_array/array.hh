#ifndef ARRAY_HH
#define ARRAY_HH

#include <cstddef>
#include <memory>
#include <string>

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

enum CPUFeature {
  FEATURE_AVX,
  FEATURE_AVX512F,
  FEATURE_SSE2,
  FEATURE_SCALAR
};

std::unique_ptr<ArrayBase> array_for_instruction_set(CPUFeature feature);

#endif // ARRAY_HH
