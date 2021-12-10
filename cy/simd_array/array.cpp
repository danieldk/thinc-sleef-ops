#include <memory>
#include <string>

#include "array.hh"

std::unique_ptr<ArrayBase> array_for_instruction_set(CPUFeature feature) {
  switch (feature) {
    case FEATURE_AVX:
      return std::unique_ptr<ArrayBase>(new Array<AVX>());
    case FEATURE_AVX512F:
      return std::unique_ptr<ArrayBase>(new Array<AVX512>());
    case FEATURE_SSE2:
      return std::unique_ptr<ArrayBase>(new Array<SSE>());
    case FEATURE_SCALAR:
      return std::unique_ptr<ArrayBase>(new Array<Scalar>());
    default:
      break;
  }

  return std::unique_ptr<ArrayBase>();
}
