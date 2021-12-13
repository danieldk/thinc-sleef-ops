#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "array.hh"
#include "dispatch.hh"

#if defined(__x86_64__) || defined(_M_X64)
#include <libcpuid.h>

std::unordered_set<InstructionSet> instruction_sets() {
  if (cpuid_present() == 0)
    throw std::runtime_error(std::string("CPU does not have CPUID instruction"));

  cpu_raw_data_t cpu_raw_data;
  cpu_id_t cpu_id;

  if (cpuid_get_raw_data(&cpu_raw_data) < 0)
    throw std::runtime_error(std::string("Cannot get raw CPU data: ") + cpuid_error());

  if (cpu_identify(&cpu_raw_data, &cpu_id) < 0)
    throw std::runtime_error(std::string("Cannot identify CPU: ") + cpuid_error());

  std::unordered_set<InstructionSet> features;
  features.insert(INSTRUCTION_SET_SCALAR);

  if (cpu_id.flags[CPU_FEATURE_SSE2])
    features.insert(INSTRUCTION_SET_SSE2);

  if (cpu_id.flags[CPU_FEATURE_AVX])
    features.insert(INSTRUCTION_SET_AVX);

  if (cpu_id.flags[CPU_FEATURE_AVX512F])
    features.insert(INSTRUCTION_SET_AVX512F);

  return features;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

std::unordered_set<InstructionSet> instruction_sets() {
  std::unordered_set<InstructionSet> features;
  features.insert(INSTRUCTION_SET_SCALAR);
  return features;
}

#else

std::unordered_set<InstructionSet> instruction_sets() {
  std::unordered_set<InstructionSet> features;
  features.insert(INSTRUCTION_SET_SCALAR);
  return features;
}

#endif

std::unique_ptr<ArrayBase> create_array() {
  auto features = instruction_sets();

  if (features.find(INSTRUCTION_SET_AVX512F) != features.end())
    return create_array_for_instruction_set(INSTRUCTION_SET_AVX512F);

  if (features.find(INSTRUCTION_SET_AVX) != features.end())
    return create_array_for_instruction_set(INSTRUCTION_SET_AVX);

  if (features.find(INSTRUCTION_SET_SSE2) != features.end())
    return create_array_for_instruction_set(INSTRUCTION_SET_SSE2);

  return create_array_for_instruction_set(INSTRUCTION_SET_SCALAR);
}

std::unique_ptr<ArrayBase> create_array_for_instruction_set(InstructionSet feature) {
  switch (feature) {
    #if defined(COMPILER_SUPPORTS_AVX)
    case INSTRUCTION_SET_AVX:
      return std::unique_ptr<ArrayBase>(new Array<AVX>());
    #endif

    #if defined(COMPILER_SUPPORTS_AVX512F)
    case INSTRUCTION_SET_AVX512F:
      return std::unique_ptr<ArrayBase>(new Array<AVX512>());
    #endif

    #if defined(COMPILER_SUPPORTS_AVX512F)
    case INSTRUCTION_SET_SSE2:
      return std::unique_ptr<ArrayBase>(new Array<SSE>());
    #endif

    case INSTRUCTION_SET_SCALAR:
      return std::unique_ptr<ArrayBase>(new Array<Scalar>());
    default:
      break;
  }

  throw std::runtime_error("Unknown instruction set");

}
