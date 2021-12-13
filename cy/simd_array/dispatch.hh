#ifndef DISPATCH_HH
#define DISPATCH_HH

#include <cstddef>
#include <memory>
#include <string>

#include "../simd_vector/vector.hh"
#include "array_base.hh"

enum InstructionSet {
  INSTRUCTION_SET_SCALAR,
  INSTRUCTION_SET_SSE2,
  INSTRUCTION_SET_AVX,
  INSTRUCTION_SET_AVX512F
};


/**
 * Detect supported (SIMD) instruction sets.
 */
std::unordered_set<InstructionSet> instruction_sets();

/**
 * Create an array for the best supported instruction set.
 */
std::unique_ptr<ArrayBase> create_array();

/**
 * Create an array for a specific instruction set.
 */
std::unique_ptr<ArrayBase> create_array_for_instruction_set(InstructionSet feature);

#endif // DISPATCH_HH
