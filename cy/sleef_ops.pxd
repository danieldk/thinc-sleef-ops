from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set

cdef extern from "simd_array/array_base.hh":
     cdef cppclass ArrayBase:
         void erff(float *a, size_t n)
         void expf(float *a, size_t n)
         void tanhf(float *a, size_t n)

cdef extern from "simd_array/dispatch.hh":
     # Note: keep in sync with dispatch.hh
     cpdef enum InstructionSet:
         INSTRUCTION_SET_AVX,
         INSTRUCTION_SET_AVX512F,
         INSTRUCTION_SET_NEON,
         INSTRUCTION_SET_SCALAR,
         INSTRUCTION_SET_SSE2

     unordered_set[InstructionSet] instruction_sets() except +
     unique_ptr[ArrayBase] create_array() except +
     unique_ptr[ArrayBase] create_array_for_instruction_set(InstructionSet instruction_set) except +

cdef class SleefOps:
  cdef unique_ptr[ArrayBase] array
