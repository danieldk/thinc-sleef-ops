from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set

ctypedef float[::1] float1d_t
ctypedef double[::1] double1d_t
ctypedef float* floats_t
ctypedef double* doubles_t

ctypedef size_t dim_t

cdef fused reals_ft:
    floats_t
    doubles_t
    float1d_t
    double1d_t

cdef extern from "simd_array/array_base.hh":
     cdef cppclass ArrayBase:
         void erf(double *a, size_t n)
         void erff(float *a, size_t n)
         void exp(double *a, size_t n)
         void expf(float *a, size_t n)
         void tanh(double *a, size_t n)
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

cdef class SleefArray:
  cdef unique_ptr[ArrayBase] array

  cdef void erf(self, reals_ft a, dim_t n)
  cdef void exp(self, reals_ft a, dim_t n)
  cdef void tanh(self, reals_ft a, dim_t n)
