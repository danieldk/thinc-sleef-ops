from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string;

cdef extern from "simd_array/array_base.hh":
     cdef cppclass ArrayBase:
         void erff(float *a, size_t n)
         void expf(float *a, size_t n)
         void tanhf(float *a, size_t n)

cdef extern from "simd_array/array.hh":
     cpdef enum CPUFeature:
         FEATURE_AVX,
         FEATURE_AVX512F,
         FEATURE_SSE2,
         FEATURE_SCALAR

     unique_ptr[ArrayBase] array_for_instruction_set(CPUFeature instruction_set)

cdef class SleefOps:
  cdef unique_ptr[ArrayBase] array
