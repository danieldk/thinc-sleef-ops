from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr

cdef extern from "simd_array/array_base.hh":
     cdef cppclass ArrayBase:
         void erff(float *a, size_t n)
         void expf(float *a, size_t n)
         void tanhf(float *a, size_t n)

cdef extern from "simd_array/array.hh":
     cdef cppclass Array[T]:
         void erff(float *a, size_t n)
         void expf(float *a, size_t n)

     cdef cppclass Scalar:
         pass

     cdef cppclass SSE:
         pass

     cdef cppclass AVX:
         pass

     cdef cppclass AVX512:
         pass

cdef class SleefOps:
  cdef unique_ptr[ArrayBase] array
