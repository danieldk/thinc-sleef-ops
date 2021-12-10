from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr

cdef extern from "arrayi.hh":
     cdef cppclass ArrayI:
         void erff(float *a, size_t n)
         void expf(float *a, size_t n)
         void tanhf(float *a, size_t n)

cdef extern from "array.hh":
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

cdef class ArrayOps:
  cdef unique_ptr[ArrayI] array
