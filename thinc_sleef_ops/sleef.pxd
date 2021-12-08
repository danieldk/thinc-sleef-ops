cdef extern from "array.hh":
    cdef cppclass Array[T]:
        @staticmethod
        void erff(float *a, size_t n)

        @staticmethod
        void expf(float *a, size_t n)

cdef extern from "../sleef/sleef.h":
    ctypedef struct __m128
    ctypedef struct __m256

cdef class ArrayOps:
  cpdef erff(self, float[:] a)
  cpdef expf(self, float[:] a)
  cdef Array[__m256] array
