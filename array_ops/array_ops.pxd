cdef extern from "array.hh":
     cdef cppclass Array[T]:
         @staticmethod
         void erff(float *a, size_t n)

         @staticmethod
         void expf(float *a, size_t n)

     cdef cppclass Scalar:
         pass

     cdef cppclass SSE:
         pass

     cdef cppclass AVX:
         pass

cdef class FloatArray:
    cdef Array[Scalar] array


cdef class ArrayOps:
  cpdef erff(self, float[:] a)
  cpdef expf(self, float[:] a)
  cdef Array[Scalar] scalar_array
  cdef Array[SSE] sse_array
  cdef Array[AVX] avx_array
  cdef Array[AVX] array
