from array cimport Array, AVX, SSE, Scalar

cdef class FloatArray:
    cdef Array[Scalar] array


cdef class ArrayOps:
  cpdef erff(self, float[:] a)
  cpdef expf(self, float[:] a)
  cdef Array[Scalar] scalar_array
  cdef Array[SSE] sse_array
  cdef Array[AVX] avx_array
  cdef Array[AVX] array
