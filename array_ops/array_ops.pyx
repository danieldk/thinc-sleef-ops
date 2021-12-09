from cython.operator cimport dereference as deref

cdef class ArrayOps:
    def __init__(self):
        self.array.reset(new Array[SSE]())

    def erf(self, a: float[:], in_place: bool=False):
        if not in_place:
            a = a.copy()
        self.erff(a)
        return a

    def exp(self, a: float[:], in_place: bool=False):
        if not in_place:
            a = a.copy()
        self.expf(a)
        return a

    cpdef erff(self, float[:] a):
        deref(self.array).erff(&a[0], len(a))

    cpdef expf(self, float[:] a):
        deref(self.array).expf(&a[0], len(a))
