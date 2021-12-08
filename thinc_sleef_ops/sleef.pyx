cdef class ArrayOps:
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
        self.array.erff(&a[0], len(a))

    cpdef expf(self, float[:] a):
        self.array.expf(&a[0], len(a))
