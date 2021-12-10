from cython.operator cimport dereference as deref

from .cpu_id import CPUID

cdef class ArrayOps:
    def __init__(self):
        self.array.reset(new Array[Scalar]())
        features = CPUID()

        if "avx512f" in features.features:
            self.array.reset(new Array[AVX512]())
        elif "avx" in features.features:
            self.array.reset(new Array[AVX]())
        elif "sse2" in features.features:
            self.array.reset(new Array[SSE]())

    def erf(self, a: float[:], in_place: bool=False):
        if not in_place:
            a = a.copy()
        deref(self.array).erff(&a[0], len(a))
        return a

    def exp(self, a: float[:], in_place: bool=False):
        if not in_place:
            a = a.copy()
        deref(self.array).expf(&a[0], len(a))
        return a

    def tanh(self, a: float[:], in_place: bool=False):
        if not in_place:
            a = a.copy()
        deref(self.array).tanhf(&a[0], len(a))
        return a
