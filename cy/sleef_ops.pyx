from cython.operator cimport dereference as deref

from .cpu_id import CPUID

cdef class SleefOps:
    def __init__(self):
        features = CPUID()
        if "avx512f" in features.features:
            self.array = array_for_instruction_set(FEATURE_AVX512F)
        elif "avx" in features.features:
            self.array = array_for_instruction_set(FEATURE_AVX)
        elif "sse2" in features.features:
            self.array = array_for_instruction_set(FEATURE_SSE2)
        else:
            self.array = array_for_instruction_set(FEATURE_SCALAR)

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
