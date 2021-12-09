from cython.operator cimport dereference as deref

cdef class ArrayOps:
    def __init__(self):
        self.array.reset(new Array[Scalar]())
        if cpuid_present() == 0:
            return

        cdef cpu_raw_data_t cpu_raw_data
        cdef cpu_id_t cpu_id

        if cpuid_get_raw_data(&cpu_raw_data) < 0:
            return

        if cpu_identify(&cpu_raw_data, &cpu_id) < 0:
            return

        if cpu_id.flags[int(CPU_FEATURE_AVX512F)] == 1:
            self.array.reset(new Array[SSE]())
        elif cpu_id.flags[int(CPU_FEATURE_AVX)] == 1:
            self.array.reset(new Array[AVX]())
        elif cpu_id.flags[int(CPU_FEATURE_SSE)] == 1:
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
