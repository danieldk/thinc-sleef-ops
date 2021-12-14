from contextlib import contextmanager
from cython.operator cimport dereference as deref

cdef class SleefArray:
    def __init__(self):
        self.array.swap(create_array())

    @staticmethod
    def instruction_sets():
        return instruction_sets()


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

@contextmanager
def with_cpu_feature(InstructionSet feature):
    array = SleefArray()
    array.array.swap(create_array_for_instruction_set(feature))
    yield array
