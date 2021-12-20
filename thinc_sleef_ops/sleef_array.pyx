from contextlib import contextmanager
from cython.operator cimport dereference as deref

cdef class SleefArray:
    def __init__(self):
        self.array.swap(create_array())

    @staticmethod
    def instruction_sets():
        return instruction_sets()

    cdef void erf(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).erff(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).erff(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).erf(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).erf(&a[0], n)
        else:
            pass

    cdef void exp(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).expf(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).expf(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).exp(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).exp(&a[0], n)
        else:
            pass

    cdef void gelu(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).geluf(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).geluf(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).gelu(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).gelu(&a[0], n)
        else:
            pass

    cdef void gelu_backward(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).geluf_backward(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).geluf_backward(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).gelu_backward(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).gelu_backward(&a[0], n)
        else:
            pass

    cdef void logisticf(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).logisticff(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).logisticff(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).logisticf(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).logisticf(&a[0], n)
        else:
            pass

    cdef void tanh(self, reals_ft a, dim_t n):
        if reals_ft is floats_t:
            deref(self.array).tanhf(a, n)
        elif reals_ft is float1d_t:
            deref(self.array).tanhf(&a[0], n)
        elif reals_ft is doubles_t:
            deref(self.array).tanh(a, n)
        elif reals_ft is double1d_t:
            deref(self.array).tanh(&a[0], n)
        else:
            pass

@contextmanager
def with_cpu_feature(InstructionSet feature):
    array = SleefArray()
    array.array.swap(create_array_for_instruction_set(feature))
    yield array
