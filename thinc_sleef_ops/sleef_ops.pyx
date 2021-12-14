# cython: cdivision=True
# cython: infer_types=True
# cython: profile=True

from contextlib import contextmanager
cimport numpy as np
import numpy as np
from thinc.api import Ops

from .sleef_array cimport InstructionSet, SleefArray
from .sleef_array import with_cpu_feature as sleef_with_cpu_feature

class SleefOps(Ops):
    def __init__(self):
        self._array = SleefArray()

    @staticmethod
    def instruction_sets():
        return SleefArray.instruction_sets()

    def erf(self, a: np.ndarray):
        cdef SleefArray array = self._array

        a = self._to_contig_or_copy(a)
        if a.dtype == np.float32:
            array.erf(<float *> a.data, len(a))
        elif a.dtype == np.float64:
            array.erf(<double *> a.data, len(a))
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def exp(self, np.ndarray a):
        cdef SleefArray array = self._array

        a = self._to_contig_or_copy(a)
        if a.dtype == np.float32:
            array.exp(<float *> a.data, len(a))
        elif a.dtype == np.float64:
            array.exp(<double *> a.data, len(a))
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def tanh(self, np.ndarray a):
        cdef SleefArray array = self._array

        a = self._to_contig_or_copy(a)
        if a.dtype == np.float32:
            array.tanh(<float *> a.data, len(a))
        elif a.dtype == np.float64:
            array.tanh(<double *> a.data, len(a))
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def _to_contig_or_copy(self, np.ndarray a):
        cdef np.ndarray a_contig = self.as_contig(a)

        if a_contig is a:
            a = a.copy()
        else:
            a = a_contig

        return a

@contextmanager
def with_cpu_feature(InstructionSet feature):
    ops = SleefOps()
    with sleef_with_cpu_feature(feature) as a:
        ops._array = a
        yield ops
