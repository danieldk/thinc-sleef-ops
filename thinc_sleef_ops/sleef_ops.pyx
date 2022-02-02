# cython: cdivision=True
# cython: infer_types=True
# cython: profile=True

from contextlib import contextmanager
cimport numpy as np
import numpy as np
from thinc.api import Ops

try:
    from thinc_apple_ops import AppleOps
    ops_superclass = AppleOps
except ImportError:
    ops_superclass = Ops

from .sleef_array cimport InstructionSet, SleefArray
from .sleef_array import with_cpu_feature as sleef_with_cpu_feature

class SleefOps(ops_superclass):
    def __init__(self):
        self._array = SleefArray()

    @staticmethod
    def instruction_sets():
        return SleefArray.instruction_sets()

    def erf(self, a: np.ndarray, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.erf(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.erf(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def exp(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.exp(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.exp(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def gelu(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.gelu(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.gelu(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def gelu_backward(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.gelu_backward(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.gelu_backward(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def sigmoid(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.logistic_cdf(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.logistic_cdf(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def swish(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.swish(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.swish(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def swish_backward(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.swish_backward(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.swish_backward(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def tanh(self, np.ndarray a, *, inplace: bool=False):
        cdef SleefArray array = self._array
        cdef size_t n = a.size

        a = self._to_contig_or_copy(a, inplace=inplace)
        if a.dtype == np.float32:
            array.tanh(<float *> a.data, n)
        elif a.dtype == np.float64:
            array.tanh(<double *> a.data, n)
        else:
            raise TypeError("Unhandled array dtype")

        return a

    def _to_contig_or_copy(self, np.ndarray a, *, inplace: bool=False):
        is_contiguous = a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]

        if inplace:
           if is_contiguous:
               return a
           else:
               raise "Cannot apply operation in-place, array is not contiguous"

        if is_contiguous:
            return a.copy()

        return self.as_contig(a)


@contextmanager
def with_cpu_feature(InstructionSet feature):
    ops = SleefOps()
    with sleef_with_cpu_feature(feature) as a:
        ops._array = a
        yield ops
