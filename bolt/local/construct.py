from numpy import float64, asarray

from bolt.construct import ConstructBase
from bolt.local.array import BoltArrayLocal


class ConstructLocal(ConstructBase):

    @staticmethod
    def array(arry):
        return BoltArrayLocal(asarray(arry))

    @staticmethod
    def ones(shape, dtype=float64, order='C'):
        from numpy import ones
        return ConstructLocal._wrap(ones, shape, dtype, order)

    @staticmethod
    def zeros(shape, dtype=float64, order='C'):
        from numpy import zeros
        return ConstructLocal._wrap(zeros, shape, dtype, order)

    @staticmethod
    def _wrap(func, shape, dtype, order):
        return BoltArrayLocal(func(shape, dtype, order))
