from numpy import ndarray, asarray
from bolt.base import BoltArray


class BoltArrayLocal(ndarray, BoltArray):

    def __new__(cls, array):
        obj = asarray(array).view(cls)
        obj._mode = 'local'
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._mode = getattr(obj, 'mode', None)

    @property
    def _constructor(self):
        return BoltArrayLocal

    """
    Functional operators
    """

    def map(self, func):
        return self._constructor([func(x) for x in self])

    def reduce(self, func):
        return reduce(func, self)

    """
    Conversions
    """

    def tospark(self, sc):
        from bolt.spark import BoltArraySpark
        return BoltArraySpark.fromarray(self, sc)

    def tonumpy(self):
        return asarray(self)

    def display(self):
        return str(self)

    def __repr__(self):
        return BoltArray.__repr__(self)