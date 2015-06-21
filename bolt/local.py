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

    def tordd(self, sc):
        from bolt.rdd import BoltArrayRDD
        return BoltArrayRDD(self, sc)

    def display(self):
        return str(self)

    def __repr__(self):
        return BoltArray.__repr__(self)