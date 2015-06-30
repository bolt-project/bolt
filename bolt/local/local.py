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

    def map(self, func):
        return self._constructor([func(x) for x in self])

    def reduce(self, func):
        return reduce(func, self)

    def tospark(self, sc, split=1):
        from bolt import array
        return array(self.toarray(), sc, split)

    def tordd(self, sc, split=1):
        from bolt import array
        return array(self.toarray(), sc, split).tordd()

    def toarray(self):
        return asarray(self)

    def display(self):
        print str(self)

    def __repr__(self):
        return BoltArray.__repr__(self)
