from numpy import asarray
from bolt.base import BoltArray


class BoltArraySpark(BoltArray):

    def __init__(self, rdd):
        self._rdd = rdd
        self._mode = 'spark'

    @property
    def _constructor(self):
        return BoltArraySpark

    @staticmethod
    def fromarray(array, context):
        keys = xrange(0, len(array))
        vals = [asarray(x) for x in array]
        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd)

    """
    Functional operators
    """

    def map(self, func):
        return self._constructor(self._rdd.map(func))

    def reduce(self, func):
        return self._constructor(self._rdd.reduce(func))

    """
    Array operators
    """

    def sum(self, axis=0):
        return self._constructor(self._rdd.sum())

    def __getitem(self):
        pass

    """
    Conversions
    """

    def tolocal(self):
        from bolt.local import BoltArrayLocal
        return BoltArrayLocal(asarray(self._rdd.collect()))

    def toarray(self):
        return asarray(self._rdd.collect())

    def display(self):
        return str(asarray(self._rdd.take(10)))

    def __str__(self):
        return str(asarray(self._rdd.collect()))