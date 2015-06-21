from numpy import asarray
from bolt.base import BoltArray


class BoltArrayRDD(BoltArray):

    def __init__(self, array, context):
        self._rdd = context.parallelize(array)
        self._context = context
        self._mode = 'spark'

    def sum(self):
        return self._rdd.sum()

    def tolocal(self):
        from bolt.local import BoltArrayLocal
        return BoltArrayLocal(asarray(self._rdd.collect()))

    def display(self):
        return str(asarray(self._rdd.take(10)))

    def __str__(self):
        return str(asarray(self._rdd.collect()))