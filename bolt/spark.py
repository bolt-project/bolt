from numpy import asarray, unravel_index, arange, prod
from bolt.base import BoltArray


class BoltArraySpark(BoltArray):

    def __init__(self, rdd, shape, split):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._mode = 'spark'

    @property
    def _constructor(self):
        return BoltArraySpark

    @property
    def shape(self):
        return self._shape

    @property
    def _keyShape(self):
        return self.shape[:self._split]

    @property
    def _valShape(self):
        return self.shape[self._split:]

    @staticmethod
    def fromarray(arry, context, split=1):

        shape = arry.shape
        ndim = len(shape)

        if split < 1:
            raise ValueError("Split axis must be greater than 0, got %g" % split)
        if split > len(shape):
            raise ValueError("Split axis must not exceed number of axes %g, got %g" % (ndim, split))

        keyShape = shape[:split]
        valShape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(keyShape))), keyShape))
        vals = arry.reshape((prod(keyShape),) + valShape)

        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd, shape=shape, split=split)

    """
    Functional operators
    """

    def map(self, func):
        return self._constructor(self._rdd.map(func))

    def reduce(self, func):
        return self._constructor(self._rdd.reduce(func))

    """
    Basic array operators
    """

    def sum(self, axis=0):
        return self._constructor(self._rdd.sum())

    def __getitem__(self):
        pass

    """
    Shaping operators
    """

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