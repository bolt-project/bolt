from numpy import asarray, unravel_index, ravel_multi_index, arange, prod
from bolt.base import BoltArray


class BoltArraySpark(BoltArray):

    _metadata = BoltArray._metadata + ['_shape', '_split']

    def __init__(self, rdd, shape=None, split=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._mode = 'spark'

    @property
    def _constructor(self):
        return BoltArraySpark

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

    # TODO make sure that operation preserves shape
    def map(self, func):
        return self._constructor(self._rdd.mapValues(func)).__finalize__(self)

    def reduce(self, func):
        return self._constructor(self._rdd.values().reduce(func)).__finalize__(self)

    """
    Basic array operators
    """

    def sum(self, axis=0):
        return self._constructor(self._rdd.sum()).__finalize__(self)

    def __getitem__(self):
        pass

    """
    Shaping operators
    """

    @property
    def shape(self):
        return self._shape

    @property
    def split(self):
        return self._split

    @property
    def parts(self):
        return tuple([0] * len(self._keyShape) + [1] * len(self._valueShape))

    @property
    def _keyShape(self):
        return self.shape[:self.split]

    @property
    def _valueShape(self):
        return self.shape[self.split:]

    def reshapeKeys(self, *new):

        new = tuple(new)
        old = self._keyShape

        if not prod(new) == prod(old):
            raise ValueError("Total size of new keys must remain unchanged")

        def f(k):
            return unravel_index(ravel_multi_index(k, old), new)

        newrdd = self._rdd.map(lambda (k, v): (f(k), v))
        newsplit = len(new)
        newshape = new + self._valueShape

        return self._constructor(newrdd, shape=newshape, split=newsplit)

    def reshapeValues(self, *new):

        new = tuple(new)
        old = self._valueShape

        if not prod(new) == prod(old):
            raise ValueError("Total size of new values must remain unchanged")

        def f(v):
            return v.reshape(new)

        newrdd = self._rdd.mapValues(f)
        newshape = self._keyShape + new

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    def transposeKeys(self, *new):

        new = tuple(new)
        old = self._keyShape

        if not len(new) == len(old):
            raise ValueError("Axes do not match axes of keys")

        def f(k):
            return tuple(k[i] for i in new)

        newrdd = self._rdd.map(lambda (k, v): (f(k), v))
        newshape = tuple(old[i] for i in new) + self._valueShape

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    def transposeValues(self, *new):

        new = tuple(new)
        old = self._valueShape

        if not len(new) == len(old):
            raise ValueError("Axes do not match axes of values")

        def f(v):
            return v.reshape(new)

        newrdd = self._rdd.mapValues(f)
        newshape = self._keyShape + tuple(old[i] for i in new)

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    """
    Conversions
    """

    def tolocal(self):
        from bolt.local import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.values().collect()
        return asarray(x).reshape(self.shape)

    def display(self):
        return str(asarray(self._rdd.take(10)))

    def __str__(self):
        return str(asarray(self._rdd.collect()))