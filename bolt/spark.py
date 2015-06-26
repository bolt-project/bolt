from numpy import asarray, unravel_index, ravel_multi_index, arange, prod
from bolt.common import tupleize
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

    # TODO handle shape changes
    # TODO add axes
    def map(self, func):
        return self._constructor(self._rdd.mapValues(func)).__finalize__(self)

    # TODO add axes
    def reduce(self, func):
        return self._constructor(self._rdd.values().reduce(func)).__finalize__(self)

    def collect(self):
        return self._rdd.collect()

    """
    Reductions
    """

    # TODO add axes
    def sum(self, axis=0):
        return self._constructor(self._rdd.sum()).__finalize__(self)

    """
    Slicing and indexing
    """

    def __getitem__(self):
        pass

    """
    Shaping operators
    """

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return prod(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def split(self):
        return self._split

    @property
    def mask(self):
        return tuple([1] * len(self.keys.shape) + [0] * len(self.values.shape))

    @property
    def keys(self):
        return BoltArraySpark._Keys(self)

    @property
    def values(self):
        return BoltArraySpark._Values(self)

    class _Shapes(object):

        @property
        def shape(self):
            raise NotImplementedError

        def reshape(self):
            raise NotImplementedError

        def transpose(self):
            raise NotImplementedError

        @staticmethod
        def _istransposeable(new, old):

            if not len(new) == len(old):
                raise ValueError("Axes do not match axes of keys")

            if not len(set(new)) == len(set(old)):
                raise ValueError("Repeated axes")

            if any(n < 0 for n in new) or max(new) > len(old) - 1:
                raise ValueError("Invalid axes")

        @staticmethod
        def _isreshapable(new, old):

            if not prod(new) == prod(old):
                raise ValueError("Total size of new keys must remain unchanged")

    class _Keys(_Shapes):

        def __init__(self, barray):
            self._barray = barray

        @property
        def shape(self):
            return self._barray.shape[:self._barray.split]

        def reshape(self, *new):

            new = tupleize(new)
            old = self.shape
            self._isreshapable(new, old)

            if new == old:
                return self._barray

            def f(k):
                return unravel_index(ravel_multi_index(k, old), new)

            newrdd = self._barray._rdd.map(lambda (k, v): (f(k), v))
            newsplit = len(new)
            newshape = new + self._barray.values.shape

            return BoltArraySpark(newrdd, shape=newshape, split=newsplit)

        def transpose(self, *new):

            new = tupleize(new)
            old = self.shape
            self._istransposeable(new, old)

            if new == range(0, len(old)):
                return self._barray

            def f(k):
                return tuple(k[i] for i in new)

            newrdd = self._barray._rdd.map(lambda (k, v): (f(k), v))
            newshape = tuple(old[i] for i in new) + self._barray.values.shape

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self)

        def __str__(self):
            s = "BoltArray Keys\n"
            s += "shape: %s" % str(self.shape)
            return s

        def __repr__(self):
            return str(self)

    class _Values(_Shapes):

        def __init__(self, barray):
            self._barray = barray

        @property
        def shape(self):
            return self._barray.shape[self._barray.split:]

        def reshape(self, *new):

            new = tupleize(new)
            old = self.shape
            self._isreshapable(new, old)

            if new == old:
                return self._barray

            def f(v):
                return v.reshape(new)

            newrdd = self._barray._rdd.mapValues(f)
            newshape = self._barray.keys.shape + new

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self)

        def transpose(self, *new):

            new = tupleize(new)
            old = self.shape
            self._istransposeable(new, old)

            if new == range(0, len(old)):
                return self._barray

            def f(v):
                return v.transpose(new)

            newrdd = self._barray._rdd.mapValues(f)
            newshape = self._barray.keys.shape + tuple(old[i] for i in new)

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self)

        def __str__(self):
            s = "BoltArray Values\n"
            s += "shape: %s" % str(self.shape)
            return s

        def __repr__(self):
            return str(self)

    """
    Conversions
    """

    def tolocal(self):
        from bolt.local import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.sortByKey().values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        return self._rdd

    def display(self):
        for x in self._rdd.take(10):
            print x
