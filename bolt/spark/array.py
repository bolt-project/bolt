from numpy import asarray, unravel_index, prod, mod, ndarray, ceil
from itertools import groupby

from bolt.spark.utils import slicify, listify
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

    # TODO handle shape changes
    # TODO add axes
    def map(self, func):
        return self._constructor(self._rdd.mapValues(func)).__finalize__(self)

    # TODO add axes
    def reduce(self, func):
        return self._constructor(self._rdd.values().reduce(func)).__finalize__(self)

    def collect(self):
        return self._rdd.collect()

    # TODO add axes
    def sum(self, axis=0):
        return self._constructor(self._rdd.sum()).__finalize__(self)

    def getbasic(self, index):
        """
        Basic indexing
        """
        index = tuple([slicify(s, d) for (s, d) in zip(index, self.shape)])
        key_slices = index[0:self.split]
        value_slices = index[self.split:]

        def key_check(key):
            check = lambda kk, ss: ss.start <= kk < ss.stop and mod(kk - ss.start, ss.step) == 0
            out = [check(k, s) for k, s in zip(key, key_slices)]
            return all(out)

        def key_func(key):
            return tuple([(k - s.start)/s.step for k, s in zip(key, key_slices)])

        filtered = self._rdd.filter(lambda (k, v): key_check(k))
        rdd = filtered.map(lambda (k, v): (key_func(k), v[value_slices]))
        shape = tuple([int(ceil((s.stop - s.start) / float(s.step))) for s in index])
        split = self.split
        return rdd, shape, split

    def getadvanced(self, index):
        """
        Advanced indexing
        """
        index = [asarray(i) for i in index]
        shape = index[0].shape
        if not all([i.shape == shape for i in index]):
            raise ValueError("shape mismatch: indexing arrays could not be broadcast "
                             "together with shapes " +
                             ("%s " * self.ndim) % tuple([i.shape for i in index]))

        index = tuple([listify(i, d) for (i, d) in zip(index, self.shape)])

        key_tuples = zip(*index[0:self.split])
        value_tuples = zip(*index[self.split:])

        d = {}
        for k, g in groupby(zip(value_tuples, key_tuples), lambda x: x[1]):
            d[k] = map(lambda x: x[0], list(g))

        def key_check(key):
            return key in key_tuples

        def key_func(key):
            return unravel_index(key, shape)

        filtered = self._rdd.filter(lambda (k, v): key_check(k))
        flattened = filtered.flatMap(lambda (k, v): [(k, v[i]) for i in d[k]])
        indexed = flattened.zipWithIndex()
        rdd = indexed.map(lambda ((_, v), ind): (key_func(ind), v))
        split = len(shape)
        return rdd, shape, split

    def __getitem__(self, index):

        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if not all([isinstance(i, (slice, int, list, set, ndarray)) for i in index]):
            raise ValueError("Each index must either be a slice, int, list, set, or ndarray")

        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        index = tuple([i[0] if isinstance(i, list) and len(i) == 1 else i for i in index])

        if all([isinstance(i, (slice, int)) for i in index]):
            rdd, shape, split = self.getbasic(index)

        elif all([isinstance(i, (set, list, ndarray)) for i in index]):
            rdd, shape, split = self.getadvanced(index)

        else:
            raise NotImplementedError("Cannot mix basic indexing (slices and ints) with "
                                      "advanced indexing (lists and ndarrays) across axes")

        return self._constructor(rdd, shape=shape, split=split).__finalize__(self)

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
        from bolt.spark.shapes import Keys
        return Keys(self)

    @property
    def values(self):
        from bolt.spark.shapes import Values
        return Values(self)

    def tolocal(self):
        from bolt.local.array import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.sortByKey().values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        return self._rdd

    def display(self):
        for x in self._rdd.take(10):
            print x
