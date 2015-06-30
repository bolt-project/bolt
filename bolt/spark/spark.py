from numpy import asarray, unravel_index, arange, prod, mod, divide
from bolt.common import slicify
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

    def __getitem__(self, index):

        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        # this should turn a slice for any index that was a slice or single number
        # and a list of indecies to include if a list of ints or a boolean array
        index = tuple([slicify(s, d) for (s, d) in zip(index, self.shape)])

        key_slices = index[0:self.split]
        value_slices = tuple([list(s) if isinstance(s,set) else s for s in index[self.split:]])
        
        def key_check(key):
            def check(kk, ss):
                if isinstance(ss, slice):
                    return ss.start <= kk < ss.stop and mod(kk - ss.start, ss.step) == 0
                elif isinstance(ss, set):
                    return kk in ss
            out = [check(k, s) for k, s in zip(key, key_slices)]
            return all(out)

        def key_func(key):
            return tuple([k - s.start for k, s in zip(key, key_slices)])

        def value_func(value):
            return value[value_slices]

        filtered = self._rdd.filter(lambda (k, v): key_check(k))
        mapped = filtered.map(lambda (k, v): (key_func(k), value_func(v)))

        shape = []
        for s in index:
            if isinstance(s, slice):
                shape.append(divide(s.stop - s.start, s.step) + mod(s.stop - s.start, s.step))
            elif isinstance(s, list):
                shape.append(len(s))
            elif isinstance(s, set):
                shape.append(len(s))
        shape = tuple(shape)
        return self._constructor(mapped, shape=shape).__finalize__(self)

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
        from bolt.local.local import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.sortByKey().values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        return self._rdd

    def display(self):
        for x in self._rdd.take(10):
            print x
