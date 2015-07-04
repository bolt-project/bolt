from numpy import asarray

class StackedArray(object):
    """
    Wraps a BoltArraySpark and provides an interface for performing stacked operations
    (operations on whole subarrays). Many methods will be restricted or forbidden until the
    Stacked object is unstacked.
    """
    def __init__(self, rdd, shape=None, split=None, stack_size=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self.stack_size = stack_size

    def __finalize__(self, other):
        self._shape = other._shape
        self._split = other._split
        self.stack_size = other.stack_size
        return self

    @property
    def shape(self):
        return self._shape

    @property
    def split(self):
        return self._split

    @property
    def _constructor(self):
        return StackedArray

    def _stack(self):

        stack_size = self.stack_size

        def tostacks(partition):
            keys = []
            arrs = []
            for key, arr in partition:
                keys.append(key)
                arrs.append(arr)
                if stack_size and 0 <= stack_size <= len(keys):
                    yield (keys, asarray(arrs))
                    keys, arrs = [], []
            if keys:
                yield (keys, asarray(arrs))

        rdd = self._rdd.mapPartitions(tostacks)
        return self._constructor(rdd).__finalize__(self)

    def unstack(self):
        from bolt.spark.array import BoltArraySpark
        return BoltArraySpark(self._rdd.flatMap(lambda kv: zip(kv[0], list(kv[1]))),
                              shape=self.shape, split=self.split)

    def map(self, func):
        rdd = self._rdd.map(lambda kv: (kv[0], func(kv[1])))
        return self._constructor(rdd).__finalize__(self)

    def __str__(self):
        s = "Stacked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        s += "stack size: %s" % str(self.stack_size)
        return s

    def __repr__(self):
        return str(self)