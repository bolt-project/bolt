from numpy import asarray

class StackedArray(object):
    """
    Wraps a BoltArraySpark and provides an interface for performing
    stacked operations (operations on whole subarrays). Many methods
    will be restricted or forbidden until the Stacked object is
    unstacked. Currently, only map() is implemented. The rationale
    is that many operations will work faster when vectorized over a
    slightly larger array.

    The implementation uses an intermediate RDD that collects all
    records on a given partition into 'stacked' (key, value) records.
    Here, a key is a 'size' long tuple of original record keys,
    and and values is a an array of the corresponding values,
    concatenated along a new 0th dimenion.

    """
    def __init__(self, rdd, shape=None, split=None, size=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self.size = size

    def __finalize__(self, other):
        self._shape = other._shape
        self._split = other._split
        self.size = other.size
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
        """
        Make an intermediate RDD where all records are combined into a
        list of keys and larger ndarray along a new 0th dimension.
        """
        size = self.size

        def tostacks(partition):
            keys = []
            arrs = []
            for key, arr in partition:
                keys.append(key)
                arrs.append(arr)
                if size and 0 <= size <= len(keys):
                    yield (keys, asarray(arrs))
                    keys, arrs = [], []
            if keys:
                yield (keys, asarray(arrs))

        rdd = self._rdd.mapPartitions(tostacks)
        return self._constructor(rdd).__finalize__(self)

    def unstack(self):
        """
        Unstack array and return a new BoltArraySpark via flatMap().
        """

        from bolt.spark.array import BoltArraySpark
        return BoltArraySpark(self._rdd.flatMap(lambda kv: zip(kv[0], list(kv[1]))),
                              shape=self.shape, split=self.split)

    def map(self, func):
        """
        Apply a function on each subarray.

        Parameters
        ----------
        func : function 
             This is applied to each value in the intermediate RDD.
        """
        rdd = self._rdd.map(lambda kv: (kv[0], func(kv[1])))
        return self._constructor(rdd).__finalize__(self)

    def __str__(self):
        s = "Stacked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        s += "stack size: %s" % str(self.size)
        return s

    def __repr__(self):
        return str(self)
