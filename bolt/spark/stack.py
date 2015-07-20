from numpy import asarray, ndarray, concatenate
from bolt.spark.utils import zip_with_index

class StackedArray(object):
    """
    Wraps a BoltArraySpark and provides an interface for performing
    stacked operations (operations on aggregated subarrays). Many methods
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
    _metadata = ['_rdd', '_shape', '_split', '_rekeyed']

    def __init__(self, rdd, shape=None, split=None, rekeyed=False):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._rekeyed = rekeyed

    def __finalize__(self, other):
        for name in self._metadata:
            other_attr = getattr(other, name, None)
            if (other_attr is not None) and (getattr(self, name, None) is None):
                object.__setattr__(self, name, other_attr)
        return self

    @property
    def shape(self):
        return self._shape

    @property
    def split(self):
        return self._split

    @property
    def rekey(self):
        return self._rekeyed

    @property
    def _constructor(self):
        return StackedArray

    def stack(self, size):
        """
        Make an intermediate RDD where all records are combined into a
        list of keys and larger ndarray along a new 0th dimension.
        """
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

        if self._rekeyed:
            rdd = self._rdd
        else:
            rdd = self._rdd.flatMap(lambda kv: zip(kv[0], list(kv[1])))

        return BoltArraySpark(rdd, shape=self.shape, split=self.split)

    def map(self, func):
        """
        Apply a function on each subarray.

        Parameters
        ----------
        func : function 
             This is applied to each value in the intermediate RDD.

        Returns
        -------
        StackedArray
        """
        vshape = self.shape[self.split:]
        x = self._rdd.values().first()
        if x.shape == vshape:
            a, b = asarray([x]), asarray([x, x])
        else:
            a, b = x, concatenate((x, x))

        try:
            atest = func(a)
            btest = func(b)
        except Exception as e:
            raise RuntimeError("Error evaluating function on test array, got error:\n %s" % e)

        if not (isinstance(atest, ndarray) and isinstance(btest, ndarray)):
            raise ValueError("Function must return ndarray")

        # different shapes map to the same new shape
        elif atest.shape == btest.shape:
            if self._rekeyed is True:
                # we've already rekeyed
                rdd = self._rdd.map(lambda kv: (kv[0], func(kv[1])))
                shape = (self.shape[0],) + atest.shape
            else:
                # do the rekeying
                count, rdd = zip_with_index(self._rdd.values())
                rdd = rdd.map(lambda kv: ((kv[1],), func(kv[0])))
                shape = (count,) + atest.shape
            split = 1
            rekeyed = True

        # different shapes stay different (along the first dimension)
        elif atest.shape[0] == a.shape[0] and btest.shape[0] == b.shape[0]:
            shape = self.shape[0:self.split] + atest.shape[1:]
            split = self.split
            rdd = self._rdd.map(lambda kv: (kv[0], func(kv[1])))
            rekeyed = self._rekeyed

        else:
            raise ValueError("Cannot infer effect of function on shape")

        return self._constructor(rdd, rekeyed=rekeyed, shape=shape, split=split).__finalize__(self)

    def tordd(self):
        """
        Return the RDD wrapped by the StackedArray.

        Returns
        -------
        RDD
        """
        return self._rdd

    def __str__(self):
        s = "Stacked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        return s

    def __repr__(self):
        return str(self)
