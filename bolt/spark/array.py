from numpy import asarray, unravel_index, prod, mod, ndarray, ceil,  r_, int16
from itertools import groupby

from bolt.spark.utils import slicify, listify
from bolt.base import BoltArray
from bolt.mixins.stacked import Stackable
from bolt.utils import check_axes


class BoltArraySpark(BoltArray, Stackable):

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

        key_shape = shape[:split]
        val_shape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(key_shape))), key_shape))
        vals = arry.reshape((prod(key_shape),) + val_shape)

        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd, shape=shape, split=split)

    def __array__(self):
        return self.toarray()

    """
    StackedBoltArray interface

    The underscored methods should only be invoked using the StackedBoltArray provided via the
    'stacked' method.
    """

    def _stack(self, stack_size=None):

        def partition_to_stacks(part_iter):
            cur_keys = []
            cur_arrs = []
            for key, arr in part_iter:
                cur_keys.append(key)
                cur_arrs.append(arr)
                if stack_size and stack_size >= 0 and len(cur_keys) >= stack_size:
                    yield (cur_keys, asarray(cur_arrs))
                    cur_keys, cur_arrs = [], []
            if cur_keys:
                yield (cur_keys, asarray(cur_arrs))

        return self._constructor(self._rdd.mapPartitions(partition_to_stacks),
                shape=self.shape, split=self.split)

    def _unstack(self):
        return self._constructor(self._rdd.flatMap(lambda (keys, arr): zip(keys, list(arr))),
                shape=self.shape, split=self.split)

    """
    Functional operators
    """

    def _configure_key_axes(self, key_axes):
        """
        The common prefix for functional operations:
            1) Ensures that the specified axes are valid
            2) Swaps key/value axes if necessary so that the underlying RDD operation is applied to the correct records

        Parameters
        ---------
        key_axes: tuple[int]
            axes that wil be iterated over during the application of a functional operator
        """
        # Ensure that the specified axes are valid
        check_axes(self.shape, key_axes)

        axis_set = set(key_axes)

        split = self.split

        # Find the value axes that should be moved into the keys (axis >= split)
        to_keys = [(a - split) for a in key_axes if a >= split]

        # Find the key axes that should be moved into the values (axis < split)
        to_values = [a for a in range(split) if a not in axis_set]

        if to_keys or to_values:
            return self.swap(to_values, to_keys)
        return self


    def map(self, func, axes=(0,)):
        """
        Applies a function to every element across the specified axis.

        What about the scenario when a map returns an ndarray per mapped element??

        TODO: Better docstring
        """

        axes = sorted(axes)
        swapped = self._configure_key_axes(axes)

        # Try to compute the size of each mapped element by applying func to a random array
        element_shape = None
        try:
            element_shape = func(random.randn(*[swapped._shape[axis] for axis in axes])).shape
        except Exception:
            print "Failed to compute the shape of the result using a test array. Retrying with Spark job."
            first_elem = swapped._rdd.first()
            if first_elem:
                # Run the function on the first element of the current (pre-mapped) RDD to see if it fails
                first_mapped = func(first_elem[1])
                element_shape = first_mapped.shape

        rdd = swapped._rdd.mapValues(func)

        # Reshaping will fail if the elements aren't uniformly shaped (is this necessary?)
        def checkShape(v):
            if v.shape != element_shape:
                raise Exception("Map operation did not produce values of uniform shape.")
            return v
        rdd = rdd.mapValues(lambda v: checkShape(v))
        shape = tuple([swapped._shape[axis] for axis in axes] + list(element_shape))

        return self._constructor(rdd, shape=shape, split=swapped.split).__finalize__(swapped)

    @staticmethod
    def _zipWithIndex(rdd):
        """
        A lightly modified version of Spark's RDD.zipWithIndex that eagerly returns the RDD's count along with the
        zipped RDD.
        """
        starts = [0]

        count = None
        if rdd.getNumPartitions() > 1:
          nums = rdd.mapPartitions(lambda it: [sum(1 for i in it)]).collect()
          count = sum(nums)
          for i in range(len(nums) - 1):
              starts.append(starts[-1] + nums[i])

        def func(k, it):
          for i, v in enumerate(it, starts[k]):
              yield v, i

        return count, rdd.mapPartitionsWithIndex(func)

    def filter(self, func, axes=(0,)):
        """

        Filter must do a count in order to get the shape, followed by a re-keying

        (x, y) -> (a, b)
        filter(func, axes=(0,2))
        (x, a) -> (y, b)

        Since arbitrary rows can be filtered out, the keys are just linearized after the filter.

        TODO: Better docstring
        """

        if len(axes) != 1:
            print "Filtering over multiple axes will not be supported until SparseBoltArray is implemented."
            raise NotImplementedError

        axes = sorted(axes)
        swapped = self._configure_key_axes(axes)

        rdd = swapped._rdd.values().filter(func)

        # Count the resulting array in order to reindex (linearize) the keys
        count, zipped = BoltArraySpark._zipWithIndex(rdd)
        if not count:
            count = zipped.count()
        reindexed = zipped.map(lambda (k, v): (v, k))

        remaining = [swapped.shape[dim] for dim in range(len(swapped.shape)) if dim not in axes]
        shape = None
        if count != 0:
            shape = tuple([count] + remaining)
        else:
            shape = (0,)

        return self._constructor(reindexed, shape=shape, split=swapped.split).__finalize__(swapped)

    def reduce(self, func, axes=(0,)):
        """

        Simple case:
        - (x, y, a, b) -> (5, 10, 15, 20)
        - reduce(func, axes=(0, 1))
        - (1, a, b) -> (1, 15, 20)

        Complicated case:
        - (x, y, a, b) -> (5, 10, 15, 20)
        - reduce(func, axes=(0, 2))
        - (x, y, a, b) -> (x, a, y, b)
        - (1, y, b) -> (1, 10, 20)

        newshape = (1, (shape of non-reduced axes))
        The ordering of the non-reduced axes is maintained after the reduce

        TODO: Better docstring
        """

        from bolt.local.array import BoltArrayLocal

        axes = sorted(axes)

        swapped = self._configure_key_axes(axes)

        arr = swapped._rdd.values().reduce(func)

        return BoltArrayLocal(arr)

    """
    Reductions
    """

    def sum(self, axes=(0,)):
        from numpy import sum
        return self._constructor(self.reduce(sum, axes)).__finalize__(self)

    def mean(self, axes=(0,)):
        from numpy import mean
        return self._constructor(self.reduce(mean, axes)).__finalize__(self)

    def max(self, axes=(0,)):
        from numpy import max
        return self._constructor(self.reduce(max, axes)).__finalize__(self)

    def min(self, axes=(0,)):
        from numpy import min
        return self._constructor(self.reduce(min, axes)).__finalize__(self)

    def collect(self):
        return self._rdd.collect()

    # TODO add axes
    def sum(self, axis=0):
        return self._constructor(self._rdd.sum()).__finalize__(self)

    def concatenate(self, arry, axis=0):
        """
        Concatenate with another bolt spark array
        """
        if isinstance(arry, ndarray):
            from bolt.spark.construct import ConstructSpark
            arry = ConstructSpark.array(arry, self._rdd.context, axes=range(0, self.split))
        else:
            if not isinstance(arry, BoltArraySpark):
                raise ValueError("other must be local array or spark array, got %s" % type(arry))

        if not all([x == y if not i == axis else True
                    for i, (x, y) in enumerate(zip(self.shape, arry.shape))]):
            raise ValueError("all the input array dimensions except for "
                             "the concatenation axis must match exactly")

        if not self.split == arry.split:
            raise NotImplementedError("two arrays must have the same split ")

        if axis < self.split:
            shape = self.keys.shape

            def key_func(key):
                key = list(key)
                key[axis] += shape[axis]
                return tuple(key)

            rdd = self._rdd.union(arry._rdd.map(lambda (k, v): (key_func(k), v)))

        else:
            from numpy import concatenate as npconcatenate
            shift = axis - self.split
            rdd = self._rdd.join(arry._rdd).map(lambda (k, v): (k, npconcatenate(v, axis=shift)))

        shape = tuple([x + y if i == axis else x
                      for i, (x, y) in enumerate(zip(self.shape, arry.shape))])

        return self._constructor(rdd, shape=shape).__finalize__(self)

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

    def swap(self, key_axes, value_axes, size=150):

        from bolt.spark.chunks import Chunks

        if len(key_axes) == self.keys.shape:
            raise ValueError('Cannot perform a swap that would end up with all data on a single key')

        # TODO: once self.dtype is implemented, change int16 to self.dtype
        c = Chunks(key_axes, value_axes, self.keys.shape, self.values.shape, int16, size)

        chunks = c.chunk(self._rdd)
        rdd = c.extract(chunks)

        shape = r_[c.key_shape[~c.key_mask], c.value_shape[c.value_mask],
                   c.key_shape[c.key_mask], c.value_shape[~c.value_mask]]
        split = self.split - len(key_axes) + len(value_axes)

        return self._constructor(rdd, shape=tuple(shape), split=split)

    def chunk(self, key_axes, value_axes, size):

        from bolt.spark.chunks import Chunks

        # TODO: once self.dtype is implemented, change int16 to self.dtype
        c = Chunks(key_axes, value_axes, self.keys.shape, self.values.shape, int16, size)
        return c.chunk(self._rdd)

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


