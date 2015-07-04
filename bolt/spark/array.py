from __future__ import print_function
from numpy import asarray, unravel_index, prod, mod, ndarray, ceil, where, r_, sort, argsort
from itertools import groupby

from bolt.base import BoltArray
from bolt.spark.stack import StackedArray
from bolt.spark.utils import zip_with_index
from bolt.spark.statcounter import StatCounter
from bolt.utils import slicify, listify, tupleize, argpack, check_axes


class BoltArraySpark(BoltArray):

    _metadata = BoltArray._metadata + ['_shape', '_split', '_dtype']

    def __init__(self, rdd, shape=None, split=None, dtype=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._dtype = dtype
        self._mode = 'spark'

    @property
    def _constructor(self):
        return BoltArraySpark

    def __array__(self):
        return self.toarray()

    def cache(self):
        self._rdd.cache()

    def unpersist(self):
        self._rdd.unpersist()

    def stack(self, stack_size=None):
        """
        Aggregates records of a distributed array.

        Stacking should improve the performance of vectorized operations,
        but the resulting Stacked object only exposes a restricted set
        of operations (e.g. map, reduce). The unstack method can be used
        to restore the full bolt array.

        Parameters
        ----------
        stack_size: int, optional, default=None
            The maximum size for each stack (number of original records),
            will aggregate groups of records per partition up to this size.

        Returns
        -------
        Stacked
        """
        stk = StackedArray(self._rdd, shape=self.shape, split=self.split, stack_size=stack_size)
        return stk._stack()

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
        # ensure that the specified axes are valid
        check_axes(self.shape, key_axes)

        axis_set = set(key_axes)

        split = self.split

        # find the value axes that should be moved into the keys (axis >= split)
        to_keys = [(a - split) for a in key_axes if a >= split]

        # find the key axes that should be moved into the values (axis < split)
        to_values = [a for a in range(split) if a not in axis_set]

        if to_keys or to_values:
            return self.swap(to_values, to_keys)
        return self

    def map(self, func, axes=(0,), dtype=None):
        """
        Applies a function to every element across the specified axis.

        What about the scenario when a map returns an ndarray per mapped element??

        TODO: Better docstring
        """
        # this check is to handle using self.map to implement astype(),
        # where we have to pass in the new dtype to _constructor()
        # any other case we don't use the parameter, so we set it to
        # the object's current attribute
        if dtype is None:
            dtype = self._dtype

        axes = sorted(axes)
        swapped = self._configure_key_axes(axes)

        # Try to compute the size of each mapped element by applying func to a random array
        element_shape = None
        try:
            element_shape = func(random.randn(*[swapped._shape[axis] for axis in axes])).shape
        except Exception:
            first_elem = swapped._rdd.first()
            if first_elem:
                # Run the function on the first element of the current (pre-mapped) RDD to see if it fails
                first_mapped = func(first_elem[1])
                element_shape = first_mapped.shape

        rdd = swapped._rdd.mapValues(func)

        # reshaping will fail if the elements aren't uniformly shaped
        def check(v):
            if v.shape != element_shape:
                raise Exception("Map operation did not produce values of uniform shape.")
            return v
        rdd = rdd.mapValues(lambda v: check(v))
        shape = tuple([swapped._shape[axis] for axis in axes] + list(element_shape))

        return self._constructor(rdd, shape=shape, split=swapped.split, dtype=dtype).__finalize__(swapped)

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
            print("Filtering over multiple axes will not be supported until SparseBoltArray is implemented.")
            raise NotImplementedError

        axes = sorted(axes)
        swapped = self._configure_key_axes(axes)

        rdd = swapped._rdd.values().filter(func)

        # count the resulting array in order to reindex (linearize) the keys
        count, zipped = zip_with_index(rdd)
        if not count:
            count = zipped.count()
        reindexed = zipped.map(lambda kv: (kv[1], kv[0]))

        remaining = [swapped.shape[dim] for dim in range(len(swapped.shape)) if dim not in axes]
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
        from numpy import ndarray

        axes = sorted(axes)

        swapped = self._configure_key_axes(axes)

        arr = swapped._rdd.values().reduce(func)

        if not isinstance(arr, ndarray):
            # the result of a reduce can also be a scalar
            return arr
        elif arr.shape == (1,):
            # ndarrays with single values in them should be converted into scalars
            return arr[0]

        return BoltArrayLocal(arr)

    def _stats(self, axes=(0,), stats='all'):
        swapped = self._configure_key_axes(axes)

        def reducer(left, right):
            return left.combine(right)

        return swapped._rdd.values()\
                           .mapPartitions(lambda i: [StatCounter(values=i, stats=stats)])\
                           .reduce(reducer)

    def mean(self, axes=(0,)):
        from bolt.local.array import BoltArrayLocal
        return BoltArrayLocal(self._stats(axes, stats='mean').mean())

    def var(self, axes=(0,)):
        from bolt.local.array import BoltArrayLocal
        return BoltArrayLocal(self._stats(axes, stats='variance').variance())

    def std(self, axes=(0,)):
        from bolt.local.array import BoltArrayLocal
        return BoltArrayLocal(self._stats(axes, stats='stdev').stdev())

    def sum(self, axes=(0,)):
        from operator import add
        return self.reduce(add, axes)

    def max(self, axes=(0,)):
        from numpy import maximum
        return self.reduce(maximum, axes)

    def min(self, axes=(0,)):
        from numpy import minimum
        return self.reduce(minimum, axes)

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

            rdd = self._rdd.union(arry._rdd.map(lambda kv: (key_func(kv[0]), kv[1])))

        else:
            from numpy import concatenate as npconcatenate
            shift = axis - self.split
            rdd = self._rdd.join(arry._rdd).map(lambda kv: (kv[0], npconcatenate(kv[1], axis=shift)))

        shape = tuple([x + y if i == axis else x
                      for i, (x, y) in enumerate(zip(self.shape, arry.shape))])

        return self._constructor(rdd, shape=shape).__finalize__(self)

    def _getbasic(self, index):
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

        filtered = self._rdd.filter(lambda kv: key_check(kv[0]))
        rdd = filtered.map(lambda kv: (key_func(kv[0]), kv[1][value_slices]))
        shape = tuple([int(ceil((s.stop - s.start) / float(s.step))) for s in index])
        split = self.split
        return rdd, shape, split

    def _getadvanced(self, index):
        """
        Advanced indexing
        """
        index = [asarray(i) for i in index]
        shape = index[0].shape
        if not all([i.shape == shape for i in index]):
            raise ValueError("shape mismatch: indexing arrays could not be broadcast "
                             "together with shapes " + ("%s " * self.ndim)
                             % tuple([i.shape for i in index]))

        index = tuple([listify(i, d) for (i, d) in zip(index, self.shape)])

        # build tuples with target indices
        key_tuples = list(zip(*index[0:self.split]))
        value_tuples = list(zip(*index[self.split:]))

        # build dictionary to look up targets in values
        d = {}
        for k, g in groupby(zip(value_tuples, key_tuples), lambda x: x[1]):
            d[k] = map(lambda x: x[0], list(g))

        def key_check(key):
            return key in key_tuples

        def key_func(key):
            return unravel_index(key, shape)

        # filter records based on key targets
        filtered = self._rdd.filter(lambda kv: key_check(kv[0]))

        # subselect and flatten records based on value targets (if they exist)
        if len(value_tuples) > 0:
            flattened = filtered.flatMap(lambda kv: [(kv[0], kv[1][i]) for i in d[kv[0]]])
        else:
            flattened = filtered

        # reindex
        indexed = flattened.zipWithIndex()
        rdd = indexed.map(lambda kkv: (key_func(kkv[1]), kkv[0][1]))
        split = len(shape)

        return rdd, shape, split

    def __getitem__(self, index):

        index = tupleize(index)

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if not all([isinstance(i, (slice, int, list, set, ndarray)) for i in index]):
            raise ValueError("Each index must either be a slice, int, list, set, or ndarray")

        # fill unspecified axes with full slices
        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        # convert ints to lists if not all ints and slices
        if not all([isinstance(i, (int, slice)) for i in index]):
            index = tuple([[i] if isinstance(i, int) else i for i in index])

        # select basic or advanced indexing
        if all([isinstance(i, (slice, int)) for i in index]):
            rdd, shape, split = self._getbasic(index)
        elif all([isinstance(i, (set, list, ndarray)) for i in index]):
            rdd, shape, split = self._getadvanced(index)
        else:
            raise NotImplementedError("Cannot mix basic indexing (slices and ints) with "
                                      "advanced indexing (lists and ndarrays) across axes")

        result = self._constructor(rdd, shape=shape, split=split).__finalize__(self)

        # squeeze out int dimensions (and squeeze to singletons if all ints)
        if all([isinstance(i, int) for i in index]):
            return result.squeeze().toarray()[()]
        else:
            tosqueeze = tuple([i for i in index if isinstance(i, int)])
            return result.squeeze(tosqueeze)

    def swap(self, key_axes, value_axes, size=150):

        key_axes, value_axes = tupleize(key_axes), tupleize(value_axes)

        if len(key_axes) == self.keys.shape:
            raise ValueError('Cannot perform a swap that would '
                             'end up with all data on a single key')

        if len(key_axes) == 0 and len(value_axes) == 0:
            return self

        from bolt.spark.swap import Swapper, Dims

        k = Dims(shape=self.keys.shape, axes=key_axes)
        v = Dims(shape=self.values.shape, axes=value_axes)
        s = Swapper(k, v, self.dtype, size)

        chunks = s.chunk(self._rdd)
        rdd = s.extract(chunks)
        shape = s.getshape()
        split = self.split - len(key_axes) + len(value_axes)

        return self._constructor(rdd, shape=tuple(shape), split=split)

    def chunk(self, key_axes, value_axes, size):

        if len(key_axes) == 0 and len(value_axes) == 0:
            return self

        from bolt.spark.swap import Swapper, Dims

        k = Dims(shape=self.keys.shape, axes=key_axes)
        v = Dims(shape=self.values.shape, axes=value_axes)
        s = Swapper(k, v, self.dtype, size)
        return s.chunk(self._rdd)

    def transpose(self, axes):

        p = asarray(argpack(axes))
        split = self.split

        # compute the keys/value axes that need to be swapped
        new_keys, new_values = p[:split], p[split:]
        swapping_keys = sort(new_values[new_values < split])
        swapping_values = sort(new_keys[new_keys >= split])
        stationary_keys = sort(new_keys[new_keys < split])
        stationary_values = sort(new_values[new_values >= split])
        
        # compute the permutation that the swap causes
        p_swap = r_[stationary_keys, swapping_values, swapping_keys, stationary_values]

        # compute the extra permutation (p_x)  on top of this that needs to happen to get the full permutation desired
        p_swap_inv = argsort(p_swap)
        p_x = p_swap_inv[p]
        p_keys, p_values = p_x[:split], p_x[split:]-split

        # perform the swap and the the within key/value permutations
        arr = self.swap(swapping_keys, swapping_values-split)
        arr = arr.keys.transpose(tuple(p_keys.tolist()))
        arr = arr.values.transpose(tuple(p_values.tolist()))
        
        return arr

    @property
    def T(self):
        return self.transpose(range(self.ndim-1,-1,-1))

    def squeeze(self, axis=None):

        if not any([d == 1 for d in self.shape]):
            return self

        if axis is None:
            drop = where(asarray(self.shape) == 1)[0]
        elif isinstance(axis, int):
            drop = asarray((axis,))
        elif isinstance(axis, tuple):
            drop = asarray(axis)
        else:
            raise ValueError("an integer or tuple is required for the axis")

        if any([self.shape[i] > 1 for i in drop]):
            raise ValueError("cannot select an axis to squeeze out which has size greater than one")

        if any(asarray(drop) < self.split):
            kmask = set([d for d in drop if d < self.split])
            kfunc = lambda k: tuple([kk for ii, kk in enumerate(k) if ii not in kmask])
        else:
            kfunc = lambda k: k

        if any(asarray(drop) >= self.split):
            vmask = tuple([d - self.split for d in drop if d >= self.split])
            vfunc = lambda v: v.squeeze(vmask)
        else:
            vfunc = lambda v: v

        rdd = self._rdd.map(lambda kv: (kfunc(kv[0]), vfunc(kv[1])))
        shape = tuple([ss for ii, ss in enumerate(self.shape) if ii not in drop])
        split = len([d for d in range(self.keys.ndim) if d not in drop])
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
    def dtype(self):
        return self._dtype

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
            print(x)

    def astype(self, new_dtype):
        return self.map(lambda array:array.astype(new_dtype), dtype=new_dtype)
