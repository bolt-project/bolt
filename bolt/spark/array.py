from __future__ import print_function
from numpy import asarray, unravel_index, prod, mod, ndarray, ceil, where, \
    r_, sort, argsort, array, random, arange, ones, expand_dims, sum
from itertools import groupby

from bolt.base import BoltArray
from bolt.spark.stack import StackedArray
from bolt.spark.utils import zip_with_index
from bolt.spark.statcounter import StatCounter
from bolt.utils import slicify, listify, tupleize, argpack, inshape, istransposeable, isreshapeable


class BoltArraySpark(BoltArray):

    _metadata = {
        '_shape': None,
        '_split': None,
        '_dtype': None,
        '_ordered': True
    }

    def __init__(self, rdd, shape=None, split=None, dtype=None, ordered=True):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._dtype = dtype
        self._mode = 'spark'
        self._ordered = ordered

    @property
    def _constructor(self):
        return BoltArraySpark

    def __array__(self):
        return self.toarray()

    def cache(self):
        """
        Cache the underlying RDD in memory.
        """
        self._rdd.cache()

    def unpersist(self):
        """
        Remove the underlying RDD from memory.
        """
        self._rdd.unpersist()

    def repartition(self, npartitions):
        """
        Repartitions the underlying RDD

        Parameters
        ----------
        npartitions : int
            Number of partitions to repartion the underlying RDD to
        """

        rdd = self._rdd.repartition(npartitions)
        return self._constructor(rdd, ordered=False).__finalize__(self)

    def stack(self, size=None):
        """
        Aggregates records of a distributed array.

        Stacking should improve the performance of vectorized operations,
        but the resulting StackedArray object only exposes a restricted set
        of operations (e.g. map, reduce). The unstack method can be used
        to restore the full bolt array.

        Parameters
        ----------
        size : int, optional, default=None
            The maximum size for each stack (number of original records),
            will aggregate groups of records per partition up to this size,
            if None will aggregate all records on each partition.

        Returns
        -------
        StackedArray
        """
        stk = StackedArray(self._rdd, shape=self.shape, split=self.split)
        return stk.stack(size)

    def _align(self, axis):
        """
        Align spark bolt array so that axes for iteration are in the keys.

        This operation is applied before most functional operators.
        It ensures that the specified axes are valid, and swaps
        key/value axes so that functional operators can be applied
        over the correct records.

        Parameters
        ----------
        axis: tuple[int]
            One or more axes that wil be iterated over by a functional operator

        Returns
        -------
        BoltArraySpark
        """
        # ensure that the specified axes are valid
        inshape(self.shape, axis)

        # find the value axes that should be moved into the keys (axis >= split)
        tokeys = [(a - self.split) for a in axis if a >= self.split]

        # find the key axes that should be moved into the values (axis < split)
        tovalues = [a for a in range(self.split) if a not in axis]

        if tokeys or tovalues:
            return self.swap(tovalues, tokeys)
        else:
            return self

    def first(self):
        """
        Return the first element of an array
        """
        from bolt.local.array import BoltArrayLocal
        rdd = self._rdd if self._ordered else self._rdd.sortByKey()
        return BoltArrayLocal(rdd.values().first())

    def map(self, func, axis=(0,), value_shape=None, dtype=None, with_keys=False):
        """
        Apply a function across an axis.

        Array will be aligned so that the desired set of axes
        are in the keys, which may incur a swap.

        Parameters
        ----------
        func : function
            Function of a single array to apply. If with_keys=True,
            function should be of a (tuple, array) pair.

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to apply function along.

        value_shape : tuple, optional, default=None
            Known shape of values resulting from operation

        dtype: numpy.dtype, optional, default=None
            Known dtype of values resulting from operation

        with_keys : bool, optional, default=False
            Include keys as an argument to the function

        Returns
        -------
        BoltArraySpark
        """
        axis = tupleize(axis)
        swapped = self._align(axis)

        if with_keys:
            test_func = lambda x: func(((0,), x))
        else:
            test_func = func

        if value_shape is None or dtype is None:
            # try to compute the size of each mapped element by applying func to a random array
            try:
                mapped = test_func(random.randn(*swapped.values.shape).astype(self.dtype))
            except Exception:
                first = swapped._rdd.first()
                if first:
                    # eval func on the first element
                    mapped = test_func(first[1])
            if value_shape is None:
                value_shape = mapped.shape
            if dtype is None:
                dtype = mapped.dtype

        shape = tuple([swapped._shape[ax] for ax in range(len(axis))]) + tupleize(value_shape)

        if with_keys:
            rdd = swapped._rdd.map(lambda kv: (kv[0], func(kv)))
        else:
            rdd = swapped._rdd.mapValues(func)

        # reshaping will fail if the elements aren't uniformly shaped
        def check(v):
            if len(v.shape) > 0 and v.shape != tupleize(value_shape):
                raise Exception("Map operation did not produce values of uniform shape.")
            return v

        rdd = rdd.mapValues(lambda v: check(v))

        return self._constructor(rdd, shape=shape, dtype=dtype, split=swapped.split).__finalize__(swapped)

    def filter(self, func, axis=(0,), sort=False):
        """
        Filter array along an axis.

        Applies a function which should evaluate to boolean,
        along a single axis or multiple axes. Array will be
        aligned so that the desired set of axes are in the keys,
        which may incur a swap.

        Parameters
        ----------
        func : function
            Function to apply, should return boolean

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to filter along.

        sort: bool, optional, default=False
            Whether or not to sort by key before reindexing

        Returns
        -------
        BoltArraySpark
        """
        axis = tupleize(axis)

        swapped = self._align(axis)
        def f(record):
            return func(record[1])
        rdd = swapped._rdd.filter(f)
        if sort:
            rdd = rdd.sortByKey().values()
        else:
            rdd = rdd.values()

        # count the resulting array in order to reindex (linearize) the keys
        count, zipped = zip_with_index(rdd)
        if not count:
            count = zipped.count()
        reindexed = zipped.map(lambda kv: (tupleize(kv[1]), kv[0]))

        # since we can only filter over one axis, the remaining shape is always the following
        remaining = list(swapped.shape[len(axis):])
        if count != 0:
            shape = tuple([count] + remaining)
        else:
            shape = (0,)

        return self._constructor(reindexed, shape=shape, split=swapped.split).__finalize__(swapped)

    def reduce(self, func, axis=(0,), keepdims=False):
        """
        Reduce an array along an axis.

        Applies a commutative/associative function of two
        arguments cumulatively to all arrays along an axis.
        Array will be aligned so that the desired set of axes
        are in the keys, which may incur a swap.

        Parameters
        ----------
        func : function
            Function of two arrays that returns a single array

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to reduce along.

        Returns
        -------
        BoltArraySpark
        """
        from bolt.local.array import BoltArrayLocal
        from numpy import ndarray

        axis = tupleize(axis)
        swapped = self._align(axis)
        arr = swapped._rdd.values().treeReduce(func, depth=3)

        if keepdims:
            for i in axis:
                arr = expand_dims(arr, axis=i)

        if not isinstance(arr, ndarray):
            # the result of a reduce can also be a scalar
            return arr
        elif arr.shape == (1,):
            # ndarrays with single values in them should be converted into scalars
            return arr[0]

        return BoltArrayLocal(arr)

    def _stat(self, axis=None, func=None, name=None, keepdims=False):
        """
        Compute a statistic over an axis.

        Can provide either a function (for use in a reduce)
        or a name (for use by a stat counter).

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        func : function, optional, default=None
            Function for reduce, see BoltArraySpark.reduce

        name : str
            A named statistic, see StatCounter

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        if axis is None:
            axis = list(range(len(self.shape)))
        axis = tupleize(axis)

        if func and not name:
            return self.reduce(func, axis, keepdims)

        if name and not func:
            from bolt.local.array import BoltArrayLocal

            swapped = self._align(axis)

            def reducer(left, right):
                return left.combine(right)

            counter = swapped._rdd.values()\
                             .mapPartitions(lambda i: [StatCounter(values=i, stats=name)])\
                             .treeReduce(reducer, depth=3)

            arr = getattr(counter, name)

            if keepdims:
                for i in axis:
                    arr = expand_dims(arr, axis=i)

            return BoltArrayLocal(arr).toscalar()

        else:
            raise ValueError('Must specify either a function or a statistic name.')

    def mean(self, axis=None, keepdims=False):
        """
        Return the mean of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        return self._stat(axis, name='mean', keepdims=keepdims)

    def var(self, axis=None, keepdims=False):
        """
        Return the variance of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        return self._stat(axis, name='variance', keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        """
        Return the standard deviation of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        return self._stat(axis, name='stdev', keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        """
        Return the sum of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        from operator import add
        return self._stat(axis, func=add, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """
        Return the maximum of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        from numpy import maximum
        return self._stat(axis, func=maximum, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        """
        Return the minimum of the array over the given axis.

        Parameters
        ----------
        axis : tuple or int, optional, default=None
            Axis to compute statistic over, if None
            will compute over all axes

        keepdims : boolean, optional, default=False
            Keep axis remaining after operation with size 1.
        """
        from numpy import minimum
        return self._stat(axis, func=minimum, keepdims=keepdims)

    def concatenate(self, arry, axis=0):
        """
        Join this array with another array.

        Paramters
        ---------
        arry : ndarray, BoltArrayLocal, or BoltArraySpark
            Another array to concatenate with

        axis : int, optional, default=0
            The axis along which arrays will be joined.

        Returns
        -------
        BoltArraySpark
        """
        if isinstance(arry, ndarray):
            from bolt.spark.construct import ConstructSpark
            arry = ConstructSpark.array(arry, self._rdd.context, axis=range(0, self.split))
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

        return self._constructor(rdd, shape=shape, ordered=False).__finalize__(self)

    def _getbasic(self, index):
        """
        Basic indexing (for slices or ints).
        """
        key_slices = index[0:self.split]
        value_slices = index[self.split:]

        def key_check(key):
            def inrange(k, s):
                if s.step > 0:
                    return s.start <= k < s.stop
                else:
                    return s.stop < k <= s.start
            def check(k, s):
                return inrange(k, s) and mod(k - s.start, s.step) == 0
            out = [check(k, s) for k, s in zip(key, key_slices)]
            return all(out)

        def key_func(key):
            return tuple([(k - s.start)/s.step for k, s in zip(key, key_slices)])

        filtered = self._rdd.filter(lambda kv: key_check(kv[0]))

        if self._split == self.ndim:
            rdd = filtered.map(lambda kv: (key_func(kv[0]), kv[1]))
        else:
            # handle use of use slice.stop = -1 for a special case (see utils.slicify)
            value_slices = [s if s.stop != -1 else slice(s.start, None, s.step) for s in value_slices]
            rdd = filtered.map(lambda kv: (key_func(kv[0]), kv[1][value_slices]))

        shape = tuple([int(ceil((s.stop - s.start) / float(s.step))) for s in index])
        split = self.split
        return rdd, shape, split

    def _getadvanced(self, index):
        """
        Advanced indexing (for sets, lists, or ndarrays).
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

    def _getmixed(self, index):
        """
        Mixed indexing (combines basic and advanced indexes)

        Assumes that only a single advanced index is used, due to the complicated
        behavior needed to be compatible with NumPy otherwise.
        """
        # find the single advanced index
        loc = where([isinstance(i, (tuple, list, ndarray)) for i in index])[0][0]
        idx = list(index[loc])

        if isinstance(idx[0], (tuple, list, ndarray)):
            raise ValueError("When mixing basic and advanced indexing, "
                             "advanced index must be one-dimensional")

        # single advanced index is on a key -- filter and update key
        if loc < self.split:
            def newkey(key):
                newkey = list(key)
                newkey[loc] = idx.index(key[loc])
                return tuple(newkey)
            rdd = self._rdd.filter(lambda kv: kv[0][loc] in idx).map(lambda kv: (newkey(kv[0]), kv[1]))
        # single advanced index is on a value -- use NumPy indexing
        else:
            slices = [slice(0, None, None) for _ in self.values.shape]
            slices[loc - self.split] = idx
            rdd = self._rdd.map(lambda kv: (kv[0], kv[1][slices]))
        newshape = list(self.shape)
        newshape[loc] = len(idx)
        barray = self._constructor(rdd, shape=tuple(newshape)).__finalize__(self)

        # apply the rest of the simple indices
        new_index = index[:]
        new_index[loc] = slice(0, None, None)
        barray = barray[tuple(new_index)]
        return barray._rdd, barray.shape, barray.split

    def __getitem__(self, index):
        """
        Get an item from the array through indexing.

        Supports basic indexing with slices and ints, or advanced
        indexing with lists or ndarrays of integers.
        Mixing basic and advanced indexing across axes is currently supported
        only for a single advanced index amidst multiple basic indices.

        Parameters
        ----------
        index : tuple of slices, ints, list, tuple, or ndarrays
            One or more index specifications

        Returns
        -------
        BoltSparkArray
        """
        if isinstance(index, tuple):
            index = list(index)
        else:
            index = [index]
        int_locs = where([isinstance(i, int) for i in index])[0]

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if not all([isinstance(i, (slice, int, list, tuple, ndarray)) for i in index]):
            raise ValueError("Each index must either be a slice, int, list, set, or ndarray")

        # fill unspecified axes with full slices
        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        # standardize slices and bounds checking
        for n, idx in enumerate(index):
            size = self.shape[n]
            if isinstance(idx, (slice, int)):
                slc = slicify(idx, size)
                # throw an error if this would lead to an empty dimension in numpy
                if slc.step > 0:
                    minval, maxval = slc.start, slc.stop
                else:
                    minval, maxval = slc.stop, slc.start
                if minval > size-1 or maxval < 1 or minval >= maxval:
                    raise ValueError("Index {} in dimension {} with shape {} would "
                                     "produce an empty dimension".format(idx, n, size))
                index[n] = slc
            else:
                adjusted = array(idx)
                inds = where(adjusted<0)
                adjusted[inds] += size
                if adjusted.min() < 0 or adjusted.max() > size-1:
                    raise ValueError("Index {} out of bounds in dimension {} with "
                                     "shape {}".format(idx, n, size))
                index[n] = adjusted

        # select basic or advanced indexing
        if all([isinstance(i, slice) for i in index]):
            rdd, shape, split = self._getbasic(index)
        elif all([isinstance(i, (tuple, list, ndarray)) for i in index]):
            rdd, shape, split = self._getadvanced(index)
        elif sum([isinstance(i, (tuple, list, ndarray)) for i in index]) == 1:
            rdd, shape, split = self._getmixed(index)
        else:
            raise NotImplementedError("When mixing basic indexing (slices and int) with "
                                      "with advanced indexing (lists, tuples, and ndarrays), "
                                      "can only have a single advanced index")

        # if any key indices used negative steps, records are no longer ordered
        if self._ordered is False or any([isinstance(s, slice) and s.step<0 for s in index[:self.split]]):
            ordered = False
        else:
            ordered = True

        result = self._constructor(rdd, shape=shape, split=split, ordered=ordered).__finalize__(self)

        # squeeze out int dimensions (and squeeze to singletons if all ints)
        if len(int_locs) == self.ndim:
            return result.squeeze().toarray()[()]
        else:
            return result.squeeze(tuple(int_locs))

    def chunk(self, size="150", axis=None, padding=None):
        """
        Chunks records of a distributed array.

        Chunking breaks arrays into subarrays, using an specified
        size of chunks along each value dimension. Can alternatively
        specify an average chunk byte size (in kilobytes) and the size of
        chunks (as ints) will be computed automatically.

        Parameters
        ----------
        size : tuple, int, or str, optional, default = "150"
            A string giving the size in kilobytes, or a tuple with the size
            of chunks along each dimension.

        axis : int or tuple, optional, default = None
            One or more axis to chunk array along, if None
            will use all axes,

        padding: tuple or int, default = None
            Number of elements per dimension that will overlap with the adjacent chunk.
            If a tuple, specifies padding along each chunked dimension; if a int, same
            padding will be applied to all chunked dimensions.

        Returns
        -------
        ChunkedArray
        """
        if type(size) is not str:
            size = tupleize((size))
        axis = tupleize((axis))
        padding = tupleize((padding))

        from bolt.spark.chunk import ChunkedArray

        chnk = ChunkedArray(rdd=self._rdd, shape=self._shape, split=self._split, dtype=self._dtype)
        return chnk._chunk(size, axis, padding)

    def swap(self, kaxes, vaxes, size="150"):
        """
        Swap axes from keys to values.

        This is the core operation underlying shape manipulation
        on the Spark bolt array. It exchanges an arbitrary set of axes
        between the keys and the valeus. If either is None, will only
        move axes in one direction (from keys to values, or values to keys).
        Keys moved to values will be placed immediately after the split;
        values moved to keys will be placed immediately before the split.

        Parameters
        ----------
        kaxes : tuple
            Axes from keys to move to values

        vaxes : tuple
            Axes from values to move to keys

        size : tuple or int, optional, default = "150"
            Can either provide a string giving the size in kilobytes,
            or a tuple with the number of chunks along each
            value dimension being moved

        Returns
        -------
        BoltArraySpark
        """
        kaxes = asarray(tupleize(kaxes), 'int')
        vaxes = asarray(tupleize(vaxes), 'int')
        if type(size) is not str:
            size = tupleize(size)

        if len(kaxes) == self.keys.ndim and len(vaxes) == 0:
            raise ValueError('Cannot perform a swap that would '
                             'end up with all data on a single key')

        if len(kaxes) == 0 and len(vaxes) == 0:
            return self

        from bolt.spark.chunk import ChunkedArray

        chunks = self.chunk(size)

        swapped = chunks.keys_to_values(kaxes).values_to_keys([v+len(kaxes) for v in vaxes])
        barray = swapped.unchunk()

        return barray

    def transpose(self, *axes):
        """
        Return an array with the axes transposed.

        This operation will incur a swap unless the
        desiured permutation can be obtained
        only by transpoing the keys or the values.

        Parameters
        ----------
        axes : None, tuple of ints, or n ints
            If None, will reverse axis order.
        """
        if len(axes) == 0:
            p = arange(self.ndim-1, -1, -1)
        else:
            p = asarray(argpack(axes))

        istransposeable(p, range(self.ndim))

        split = self.split

        # compute the keys/value axes that need to be swapped
        new_keys, new_values = p[:split], p[split:]
        swapping_keys = sort(new_values[new_values < split])
        swapping_values = sort(new_keys[new_keys >= split])
        stationary_keys = sort(new_keys[new_keys < split])
        stationary_values = sort(new_values[new_values >= split])

        # compute the permutation that the swap causes
        p_swap = r_[stationary_keys, swapping_values, swapping_keys, stationary_values]

        # compute the extra permutation (p_x)  on top of this that
        # needs to happen to get the full permutation desired
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
        """
        Transpose by reversing the order of the axes.
        """
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        """
        Return the array with two axes interchanged.

        Parameters
        ----------
        axis1 : int
            The first axis to swap

        axis2 : int
            The second axis to swap
        """
        p = list(range(self.ndim))
        p[axis1] = axis2
        p[axis2] = axis1

        return self.transpose(p)

    def reshape(self, *shape):
        """
        Return an array with the same data but a new shape.

        Currently only supports reshaping that independently
        reshapes the keys, or the values, or both.

        Parameters
        ----------
        shape :  tuple of ints, or n ints
            New shape
        """
        new = argpack(shape)
        isreshapeable(new, self.shape)

        if new == self.shape:
            return self

        i = self._reshapebasic(new)
        if i == -1:
            raise NotImplementedError("Currently no support for reshaping between "
                                      "keys and values for BoltArraySpark")
        else:
            new_key_shape, new_value_shape = new[:i], new[i:]
            return self.keys.reshape(new_key_shape).values.reshape(new_value_shape)

    def _reshapebasic(self, shape):
        """
        Check if the requested reshape can be broken into independant reshapes
        on the keys and values. If it can, returns the index in the new shape
        separating keys from values, otherwise returns -1
        """
        new = tupleize(shape)
        old_key_size = prod(self.keys.shape)
        old_value_size = prod(self.values.shape)

        for i in range(len(new)):
            new_key_size = prod(new[:i])
            new_value_size = prod(new[i:])
            if new_key_size == old_key_size and new_value_size == old_value_size:
                return i

        return -1

    def squeeze(self, axis=None):
        """
        Remove one or more single-dimensional axes from the array.

        Parameters
        ----------
        axis : tuple or int
            One or more singleton axes to remove.
        """
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

    def astype(self, dtype, casting='unsafe'):
        """
        Cast the array to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to cast the array to (see numpy)
        """
        rdd = self._rdd.mapValues(lambda v: v.astype(dtype, 'K', casting))
        return self._constructor(rdd, dtype=dtype).__finalize__(self)

    def clip(self, min=None, max=None):
        """
        Clip values above and below.

        Parameters
        ----------
        min : scalar or array-like
            Minimum value. If array, will be broadcasted

        max : scalar or array-like
            Maximum value. If array, will be broadcasted.
        """
        rdd = self._rdd.mapValues(lambda v: v.clip(min=min, max=max))
        return self._constructor(rdd).__finalize__(self)

    @property
    def shape(self):
        """
        Size of each dimension.
        """
        return self._shape

    @property
    def size(self):
        """
        Total number of elements.
        """
        return prod(self._shape)

    @property
    def ndim(self):
        """
        Number of dimensions.
        """
        return len(self._shape)

    @property
    def split(self):
        """
        Axis at which the array is split into keys/values.
        """
        return self._split

    @property
    def dtype(self):
        """
        Data-type of array.
        """
        return self._dtype

    @property
    def mask(self):
        return tuple([1] * len(self.keys.shape) + [0] * len(self.values.shape))

    @property
    def keys(self):
        """
        Returns a restricted keys.
        """
        from bolt.spark.shapes import Keys
        return Keys(self)

    @property
    def values(self):
        from bolt.spark.shapes import Values
        return Values(self)

    def tolocal(self):
        """
        Returns a local bolt array by first collecting as an array.
        """
        from bolt.local.array import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        """
        Returns the contents as a local array.

        Will likely cause memory problems for large objects.
        """
        rdd = self._rdd if self._ordered else self._rdd.sortByKey()
        x = rdd.values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        """
        Return the underlying RDD of the bolt array.
        """
        return self._rdd

    def display(self):
        """
        Show a pretty-printed representation of this BoltArrayLocal.
        """
        for x in self._rdd.take(10):
            print(x)
