from numpy import asarray, unravel_index, prod, mod, ndarray, ceil,  zeros, where, arange, r_, int16, sort, argsort
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

    def __array__(self):
        return self.toarray()

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
                             "together with shapes " + ("%s " * self.ndim)
                             % tuple([i.shape for i in index]))

        index = tuple([listify(i, d) for (i, d) in zip(index, self.shape)])

        # build tuples with target indices
        key_tuples = zip(*index[0:self.split])
        value_tuples = zip(*index[self.split:])

        # build dictionary to look up targets in values
        d = {}
        for k, g in groupby(zip(value_tuples, key_tuples), lambda x: x[1]):
            d[k] = map(lambda x: x[0], list(g))

        def key_check(key):
            return key in key_tuples

        def key_func(key):
            return unravel_index(key, shape)

        # filter records based on key targets
        filtered = self._rdd.filter(lambda (k, v): key_check(k))

        # subselect and flatten records based on value targets (if they exist)
        if len(value_tuples) > 0:
            flattened = filtered.flatMap(lambda (k, v): [(k, v[i]) for i in d[k]])
        else:
            flattened = filtered

        # reindex
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

        # fill unspecified axes with full slices
        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        # convert ints to lists if not all ints and slices
        if not all([isinstance(i, (int, slice)) for i in index]):
            index = tuple([[i] if isinstance(i, int) else i for i in index])

        # select basic or advanced indexing
        if all([isinstance(i, (slice, int)) for i in index]):
            rdd, shape, split = self.getbasic(index)
        elif all([isinstance(i, (set, list, ndarray)) for i in index]):
            rdd, shape, split = self.getadvanced(index)
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

    # TODO: once self.dtype is implemented, change int16 to self.dtype

    def swap(self, key_axes, value_axes, size=150):

        if len(key_axes) == self.keys.shape:
            raise ValueError('Cannot perform a swap that would '
                             'end up with all data on a single key')

        if len(key_axes) == 0 and len(value_axes) == 0:
            return self

        from bolt.spark.swap import Swapper, Dims

        k = Dims(shape=self.keys.shape, axes=key_axes)
        v = Dims(shape=self.values.shape, axes=value_axes)
        s = Swapper(k, v, int16, size)

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
        s = Swapper(k, v, int16, size)
        return s.chunk(self._rdd)

    def transpose(self, permutation):
        
        p = asarray(permutation)
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

        rdd = self._rdd.map(lambda (k, v): (kfunc(k), vfunc(v)))
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
