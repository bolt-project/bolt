from numpy import zeros, ones, asarray, r_, concatenate, arange, ceil, prod, \
    empty, mod, floor, any, ndarray, amin, amax, array_equal, squeeze, array, \
    where, random, ravel_multi_index

from itertools import product

from bolt.utils import tuplesort, tupleize, allstack, iterexpand
from bolt.spark.array import BoltArraySpark


class ChunkedArray(object):
    """
    Wraps a BoltArraySpark and provides an interface for chunking
    into subarrays and performing operations on chunks. Many methods will
    be restricted until the chunked array is unchunked.

    The general form supports axis movement during chunking, specifically,
    moving axes from keys to values and vice versa. For every
    value-dimension that becomes a key, the values are sliced along that
    dimension into 'chunks' of a user-specified size. This is an
    intermediate form that can be transformed back into a BoltSparkArray.
    """
    _metadata = ['_shape', '_split', '_dtype', '_plan', '_padding', '_ordered']

    def __init__(self, rdd, shape=None, split=None, dtype=None, plan=None, padding=None, ordered=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._dtype = dtype
        self._plan = plan
        self._padding = padding
        self._ordered = ordered

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def split(self):
        return self._split

    @property
    def plan(self):
        return self._plan

    @property
    def padding(self):
        return self._padding

    @property
    def uniform(self):
        return all([mod(x, y) == 0 for x, y in zip(self.vshape, self.plan)])

    @property
    def padded(self):
        return not all([p == 0 for p in self.padding])

    @property
    def kshape(self):
        return asarray(self._shape[:self._split])

    @property
    def vshape(self):
        return asarray(self._shape[self._split:])

    def kmask(self, axes):
        return self.getmask(axes, len(self.kshape))

    def vmask(self, axes):
        return self.getmask(axes, len(self.vshape))

    @property
    def _constructor(self):
        return ChunkedArray

    def __finalize__(self, other):
        for name in self._metadata:
            other_attr = getattr(other, name, None)
            if (other_attr is not None) and (getattr(self, name, None) is None):
                object.__setattr__(self, name, other_attr)
        return self

    def _chunk(self, size="150", axis=None, padding=None):
        """
        Split values of distributed array into chunks.

        Transforms an underlying pair RDD of (key, value) into
        records of the form: (key, chunk id), (chunked value).
        Here, chunk id is a tuple identifying the chunk and
        chunked value is a subset of the data from each original value,
        that has been divided along the specified dimensions.

        Parameters
        ----------
        size : str or tuple or int
            If str, the average size (in KB) of the chunks in all value dimensions.
            If int or tuple, an explicit specification of the number chunks in
            each value dimension.

        axis : tuple, optional, default=None
            One or more axes to estimate chunks for, if provided any
            other axes will use one chunk.

        padding: tuple or int, default = None
            Number of elements per dimension that will overlap with the adjacent chunk.
            If a tuple, specifies padding along each chunked dimension; if a int, same
            padding will be applied to all chunked dimensions.
        """
        if self.split == len(self.shape) and padding is None:
            self._rdd = self._rdd.map(lambda kv: (kv[0]+(0,), array(kv[1], ndmin=1)))
            self._shape = self._shape + (1,)
            self._plan = (1,)
            self._padding = array([0])
            return self

        rdd = self._rdd
        self._plan, self._padding = self.getplan(size, axis, padding)

        if any([x + y > z for x, y, z in zip(self.plan, self.padding, self.vshape)]):
            raise ValueError("Chunk sizes %s plus padding sizes %s cannot exceed value dimensions %s along any axis"
                             % (tuple(self.plan), tuple(self.padding), tuple(self.vshape)))

        if any([x > y for x, y in zip(self.padding, self.plan)]):
            raise ValueError("Padding sizes %s cannot exceed chunk sizes %s along any axis"
                             % (tuple(self.padding), tuple(self.plan)))

        slices = self.getslices(self.plan, self.padding, self.vshape)
        labels = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [list(zip(*s)) for s in labels]

        def _chunk(record):
            k, v = record[0], record[1]
            for (chk, slc) in scheme:
                if type(k) is int:
                    k = (k,)
                yield k + chk, v[slc]

        rdd = rdd.flatMap(_chunk)
        return self._constructor(rdd, shape=self.shape, split=self.split,
                                 dtype=self.dtype, plan=self.plan, padding=self.padding, ordered=self._ordered)

    def unchunk(self):
        """
        Convert a chunked array back into a full array with (key,value) pairs
        where key is a tuple of indices, and value is an ndarray.
        """
        plan, padding, vshape, split = self.plan, self.padding, self.vshape, self.split
        nchunks = self.getnumber(plan, vshape)
        full_shape = concatenate((nchunks, plan))
        n = len(vshape)
        perm = concatenate(list(zip(range(n), range(n, 2*n))))

        if self.uniform:
            def _unchunk(it):
                ordered = sorted(it, key=lambda kv: kv[0][split:])
                keys, values = zip(*ordered)
                yield keys[0][:split], asarray(values).reshape(full_shape).transpose(perm).reshape(vshape)
        else:
            def _unchunk(it):
                ordered = sorted(it, key=lambda kv: kv[0][split:])
                keys, values = zip(*ordered)
                k_chks = [k[split:] for k in keys]
                arr = empty(nchunks, dtype='object')
                for (i, d) in zip(k_chks, values):
                    arr[i] = d
                yield keys[0][:split], allstack(arr.tolist())

        # remove padding
        if self.padded:
            removepad = self.removepad
            rdd = self._rdd.map(lambda kv: (kv[0], removepad(kv[0][split:], kv[1], nchunks, padding, axes=range(n))))
        else:
            rdd = self._rdd

        # skip partitionBy if there is not actually any chunking
        if array_equal(self.plan, self.vshape):
           rdd = rdd.map(lambda kv: (kv[0][:split], kv[1]))
           ordered = self._ordered
        else:
            ranges = self.kshape
            npartitions = int(prod(ranges))
            if len(self.kshape) == 0:
                partitioner = lambda k: 0
            else:
                partitioner = lambda k: ravel_multi_index(k[:split], ranges)
            rdd = rdd.partitionBy(numPartitions=npartitions, partitionFunc=partitioner).mapPartitions(_unchunk)
            ordered = True

        if array_equal(self.vshape, [1]):
            rdd = rdd.mapValues(lambda v: squeeze(v))
            newshape = self.shape[:-1]
        else:
            newshape = self.shape

        return BoltArraySpark(rdd, shape=newshape, split=self._split,
                              dtype=self.dtype, ordered=ordered)

    def keys_to_values(self, axes, size=None):
        """
        Move indices in the keys into the values.

        Padding on these new value-dimensions is not currently supported and is set to 0.

        Parameters
        ----------
        axes : tuple
            Axes from keys to move to values.

        size : tuple, optional, default=None
            Size of chunks for the values along the new dimensions.
            If None, then no chunking for all axes (number of chunks = 1)

        Returns
        -------
        ChunkedArray
        """
        if len(axes) == 0:
            return self

        kmask = self.kmask(axes)

        if size is None:
            size = self.kshape[kmask]

        # update properties
        newplan = r_[size, self.plan]
        newsplit = self._split - len(axes)
        newshape = tuple(r_[self.kshape[~kmask], self.kshape[kmask], self.vshape].astype(int))
        newpadding = r_[zeros(len(axes), dtype=int), self.padding]

        result = self._constructor(None, shape=newshape, split=newsplit,
                                   dtype=self.dtype, plan=newplan, padding=newpadding, ordered=True)

        # convert keys into chunk + within-chunk label
        split = self.split
        def _relabel(record):
            k, data = record
            keys, chks = asarray(k[:split], 'int'), k[split:]
            movingkeys, stationarykeys = keys[kmask], keys[~kmask]
            newchks = [int(m) for m in movingkeys/size]  # element-wise integer division that works in Python 2 and 3
            labels = mod(movingkeys, size)
            return tuple(stationarykeys) + tuple(newchks) + tuple(chks) + tuple(labels), data

        rdd = self._rdd.map(_relabel)

        # group the new chunks together
        nchunks = result.getnumber(result.plan, result.vshape)
        npartitions = int(prod(result.kshape) * prod(nchunks))
        ranges = tuple(result.kshape) + tuple(nchunks)
        n = len(axes)
        if n == 0:
            s = slice(None)
        else:
            s = slice(-n)
        partitioner = lambda k: ravel_multi_index(k[s], ranges)

        rdd = rdd.partitionBy(numPartitions=npartitions, partitionFunc=partitioner)

        # reassemble the pieces in the chunks by sorting and then stacking
        uniform = result.uniform

        def _rebuild(it):
            ordered = sorted(it, key=lambda kv: kv[0][n:])
            keys, data = zip(*ordered)

            k = keys[0][s]
            labels = asarray([x[-n:] for x in keys])

            if uniform:
                labelshape = tuple(size)
            else:
                labelshape = tuple(amax(labels, axis=0) - amin(labels, axis=0) + 1)

            valshape = data[0].shape
            fullshape = labelshape + valshape
            yield k, asarray(data).reshape(fullshape)

        result._rdd = rdd.mapPartitions(_rebuild)

        if array_equal(self.vshape, [1]):
            result._rdd = result._rdd.mapValues(lambda v: squeeze(v))
            result._shape = result.shape[:-1]
            result._plan = result.plan[:-1]

        return result

    def values_to_keys(self, axes):

        vmask = self.vmask(axes)
        split = self.split

        # update properties
        newplan = self.plan[~vmask]
        newsplit = split + len(axes)
        newshape = tuple(r_[self.kshape, self.vshape[vmask], self.vshape[~vmask]].astype('int'))
        newpadding = self.padding[~vmask]

        result = self._constructor(None, shape=newshape, split=newsplit,
                                   dtype=self.dtype, plan=newplan, padding=newpadding, ordered=self._ordered)

        # remove padding
        if self.padded:
            plan, padding = self.plan, self.padding
            nchunks = self.getnumber(plan, self.vshape)
            removepad = self.removepad
            rdd = self._rdd.map(lambda kv: (kv[0], removepad(kv[0][split:], kv[1], nchunks, padding, axes=axes)))
        else:
            rdd = self._rdd

        # extract new records
        slices = [None if vmask[i] else slice(0, self.vshape[i], 1) for i in range(len(vmask))]
        slices = asarray(slices)

        movingsizes = self.plan[vmask]
        split = self.split
        def _extract(record):

            keys, data = record
            k, chk = keys[:split], keys[split:]

            movingchks = asarray(chk)[vmask]
            newchks = tuple(asarray(chk)[~vmask])
            keyoffsets = prod([movingchks, movingsizes], axis=0)

            bounds = asarray(data.shape)[vmask]
            indices = list(product(*map(lambda x: arange(x), bounds)))

            for b in indices:
                s = slices.copy()
                s[vmask] = b
                newdata = data[tuple(s)]
                newkeys = tuple(r_[k, keyoffsets + b].astype('int'))
                yield newkeys + newchks, newdata

        result._rdd = rdd.flatMap(_extract)

        if len(result.vshape) == 0:
            result._rdd = result._rdd.mapValues(lambda v: array(v, ndmin=1))
            result._shape = result._shape + (1,)
            result._plan = (1,)
            result._padding = array([0])

        return result

    def map(self, func, value_shape=None, dtype=None):
        """
        Apply an array -> array function on each subarray.

        The function can change the shape of the subarray, but only along
        dimensions that are not chunked.

        Parameters
        ----------
        func : function
            Function of a single subarray to apply

        value_shape:
            Known shape of chunking plan after the map

        dtype: numpy.dtype, optional, default=None
            Known dtype of values resulting from operation

        Returns
        -------
        ChunkedArray
        """

        if value_shape is None or dtype is None:
            # try to compute the size of each mapped element by applying func to a random array
            try:
                mapped = func(random.randn(*self.plan).astype(self.dtype))
            except Exception:
                first = self._rdd.first()
                if first:
                    # eval func on the first element
                    mapped = func(first[1])
            if value_shape is None:
                value_shape = mapped.shape
            if dtype is None:
                dtype = mapped.dtype

        chunked_dims = where(self.plan != self.vshape)[0]
        unchunked_dims = where(self.plan == self.vshape)[0]

        # check that no dimensions are dropped
        if len(value_shape) != len(self.plan):
            raise NotImplementedError('map on ChunkedArray cannot drop dimensions')

        # check that chunked dimensions did not change shape
        if any([value_shape[i] != self.plan[i] for i in chunked_dims]):
            raise ValueError('map cannot change the sizes of chunked dimensions')

        def check_and_apply(v):
            new = func(v)
            if len(unchunked_dims) > 0:
                if any([new.shape[i] != value_shape[i] for i in unchunked_dims]):
                    raise Exception("Map operation did not produce values of uniform shape.")
            if len(chunked_dims) > 0:
                if any([v.shape[i] != new.shape[i] for i in chunked_dims]):
                    raise Exception("Map operation changed the size of a chunked dimension")
            return new

        rdd = self._rdd.mapValues(check_and_apply)

        vshape = [value_shape[i] if i in unchunked_dims else self.vshape[i] for i in range(len(self.vshape))]
        newshape = r_[self.kshape, vshape].astype(int)

        return self._constructor(rdd, shape=tuple(newshape), dtype=dtype,
                                 plan=asarray(value_shape)).__finalize__(self)

    def map_generic(self, func):
        """
        Apply a generic array -> object to each subarray

        The resulting object is a BoltArraySpark of dtype object where the
        blocked dimensions are replaced with indices indication block ID.
        """
        def process_record(val):
            newval = empty(1, dtype="object")
            newval[0]  = func(val)
            return newval

        rdd = self._rdd.mapValues(process_record)

        nchunks = self.getnumber(self.plan, self.vshape)
        newshape = tuple([int(s) for s in r_[self.kshape, nchunks]])
        newsplit = len(self.shape)
        return BoltArraySpark(rdd, shape=newshape, split=newsplit, ordered=self._ordered, dtype="object")

    def getplan(self, size="150", axes=None, padding=None):
        """
        Identify a plan for chunking values along each dimension.

        Generates an ndarray with the size (in number of elements) of chunks
        in each dimension. If provided, will estimate chunks for only a
        subset of axes, leaving all others to the full size of the axis.

        Parameters
        ----------
        size : string or tuple
             If str, the average size (in KB) of the chunks in all value dimensions.
             If int/tuple, an explicit specification of the number chunks in
             each moving value dimension.

        axes : tuple, optional, default=None
              One or more axes to estimate chunks for, if provided any
              other axes will use one chunk.

        padding : tuple or int, option, default=None
            Size over overlapping padding between chunks in each dimension.
            If tuple, specifies padding along each chunked dimension; if int,
            all dimensions use same padding; if None, no padding
        """
        from numpy import dtype as gettype

        # initialize with all elements in one chunk
        plan = self.vshape

        # check for subset of axes
        if axes is None:
            if isinstance(size, str):
                axes = arange(len(self.vshape))
            else:
                axes = arange(len(size))
        else:
            axes = asarray(axes, 'int')

        # set padding
        pad = array(len(self.vshape)*[0, ])
        if padding is not None:
            pad[axes] = padding

        # set the plan
        if isinstance(size, tuple):
            plan[axes] = size

        elif isinstance(size, str):
            # convert from kilobytes
            size = 1000.0 * float(size)

            # calculate from dtype
            elsize = gettype(self.dtype).itemsize
            nelements = prod(self.vshape)
            dims = self.vshape[self.vmask(axes)]

            if size <= elsize:
                s = ones(len(axes))

            else:
                remsize = 1.0 * nelements * elsize
                s = []
                for (i, d) in enumerate(dims):
                    minsize = remsize/d
                    if minsize >= size:
                        s.append(1)
                        remsize = minsize
                        continue
                    else:
                        s.append(min(d, floor(size/minsize)))
                        s[i+1:] = plan[i+1:]
                        break

            plan[axes] = s

        else:
            raise ValueError("Chunk size not understood, must be tuple or int")

        return plan, pad

    @staticmethod
    def removepad(idx, value, number, padding, axes=None):
        """
        Remove the padding from chunks.

        Given a chunk and its corresponding index, use the plan and padding to remove any
        padding from the chunk along with specified axes.

        Parameters
        ----------
        idx: tuple or array-like
            The chunk index, indicating which chunk this is.

        value: ndarray
            The chunk that goes along with the index.

        number: ndarray or array-like
            The number of chunks along each dimension.

        padding: ndarray or array-like
            The padding scheme.

        axes: tuple, optional, default = None
            The axes (in the values) along which to remove padding.
        """
        if axes is None:
            axes = range(len(number))
        mask = len(number)*[False, ]
        for i in range(len(mask)):
            if i in axes and padding[i] != 0:
                mask[i] = True

        starts = [0 if (i == 0 or not m) else p for (i, m, p) in zip(idx, mask, padding)]
        stops = [None if (i == n-1 or not m) else -p for (i, m, p, n) in zip(idx, mask, padding, number)]
        slices = [slice(i1, i2) for (i1, i2) in zip(starts, stops)]

        return value[slices]

    @staticmethod
    def getnumber(plan, shape):
        """
        Obtain number of chunks for the given dimensions and chunk sizes.

        Given a plan for the number of chunks along each dimension,
        calculate the number of chunks that this will lead to.

        Parameters
        ----------
        plan: tuple or array-like
            Size of chunks (in number of elements) along each dimensions.
            Length must be equal to the number of dimensions.

        shape : tuple
             Shape of array to be chunked.
        """
        nchunks = []
        for size, d in zip(plan, shape):
            nchunks.append(int(ceil(1.0 * d/size)))
        return nchunks

    @staticmethod
    def getslices(plan, padding, shape):
        """
        Obtain slices for the given dimensions, padding, and chunks.

        Given a plan for the number of chunks along each dimension and the amount of padding,
        calculate a list of slices required to generate those chunks.

        Parameters
        ----------
        plan: tuple or array-like
            Size of chunks (in number of elements) along each dimensions.
            Length must be equal to the number of dimensions.

        padding: tuple or array-like
            Size of overlap (in number of elements) between chunks along each dimension.
            Length must be equal to the number of dimensions.

        shape: tuple
             Dimensions of axes to be chunked.
        """
        slices = []
        for size, pad, d in zip(plan, padding, shape):
            nchunks = int(floor(d/size))
            remainder = d % size
            start = 0
            dimslices = []
            for idx in range(nchunks):
                end = start + size
                # left endpoint
                if idx == 0:
                    left = start
                else:
                    left = start - pad
                # right endpoint
                if idx == nchunks:
                    right = end
                else:
                    right = end + pad
                dimslices.append(slice(left, right, 1))
                start = end
            if remainder:
                dimslices.append(slice(end - pad, d, 1))
            slices.append(dimslices)
        return slices

    @staticmethod
    def getmask(inds, n):
        """
        Obtain a binary mask by setting a subset of entries to true.

        Parameters
        ----------
        inds : array-like
            Which indices to set as true.

        n : int
            The length of the target mask.
        """
        inds = asarray(inds, 'int')
        mask = zeros(n, dtype=bool)
        mask[inds] = True
        return mask

    def tordd(self):
        """
        Return the RDD wrapped by the ChunkedArray.

        Returns
        -------
        RDD
        """
        return self._rdd

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

    def __str__(self):
        s = "Chunked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        return s

    def __repr__(self):
        string = str(self)
        if array_equal(self.vshape, [1]):
            newlines = [i for (i, char) in enumerate(string) if char=='\n']
            string = string[:newlines[-2]+1]
            string += "shape: %s\n" % str(self.shape[:-1])
        string += "chunk size: %s\n" % str(tuple(self.plan))
        if self.padded:
            string += "padding: %s\n" % str(tuple(self.padding))
        else:
            string += "padding: none\n"

        return string
