from numpy import zeros, ones, asarray, r_, concatenate, arange, ceil, prod, \
    empty, mod, floor, any, ndarray, amin, amax, array_equal, squeeze, array


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
    _metadata = ['_shape', '_split', '_dtype', '_plan']

    def __init__(self, rdd, shape=None, split=None, dtype=None, plan=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._dtype = dtype
        self._plan = plan

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
    def uniform(self):
        return all([mod(x, y) == 0 for x, y in zip(self.vshape, self.plan)])
    
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

    def _chunk(self, size, axis=None):
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
            If str, the average size (in MB) of the chunks in all value dimensions.
            If int or tuple, an explicit specification of the number chunks in
            each value dimension.

        axis : tuple, optional, default=None
            One or more axes to estimate chunks for, if provided any
            other axes will use one chunk.
        """
        if self.split == len(self.shape):
            self._rdd = self._rdd.map(lambda kv: ((kv[0], ()), array(kv[1], ndmin=1)))
            self._shape = self._shape + (1,)
            self._plan = (1,)
            return self

        rdd = self._rdd
        self._plan = self.getplan(size, axis)

        if any([x > y for x, y in zip(self.plan, self.vshape)]):
            raise ValueError("Chunk sizes %s cannot exceed value dimensions %s along any axis"
                             % (tuple(self.plan), tuple(self.vshape)))

        slices = self.getslices(self.plan, self.vshape)
        labels = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [list(zip(*s)) for s in labels]

        def _chunk(record):
            k, v = record[0], record[1]
            for (chk, slc) in scheme:
                if type(k) is int:
                    k = (k,)
                yield (k, chk), v[slc]

        rdd = rdd.flatMap(_chunk)
        return self._constructor(rdd, shape=self.shape, split=self.split,
                                 dtype=self.dtype, plan=self.plan)

    def unchunk(self):
        """
        Convert a chunked array back into a full array with (key,value) pairs
        where key is a tuple of indices, and value is an ndarray.
        """
        plan, vshape = self.plan, self.vshape
        nchunks = self.getnumber(plan, vshape)
        full_shape = concatenate((nchunks, plan))
        n = len(vshape)
        perm = concatenate(list(zip(range(n), range(n, 2*n))))

        if self.uniform:
            def _unchunk(v):
                idx, data = zip(*v.data)
                sorted_idx = tuplesort(idx)
                return asarray(data)[sorted_idx].reshape(full_shape).transpose(perm).reshape(vshape)
        else:
            def _unchunk(v):
                idx, data = zip(*v.data)
                arr = empty(nchunks, dtype='object')
                for (i, d) in zip(idx, data):
                    arr[i] = d
                return allstack(arr.tolist())

        switch = self.switch
        rdd = self._rdd.map(switch).groupByKey().mapValues(_unchunk)

        if array_equal(self.vshape, [1]):
            rdd = rdd.mapValues(lambda v: squeeze(v))
            newshape = self.shape[:-1]
        else:
            newshape = self.shape

        return BoltArraySpark(rdd, shape=newshape, split=self._split, dtype=self.dtype)

    def keys_to_values(self, axes, size=None):
        """
        Move indices in the keys into the values.

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
        kmask = self.kmask(axes)

        if size is None:
            size = self.kshape[kmask]

        # update properties
        newplan = r_[size, self.plan]
        newsplit = self._split - len(axes)
        newshape = tuple(r_[self.kshape[~kmask], self.kshape[kmask], self.vshape].astype(int))

        result = self._constructor(None, shape=newshape, split=newsplit,
                                   dtype=self.dtype, plan=newplan)

        # convert keys into chunk + within-chunk label
        def _relabel(record):
            k, chk = asarray(record[0][0], 'int'), asarray(record[0][1], 'int')
            data = asarray(record[1])
            movingkeys, stationarykeys = k[kmask], k[~kmask]
            newchks = [int(m) for m in movingkeys/size]  # element-wise integer division that works in Python 2 and 3
            labels = mod(movingkeys, size)
            return (tuple(stationarykeys), tuple(newchks)+tuple(chk)), (tuple(labels), data)

        rdd = self._rdd.map(_relabel)

        # group the new chunks together
        rdd = rdd.groupByKey()

        # reassemble the pieces in the chunks by sorting and then stacking
        uniform = result.uniform

        def _rebuild(v):
            labels, data = zip(*v.data)
            sortinginds = tuplesort(labels)

            if uniform:
                labelshape = tuple(size)
            else:
                labelshape = tuple(amax(labels, axis=0) - amin(labels, axis=0) + 1)
            valshape = data[0].shape
            fullshape = labelshape + valshape
            return asarray(data)[sortinginds].reshape(fullshape)

        result._rdd = rdd.mapValues(_rebuild)

        if array_equal(self.vshape, [1]):
            result._rdd = result._rdd.mapValues(lambda v: squeeze(v))
            result._shape = result.shape[:-1]
            result._plan = result.plan[:-1]

        return result

    def values_to_keys(self, axes):

        vmask = self.vmask(axes)

        # update properties
        newplan = self.plan[~vmask]
        newsplit = self._split + len(axes)
        newshape = tuple(r_[self.kshape, self.vshape[vmask], self.vshape[~vmask]].astype('int'))

        result = self._constructor(None, shape=newshape, split=newsplit,
                                   dtype=self.dtype, plan=newplan)

        slices = [None if vmask[i] else slice(0, self.vshape[i], 1) for i in range(len(vmask))]
        slices = asarray(slices)

        movingsizes = self.plan[vmask]

        def _extract(record):

            (k, chk), data = record

            movingchks = asarray(chk)[vmask]
            newchks = tuple(asarray(chk)[~vmask])
            keyoffsets = prod([movingchks, movingsizes], axis=0)

            bounds = asarray(data.shape)[vmask]
            indices = list(product(*map(lambda x: arange(x), bounds)))

            for b in indices:
                s = slices.copy()
                s[vmask] = b
                newdata = data[tuple(s)]
                newkeys = tuple(r_[k, keyoffsets + b])
                yield (newkeys, newchks), newdata

        result._rdd = self._rdd.flatMap(_extract)

        if len(result.vshape) == 0:
            result._rdd = result._rdd.mapValues(lambda v: array(v, ndmin=1))
            result._shape = result._shape + (1,)
            result._plan = (1,)

        return result

    def map(self, func):
        """
        Apply a function on each subarray.

        The function can change the shape of the underlying chunks
        and shape information will be correctly propagated,
        but if the function changes shape in a non-constant way
        (i.e. yields different shapes for different arrays)
        unexpected errors may occur.

        Parameters
        ----------
        func : function
             This is applied to each value in the intermediate RDD,
             which correspond to chunks of the original values.

        Returns
        -------
        ChunkedArray
        """
        if not self.uniform:
            raise NotImplementedError("Map only supported on evenly chunked arrays")

        x = self._rdd.values().first()

        try:
            xtest = func(x)
        except Exception as e:
            raise RuntimeError("Error evaluating function on test array, got error:\n %s" % e)

        if not (isinstance(xtest, ndarray)):
            raise ValueError("Function must return ndarray")

        missing = x.ndim - xtest.ndim

        if missing > 0:
            # the function dropped a dimension
            # add new empty dimensions so that unchunking will work
            mapfunc = lambda v: iterexpand(func(v), missing)
            xtest = mapfunc(x)
        else:
            mapfunc = func

        full = asarray(xtest.shape) * self.getnumber(self.plan, self.vshape)
        plan = asarray(xtest.shape)
        shape = tuple(self.kshape) + tuple(full)

        rdd = self._rdd.mapValues(mapfunc)
        return self._constructor(rdd, shape=shape, plan=plan).__finalize__(self)

    def getplan(self, size="150", axes=None):
        """
        Identify a plan for chunking values along each dimension.

        Generates an ndarray with the size (in number of elements) of chunks
        in each dimension as well as the original dimensions of the values
        used in this computation. If provided, will estimate chunks for only a
        subset of axes, leaving all others to the full size of the axis.

        Parameters
        ----------
        size : string or tuple
             If str, the average size (in MB) of the chunks in all value dimensions.  
             If int/tuple, an explicit specification of the number chunks in 
             each moving value dimension.

        axes : tuple, optional, default=None
              One or more axes to estimate chunks for, if provided any
              other axes will use one chunk.
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

        if isinstance(size, tuple):
            plan[axes] = size

        elif isinstance(size, str):
            # convert from megabytes
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
                        break

            plan[axes] = s

        else:
            raise ValueError("Chunk size not understood, must be tuple or int")

        return plan

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
    def getslices(plan, shape):
        """
        Obtain slices for the given dimensions and chunks.

        Given a plan for the number of chunks along each dimension,
        calculate a list of slices required to generate those chunks.

        Parameters
        ----------
        plan: tuple or array-like
            Size of chunks (in number of elements) along each dimensions.
            Length must be equal to the number of dimensions.

        shape: tuple
             Dimensions of axes to be chunked.
        """
        slices = []
        for size, d in zip(plan, shape):
            nchunks = int(floor(d/size))
            remainder = d % size 
            start = 0
            dimslices = []
            for idx in range(nchunks):
                end = start + size
                dimslices.append(slice(start, end, 1))
                start = end
            if remainder:
                dimslices.append(slice(end, d, 1))
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

    @staticmethod
    def switch(record):
        """
        Helper function that moves the chunk ids from the key to the value
        """
        (k, chk), v = record
        return k, (chk, v)

    def tordd(self):
        """
        Return the RDD wrapped by the ChunkedArray.

        Returns
        -------
        RDD
        """
        return self._rdd

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
        return string

