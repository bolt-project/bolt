from numpy import zeros, ones, asarray, r_, concatenate, arange, ceil, prod, empty, mod, floor, any, logical_and

from itertools import product

from bolt.utils import tuplesort, tupleize, allstack
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
    def plan(self):
        return self._plan
    
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

    def chunk(self, size, axis=None):
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
        axis = tupleize(axis)
        plan = self.getplan(size, axis)
        slices = self.getslices(plan, self.vshape)
        labels = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [list(zip(*s)) for s in labels]

        def _chunk(record):
            k, v = record[0], record[1]
            for (chk, slc) in scheme:
                yield (k, chk), v[slc]

        rdd = self._rdd.flatMap(_chunk)
        return self._constructor(rdd, plan=plan).__finalize__(self)

    def unchunk(self):
        """
        Convert a chunked array back into a full array with (key,value) pairs
        where key is a tuple of indicies, and value is an ndarray.
        """
        plan, vshape = self.plan, self.vshape
        nchunks = self.getnumber(plan, vshape)

        full_shape = concatenate((nchunks, plan))
        n = len(vshape)
        perm = concatenate(zip(arange(n), range(n, 2*n)))

        if sum(mod(vshape, plan)) == 0:
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
        return BoltArraySpark(rdd, shape=self.shape, split=self._split)

    def move(self, kaxes=(), vaxes=()):
        """
        Move some of the dimensions in the values into the keys.

        Along with chunking, this is required as part of the swap method
        on the BoltArraySpark, see there for further details.

        Parameters
        ----------
        kaxes : tuple, optional, default=()
            Axes from keys to move to values.

        vaxes : tuple, optional, default=()
            Axes from values to move to keys.
        """
        kmask, vmask = self.kmask(kaxes), self.vmask(vaxes)
        
        # make sure chunking is done only on the moving dimensions 
        nchunks = asarray(self.getnumber(self.plan, self.vshape))
        if any(logical_and(nchunks != 1, ~vmask)):
            raise NotImplementedError("Chunking along non-swapped axes is not supported")

        def _move(record):
            k, chk, data = asarray(record[0]), asarray(record[1][0]), asarray(record[1][1])
            stationary_keys, moving_keys = tuple(k[~kmask]), tuple(k[kmask])
            moving_values, stationary_values = tuple(chk[vmask]), tuple(chk[~vmask])
            return (stationary_keys, moving_values), (moving_keys + stationary_values, data)

        switch = self.switch
        rdd = self._rdd.map(switch).map(_move)

        moving_kshape = tuple(self.kshape[kmask])

        def _rebuild(v):
            idx, data = zip(*v.data)
            valshape = data[0].shape
            fullshape = moving_kshape + valshape 
            sorted_idx = tuplesort(idx)
            return asarray(data)[sorted_idx].reshape(fullshape)

        rdd = rdd.groupByKey().mapValues(_rebuild)
            
        mask = [False for _ in moving_kshape]
        mask.extend([True if vmask[k] else False for k in range(len(vmask))])
        mask = asarray(mask)

        slices = [slice(0, i, 1) for i in moving_kshape]
        slices.extend([None if vmask[i] else slice(0, self.vshape[i], 1) for i in range(len(vmask))])
        slices = asarray(slices)

        sizes = self.plan

        def _extract(record):

            k, v = record[0], record[1]

            stationary_key, chunk = k[0], k[1]
            key_offsets = prod([asarray(chunk), asarray(sizes)[vmask]], axis=0)

            bounds = asarray(v.shape[len(kaxes):])[vmask]
            indices = list(product(*map(lambda x: arange(x), bounds)))

            for b in indices:
                s = slices.copy()
                s[mask] = b
                yield (tuple(asarray(r_[stationary_key, key_offsets + b], dtype='int')), v[tuple(s)])
            
        rdd = rdd.flatMap(_extract)
        split = self._split - len(kaxes) + len(vaxes)
        shape = tuple(r_[self.kshape[~kmask], self.vshape[vmask],
                         self.kshape[kmask], self.vshape[~vmask]].astype('int'))
        return BoltArraySpark(rdd, shape=shape, split=split)

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

        axis : tuple, optional, default=None
              One or more axes to estimate chunks for, if provided any
              other axes will use one chunk.
        """
        from numpy import dtype as gettype

        # initialize with all elements in one chunk
        plan = self.vshape

        # check for subset of axes
        if axes is None:
            axes = arange(len(self.vshape))
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
        chunk_sizes: tuple or array-like
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
        chunk_sizes: tuple or array-like
            Size of chunks (in number of elements) along each dimensions.
            Length must be equal to the number of dimensions.

        dims : tuple
             Dimensions of axes to be chunked.
        """
        slices = []
        for size, d in zip(plan, shape):
            nchunks = d/size #integer division
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

    def __str__(self):
        s = "Chunked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        return s

    def __repr__(self):
        return str(self)
