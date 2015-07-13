from numpy import zeros, ones, asarray, r_, concatenate, arange, ceil, prod

from itertools import product

from bolt.utils import tuplesort
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
    _metadata = ['_shape', '_split', '_dtype']

    def __init__(self, rdd, shape=None, split=None, dtype=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

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

    def chunk(self, plan, kaxes=(), vaxes=()):
        """
        Break values of distributed array into chunks.

        Transforms an underlying pair RDD of (keys, values) into
        records of the form: (chunk id, stationary keys), (moving keys, chunked values).
        Here, chunk id, stationary keys, moving keys are all tuples.
        Chunked data is a subset of the data in each value, that has
        been sliced along 'chunk' lines. That is, for each
        value-dimnesion that is going to become a key-dimension, you
        break the value (i.e. the data in a single record) into chunks
        along those dimensions.

        Parameters
        ----------
        plan : array-like
            Number of chunks along each dimension

        kaxes : array-like
            Tuple of keys to move into values

        vaxes : array-like
            Tuple of values to move into keys
        """
        kmask, vmask = self.kmask(kaxes), self.vmask(vaxes)

        slices = self.getslices(plan, self.vshape)
        labels = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [list(zip(*s)) for s in labels]

        # this helper function returns a new pair rdd
        # keys = (chunk #, non-swapped keys)
        # values = (swapped keys, chunked data)
        def _chunk(record):
            k, v = record[0], record[1]
            k = asarray(k)
            stationary = tuple(k[~kmask])
            moving = k[kmask]
            for (chk, slc) in scheme:
                k = (tuple(asarray(chk)[vmask]), stationary)
                yield k, (moving, v[slc])

        rdd = self._rdd.flatMap(_chunk)
        return self._constructor(rdd).__finalize__(self)

    def unchunk(self, plan, kaxes=(), vaxes=()):
        """
        Convert a chunked array back into a full array with (key,value) pairs
        where key is a tuple of indicies, and value is an ndarray.

        Parameters
        ----------
        plan : array-like
            Number of chunks along each dimension

        kaxes : array-like
            Tuple of keys moved into values

        vaxes : array-like
            Tuple of values moved into keys
        """
        kshape, vshape = self.kshape, self.vshape
        kmask, vmask = self.kmask(kaxes), self.vmask(vaxes)

        sizes = self.getsizes(plan, vshape)

        moving_key_shape = kshape[kmask]

        mask = [False for _ in moving_key_shape]
        mask.extend([True if vmask[k] else False for k in range(len(vmask))])
        mask = asarray(mask)

        slices = [slice(0, i, 1) for i in moving_key_shape]
        slices.extend([None if vmask[i] else slice(0, vshape[i], 1) for i in range(len(vmask))])
        slices = asarray(slices)

        def _extract(record):

            k, v = record[0], record[1]

            chunk, stationary_key = k[0], k[1]
            key_offsets = prod([asarray(chunk), asarray(sizes)[vmask]], axis=0)
            moving_keys, values = zip(*v.data)
            sorted_keys = tuplesort([i.tolist() for i in moving_keys])
            values_sorted = asarray(values)[sorted_keys]
            expanded_shape = concatenate([moving_key_shape, values_sorted.shape[1:]])
            bounds = asarray(values_sorted[0].shape)[vmask]
            indices = list(product(*map(lambda x: arange(x), bounds)))
            values = values_sorted.reshape(expanded_shape)

            for b in indices:
                s = slices.copy()
                s[mask] = b
                yield (tuple(asarray(r_[stationary_key, key_offsets + b], dtype='int')), values[tuple(s)])

        rdd = self._rdd.groupByKey().flatMap(_extract)
        split = self._split - len(kaxes) + len(vaxes)
        shape = tuple(r_[kshape[~kmask], vshape[vmask],
                         kshape[kmask], vshape[~vmask]].astype('int'))

        return BoltArraySpark(rdd, shape=shape, split=split)

    def getplan(self, size=150, axes=None):
        """
        Identify a plan for chunking values along each dimension.

        Generates an ndarray with the number of chunks in each dimension.
        If provided, will estimate chunks for only a subset of axes,
        leaving all others as one.

        Parameters
        ----------
        size : integer or tuple
             If int, the average size of the chunks in all value dimensions.  
             If tuple, an explicit specification of the number chunks in 
             each moving value dimension.

        axis : tuple, optional, default=None
              One or more axes to estimate chunks for, if provided any
              other axes will use one chunk.
        """
        from numpy import dtype as gettype

        # initialize with ones
        plan = ones(len(self.vshape), dtype=int)

        # check for subset of axes
        if axes is None:
            axes = arange(len(self.vshape))
        else:
            axes = asarray(axes, 'int')

        if isinstance(size, tuple):
            plan[axes] = size

        elif isinstance(size, int):
            # convert from megabytes
            size *= 1000.0

            # calculate from dtype
            elsize = gettype(self.dtype).itemsize
            nelements = prod(self.vshape)
            dims = self.vshape[self.vmask(axes)]

            if size <= elsize:
                return dims

            remsize = 1.0 * nelements * elsize
            nchunks = ones(len(dims))
            for (i, d) in enumerate(dims):
                minsize = remsize/d
                if minsize >= size:
                    nchunks[i] = d
                    remsize = minsize
                    continue
                else:
                    nchunks[i] = ceil(remsize/size)
                    break

            plan[axes] = nchunks

        else:
            raise ValueError("Chunk size not understood, must be tuple or int")

        return plan

    @staticmethod
    def getslices(plan, dims):
        """
        Obtain slices for the given dimensions and chunks.

        Given a plan for the number of chunks along each dimension,
        calculate a list of slices required to generate those chunks.

        Parameters
        ----------
        plan : tuple or array-like
             Each entry contains the number of chunks along that dimension.
             Length must be equal to the number of dimensions.

        dims : tuple
             Dimensions of axes to be chunked.
        """
        slices = []
        for nchunks, d in zip(plan, dims):
            size = ceil(1.0 * d/nchunks)
            remainder = d % nchunks
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
    def getsizes(plan, dims):
        """
        Obtain sizes for the given dimensions and chunks.

        Given a plan for the number of chunks along each dimension,
        calculate the size of each chunk.

        Parameters
        ----------
        plan : tuple or array-like
             Each entry contains the number of chunks along that dimension.
             Length must be equal to the number of dimensions.

        dims : tuple
             Dimensions of axes to be chunked.
        """
        sizes = []
        for nchunks, d in zip(plan, dims):
            size = ceil(1.0 * d/nchunks)
            sizes.append(size)
        return sizes

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

    def __str__(self):
        s = "Chunked BoltArray\n"
        s += "shape: %s\n" % str(self.shape)
        return s

    def __repr__(self):
        return str(self)
