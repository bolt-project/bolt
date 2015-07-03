from numpy import zeros, ones, asarray, r_, concatenate, arange, ceil, prod

from itertools import product

from bolt.utils import tuplesort


class Swapper(object):
    """
    Class for handling swap operations
    """
    def __init__(self, key, value, dtype, size=150):
        self.key = key
        self.value = value
        plan = self.getplan(size, dtype)
        self.slices, self.chunk_sizes = self.getslices(plan, self.value.shape)

    def getshape(self):
        """
        Get resulting shape after swapping
        """
        return r_[self.key.shape[~self.key.mask], self.value.shape[self.value.mask],
                  self.key.shape[self.key.mask], self.value.shape[~self.value.mask]]

    def chunk(self, rdd):
        """
        Convert values of bolt spark array into chunks
        """
        kmask, vmask, slices = self.key.mask, self.value.mask, self.slices

        labeled_slices = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [zip(*s) for s in labeled_slices]

        def _chunk(record):
            k, v = record[0], record[1]
            k = asarray(k)
            stationary = tuple(k[~kmask])
            moving = k[kmask]
            for (chk, slc) in scheme:
                k = (tuple(asarray(chk)[vmask]), stationary)
                yield k, (moving, v[slc])

        return rdd.flatMap(_chunk).groupByKey()

    def extract(self, rdd):
        """
        Extract values from chunks
        """
        kmask, vmask = self.key.mask, self.value.mask
        kshape, vshape = self.key.shape, self.value.shape
        chunk_sizes = self.chunk_sizes

        moving_key_shape = kshape[kmask]

        mask = [False for _ in moving_key_shape]
        mask.extend([True if vmask[k] else False for k in xrange(len(vmask))])
        mask = asarray(mask)

        slices = [slice(0, i, 1) for i in moving_key_shape]
        slices.extend([None if vmask[i] else slice(0, vshape[i], 1) for i in xrange(len(vmask))])
        slices = asarray(slices)

        def _extract(record):

            k, v = record[0], record[1]

            chunk, stationary_key = k[0], k[1]
            key_offsets = prod([asarray(chunk), asarray(chunk_sizes)[vmask]], axis=0)
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
                yield (tuple(asarray(r_[stationary_key, key_offsets + b], dtype='int')), values[s.tolist()])

        return rdd.flatMap(_extract)

    def getplan(self, size, dtype):
        """
        Identify the plan for chunking along each dimension
        """
        from numpy import dtype as gettype
        plan = ones(len(self.value.shape), dtype=int)

        if isinstance(size, tuple):
            plan[self.value.axes] = size

        else:
            # convert from megabytes
            size *= 1000.0

            # calculate from dtype
            element_size = gettype(dtype).itemsize
            nelements = prod(self.value.shape)
            total_size = nelements * element_size
            moving_value_shapes = self.value.shape[self.value.mask]

            if size <= element_size:
                return moving_value_shapes

            remaining_size = 1.0*total_size
            nchunks = ones(len(moving_value_shapes))
            for (i, s) in enumerate(moving_value_shapes):
                min_chunk_size = remaining_size/s
                if min_chunk_size >= size:
                    nchunks[i] = s
                    remaining_size = min_chunk_size
                    continue
                else:
                    nchunks[i] = ceil(remaining_size/size)
                    break

            plan[self.value.axes] = nchunks

        return plan

    @staticmethod
    def getslices(plan, dims):
        """
        Obtain slices for the given dimensions and chunks
        """
        slices = []
        sizes = []
        for nchunks, d in zip(plan, dims):
            size = ceil(1.0*d/nchunks)
            sizes.append(size)
            chunk_remainder = d % nchunks
            start = 0
            dim_slices = []
            for idx in xrange(nchunks):
                end = start + size
                dim_slices.append(slice(start, end, 1))
                start = end
            if chunk_remainder:
                dim_slices.append(slice(end, d, 1))
            slices.append(dim_slices)
        return slices, sizes


class Dims(object):
    """
    Class for storing properties associated with dimensionality
    """
    def __init__(self, axes, shape):
        self.axes = asarray(axes, 'int')
        self.shape = asarray(shape)

    @property
    def mask(self):
        mask = zeros(len(self.shape), dtype=bool)
        mask[self.axes] = True
        return mask
