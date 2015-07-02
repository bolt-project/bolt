from numpy import zeros, ones, asarray, r_, concatenate, arange, dtype, ceil, prod

from itertools import product

from bolt.utils import tuplesort

class Chunks(object):

    def __init__(self, key_axes, value_axes, key_shape, value_shape, type, size=150):
        self.key_axes = key_axes
        self.value_axes = value_axes
        self.key_shape = asarray(key_shape)
        self.value_shape = asarray(value_shape)
        self.type = type
        self.size = size
        self.chunks_per_dim = self.compute_chunks(self.size, self.type)
        self.slices, self.chunk_sizes = self.getslices(self.chunks_per_dim, self.value_shape)

    @property
    def key_mask(self):
        key_mask = zeros(len(self.key_shape), dtype=bool)
        key_mask[self.key_axes] = True
        return key_mask

    @property
    def value_mask(self):
        value_mask = zeros(len(self.value_shape), dtype=bool)
        value_mask[self.value_axes] = True
        return value_mask

    def compute_chunks(self, size, type):

        chunks_per_dim = ones(len(self.value_shape), dtype=int)

        if isinstance(size, tuple):
            chunks_per_dim[self.value_axes] = size

        else:
            element_size = dtype(type).itemsize
            nelements = prod(self.value_shape)
            total_size = nelements * element_size
            moving_value_shapes = self.value_shape[self.value_mask]

            size *= 1000.0 # convert from megabytes

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

            chunks_per_dim[self.value_axes] = nchunks

        return chunks_per_dim

    def chunk(self, rdd):

        key_mask, value_mask = self.key_mask, self.value_mask
        slices, chunk_sizes = self.slices, self.chunk_sizes

        labeled_slices = list(product(*[list(enumerate(s)) for s in slices]))
        scheme = [zip(*s) for s in labeled_slices]

        def _chunk(record):
            k, v = record[0], record[1]
            k = asarray(k)
            stationary = tuple(k[~key_mask])
            moving = k[key_mask]
            for (chk, slc) in scheme:
                k = (tuple(asarray(chk)[value_mask]), stationary)
                yield k, (moving, v[slc])

        return rdd.flatMap(_chunk).groupByKey()

    def extract(self, rdd):

        key_mask, value_mask = self.key_mask, self.value_mask
        key_shape, value_shape = self.key_shape, self.value_shape
        chunk_sizes = self.chunk_sizes

        moving_key_shape, stationary_key_shape = key_shape[key_mask], key_shape[~key_mask]

        mask = [False for _ in moving_key_shape]
        mask.extend([True if value_mask[k] else False for k in xrange(len(value_mask))])
        mask = asarray(mask)

        slices = [slice(0, i, 1) for i in moving_key_shape]
        slices.extend([None if value_mask[i] else slice(0, value_shape[i], 1) for i in xrange(len(value_mask))])
        slices = asarray(slices)

        def _extract(record):

            k, v = record[0], record[1]

            chunk, stationary_key = k[0], k[1]
            key_offsets = prod([asarray(chunk), asarray(chunk_sizes)[value_mask]], axis=0)

            moving_keys, values = zip(*v.data)
            sorted_keys = tuplesort([i.tolist() for i in moving_keys])

            values_sorted = asarray(values)[sorted_keys]

            expanded_shape = concatenate([moving_key_shape, values_sorted.shape[1:]])

            values = values_sorted.reshape(expanded_shape)

            bounds = asarray(values_sorted[0].shape)[value_mask]
            indices = list(product(*map(lambda x: arange(x), bounds)))

            for b in indices:
                s = slices.copy()
                s[mask] = b
                yield (tuple(asarray(r_[stationary_key, key_offsets + b], dtype='int')), values[s.tolist()])

        return rdd.flatMap(_extract)

    @staticmethod
    def getslices(chunks_per_dim, dims):
        # slices will be sequence of sequences of slices
        # slices[i] will hold slices for ith dimension
        slices = []
        chunk_sizes = []
        for nchunks, dim_size in zip(chunks_per_dim, dims):
            chunk_size = ceil(1.0*dim_size/nchunks)
            chunk_sizes.append(chunk_size)
            chunk_remainder = dim_size % nchunks
            start = 0
            dim_slices = []
            for idx in xrange(nchunks):
                end = start + chunk_size
                dim_slices.append(slice(start, end, 1))
                start = end
            if chunk_remainder:
                dim_slices.append(slice(end, dim_size, 1))
            slices.append(dim_slices)
        return slices, chunk_sizes
