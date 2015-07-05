from __future__ import print_function
from numpy import ndarray, asarray, ufunc, prod
from bolt.base import BoltArray
from bolt.utils import inshape, tupleize
from functools import reduce


class BoltArrayLocal(ndarray, BoltArray):

    def __new__(cls, array):
        obj = asarray(array).view(cls)
        obj._mode = 'local'
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._mode = getattr(obj, 'mode', None)

    @property
    def _constructor(self):
        return BoltArrayLocal

    def _align(self, axes, key_shape=None):

        # ensure that the key axes are valid for an ndarray of this shape
        inshape(self.shape, axes)

        # compute the set of dimensions/axes that will be used to reshape
        remaining = [dim for dim in range(len(self.shape)) if dim not in axes]
        key_shape = key_shape if key_shape else [self.shape[axis] for axis in axes]
        remaining_shape = [self.shape[axis] for axis in remaining]
        linearized_shape = [prod(key_shape)] + remaining_shape

        # compute the transpose permutation
        transpose_order = axes + remaining

        # transpose the array so that the keys being mapped over come first, then linearize keys
        reshaped = self.transpose(*transpose_order).reshape(*linearized_shape)

        return reshaped

    def filter(self, func, axis=0):
        """
        """
        axes = sorted(tupleize(axis))
        reshaped = self._align(axes)

        filtered = asarray(list(filter(func, reshaped)))

        return self._constructor(filtered)

    def map(self, func, axis=0):
        """
        """

        axes = sorted(tupleize(axis))
        key_shape = [self.shape[axis] for axis in axes]
        reshaped = self._align(axes, key_shape=key_shape)

        mapped = asarray(list(map(func, reshaped)))
        elem_shape = mapped[0].shape

        # invert the previous reshape operation, using the shape of the map result
        linearized_shape_inv = key_shape + list(elem_shape)
        reordered = mapped.reshape(*linearized_shape_inv)

        return self._constructor(reordered)

    def reduce(self, func, axis=0):
        """
        """

        axes = sorted(tupleize(axis))

        # if the function is a ufunc, it can automatically handle reducing over multiple axes
        if isinstance(func, ufunc):
            inshape(self.shape, axes)
            reduced = func.reduce(self, axis=tuple(axes))
        else:
            reshaped = self._align(axes)
            reduced = reduce(func, reshaped)

        new_array = self._constructor(reduced)

        # ensure that the shape of the reduced array is valid
        expected_shape = [self.shape[i] for i in range(len(self.shape)) if i not in axes]
        if new_array.shape != tuple(expected_shape):
            raise ValueError("reduce did not yield a BoltArray with valid dimensions")

        return new_array

    def concatenate(self, arry, axis=0):
        if isinstance(arry, ndarray):
            from bolt import concatenate
            return concatenate((self, arry), axis)
        else:
            raise ValueError("other must be local array, got %s" % type(arry))

    def tospark(self, sc, axis=0):
        from bolt import array
        return array(self.toarray(), sc, axis=axis)

    def tordd(self, sc, axis=0):
        from bolt import array
        return array(self.toarray(), sc, axis=axis).tordd()

    def toarray(self):
        return asarray(self)

    def display(self):
        print(str(self))

    def __repr__(self):
        return BoltArray.__repr__(self)
