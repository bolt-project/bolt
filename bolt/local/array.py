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

    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()]
        else:
            return ndarray.__array_wrap__(self, obj)

    @property
    def _constructor(self):
        return BoltArrayLocal

    def _align(self, axes, key_shape=None):
        """
        Align local bolt array so that axes for iteration are in the keys.

        This operation is applied before most functional operators.
        It ensures that the specified axes are valid, and might transpose/reshape
        the underlying array so that the functional operators can be applied
        over the correct records.

        Parameters
        ----------
        axes: tuple[int]
            One or more axes that will be iterated over by a functional operator

        Returns
        -------
        BoltArrayLocal
        """

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

    def filter(self, func, axis=(0,)):
        """
        Filter array along an axis.

        Applies a function which should evaluate to boolean,
        along a single axis or multiple axes. Array will be
        aligned so that the desired set of axes are in the
        keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function to apply, should return boolean

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to filter along.

        Returns
        -------
        BoltArrayLocal
        """
        axes = sorted(tupleize(axis))
        reshaped = self._align(axes)

        filtered = asarray(list(filter(func, reshaped)))

        return self._constructor(filtered)

    def map(self, func, axis=(0,)):
        """
        Apply a function across an axis.

        Array will be aligned so that the desired set of axes
        are in the keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of a single array to apply

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to apply function along.

        Returns
        -------
        BoltArrayLocal
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
        Reduce an array along an axis.

        Applies an associative/commutative function of two arguments
        cumulatively to all arrays along an axis. Array will be aligned
        so that the desired set of axes are in the keys, which may
        require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of two arrays that returns a single array

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to reduce along.

        Returns
        -------
        BoltArrayLocal
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

    def first(self):
        """
        Return first element of the array
        """
        return self[0]

    def concatenate(self, arry, axis=0):
        """
        Join this array with another array.

        Paramters
        ---------
        arry : ndarray or BoltArrayLocal
            Another array to concatenate with

        axis : int, optional, default=0
            The axis along which arrays will be joined.

        Returns
        -------
        BoltArrayLocal
        """
        if isinstance(arry, ndarray):
            from bolt import concatenate
            return concatenate((self, arry), axis)
        else:
            raise ValueError("other must be local array, got %s" % type(arry))

    def toscalar(self):
        """
        Returns the single scalar element contained in an array of shape (), if
        the array has that shape. Returns self otherwise.
        """
        if self.shape == ():
            return self.toarray().reshape(1)[0]
        else:
            return self

    def tospark(self, sc, axis=0):
        """
        Converts a BoltArrayLocal into a BoltArraySpark

        Parameters
        ----------
        sc : SparkContext
            The SparkContext which will be used to create the BoltArraySpark

        axis : tuple or int, optional, default=0
            The axis (or axes) across which this array will be parallelized

        Returns
        -------
        BoltArraySpark
        """
        from bolt import array
        return array(self.toarray(), sc, axis=axis)

    def tordd(self, sc, axis=0):
        """
        Converts a BoltArrayLocal into an RDD

        Parameters
        ----------
        sc : SparkContext
            The SparkContext which will be used to create the BoltArraySpark

        axis : tuple or int, optional, default=0
            The axis (or axes) across which this array will be parallelized

        Returns
        -------
        RDD[(tuple, ndarray)]
        """
        from bolt import array
        return array(self.toarray(), sc, axis=axis).tordd()

    def toarray(self):
        """
        Returns the underlying ndarray wrapped by this BoltArrayLocal
        """
        return asarray(self)

    def display(self):
        """
        Show a pretty-printed representation of this BoltArrayLocal
        """
        print(str(self))

    def __repr__(self):
        return BoltArray.__repr__(self)
