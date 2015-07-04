from numpy import float64, asarray

from bolt.construct import ConstructBase
from bolt.local.array import BoltArrayLocal


class ConstructLocal(ConstructBase):

    @staticmethod
    def array(a, dtype=None, order='C'):
        """
        Create a local bolt array.

        Parameters
        ----------
        a : array-like
            An array, any object exposing the array interface, an
            object whose __array__ method returns an array, or any
            (nested) sequence.

        dtype : data-type, optional, default=None
            The desired data-type for the array. If None, will
            be determined from the data. (see numpy)

        order : {'C', 'F', 'A'}, optional, default='C'
            The order of the array. (see numpy)

        Returns
        -------
        BoltArrayLocal
        """
        return BoltArrayLocal(asarray(a, dtype, order))

    @staticmethod
    def ones(shape, dtype=float64, order='C'):
        """
        Create a local bolt array of ones.

        Parameters
        ----------
        shape : tuple
            Dimensions of the desired array

        dtype : data-type, optional, default=float64
            The desired data-type for the array. (see numpy)

        order : {'C', 'F', 'A'}, optional, default='C'
            The order of the array. (see numpy)

        Returns
        -------
        BoltArrayLocal
        """
        from numpy import ones
        return ConstructLocal._wrap(ones, shape, dtype, order)

    @staticmethod
    def zeros(shape, dtype=float64, order='C'):
        """
        Create a local bolt array of zeros.

        Parameters
        ----------
        shape : tuple
            Dimensions of the desired array.

        dtype : data-type, optional, default=float64
            The desired data-type for the array. (see numpy)

        order : {'C', 'F', 'A'}, optional, default='C'
            The order of the array. (see numpy)

        Returns
        -------
        BoltArrayLocal
        """
        from numpy import zeros
        return ConstructLocal._wrap(zeros, shape, dtype, order)

    @staticmethod
    def _wrap(func, shape, dtype, order):
        return BoltArrayLocal(func(shape, dtype, order))

    @staticmethod
    def concatenate(arrays, axis=0):
        """
        Join a sequence of arrays together.

        Parameters
        ----------
        arrays : tuple
            A sequence of array-like e.g. (a1, a2, ...)

        axis : int, optional, default=0
            The axis along which the arrays will be joined.

        Returns
        -------
        BoltArrayLocal
        """
        if not isinstance(arrays, tuple):
            raise ValueError("data type not understood")
        arrays = tuple([asarray(a) for a in arrays])
        from numpy import concatenate
        return BoltArrayLocal(concatenate(arrays, axis))