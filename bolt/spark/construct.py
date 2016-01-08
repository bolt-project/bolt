from numpy import unravel_index, prod, arange, asarray, float64

from itertools import product

from bolt.construct import ConstructBase
from bolt.spark.array import BoltArraySpark
from bolt.spark.utils import get_kv_shape, get_kv_axes


class ConstructSpark(ConstructBase):

    @staticmethod
    def array(a, context=None, axis=(0,), dtype=None, npartitions=None):
        """
        Create a spark bolt array from a local array.

        Parameters
        ----------
        a : array-like
            An array, any object exposing the array interface, an
            object whose __array__ method returns an array, or any
            (nested) sequence.

        context : SparkContext
            A context running Spark. (see pyspark)

        axis : tuple, optional, default=(0,)
            Which axes to distribute the array along. The resulting
            distributed object will use keys to represent these axes,
            with the remaining axes represented by values.

        dtype : data-type, optional, default=None
            The desired data-type for the array. If None, will
            be determined from the data. (see numpy)

        npartitions : int
            Number of partitions for parallization.

        Returns
        -------
        BoltArraySpark
        """
        if dtype is None:
            arry = asarray(a)
            dtype = arry.dtype
        else:
            arry = asarray(a, dtype)
        shape = arry.shape
        ndim = len(shape)

        # handle the axes specification and transpose if necessary
        axes = ConstructSpark._format_axes(axis, arry.shape)
        key_axes, value_axes = get_kv_axes(arry.shape, axes)
        permutation = key_axes + value_axes
        arry = arry.transpose(*permutation)
        split = len(axes)

        if split < 1:
            raise ValueError("split axis must be greater than 0, got %g" % split)
        if split > len(shape):
            raise ValueError("split axis must not exceed number of axes %g, got %g" % (ndim, split))

        key_shape = shape[:split]
        val_shape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(key_shape))), key_shape))
        vals = arry.reshape((prod(key_shape),) + val_shape)

        rdd = context.parallelize(zip(keys, vals), npartitions)
        return BoltArraySpark(rdd, shape=shape, split=split, dtype=dtype)

    @staticmethod
    def ones(shape, context=None, axis=(0,), dtype=float64, npartitions=None):
        """
        Create a spark bolt array of ones.

        Parameters
        ----------
        shape : tuple
            The desired shape of the array.

        context : SparkContext
            A context running Spark. (see pyspark)

        axis : tuple, optional, default=(0,)
            Which axes to distribute the array along. The resulting
            distributed object will use keys to represent these axes,
            with the remaining axes represented by values.

        dtype : data-type, optional, default=float64
            The desired data-type for the array. If None, will
            be determined from the data. (see numpy)

        npartitions : int
            Number of partitions for parallization.

        Returns
        -------
        BoltArraySpark
        """
        from numpy import ones
        return ConstructSpark._wrap(ones, shape, context, axis, dtype, npartitions)

    @staticmethod
    def zeros(shape, context=None, axis=(0,), dtype=float64, npartitions=None):
        """
        Create a spark bolt array of zeros.

        Parameters
        ----------
        shape : tuple
            The desired shape of the array.

        context : SparkContext
            A context running Spark. (see pyspark)

        axis : tuple, optional, default=(0,)
            Which axes to distribute the array along. The resulting
            distributed object will use keys to represent these axes,
            with the remaining axes represented by values.

        dtype : data-type, optional, default=float64
            The desired data-type for the array. If None, will
            be determined from the data. (see numpy)

        npartitions : int
            Number of partitions for parallization.

        Returns
        -------
        BoltArraySpark
        """
        from numpy import zeros
        return ConstructSpark._wrap(zeros, shape, context, axis, dtype, npartitions)

    @staticmethod
    def concatenate(arrays, axis=0):
        """
        Join two bolt arrays together, at least one of which is in spark.

        Parameters
        ----------
        arrays : tuple
            A pair of arrays. At least one must be a spark array,
            the other can be a local bolt array, a local numpy array,
            or an array-like.

        axis : int, optional, default=0
            The axis along which the arrays will be joined.

        Returns
        -------
        BoltArraySpark
        """
        if not isinstance(arrays, tuple):
            raise ValueError("data type not understood")
        if not len(arrays) == 2:
            raise NotImplementedError("spark concatenation only supports two arrays")

        first, second = arrays
        if isinstance(first, BoltArraySpark):
            return first.concatenate(second, axis)
        elif isinstance(second, BoltArraySpark):
            first = ConstructSpark.array(first, second._rdd.context)
            return first.concatenate(second, axis)
        else:
            raise ValueError("at least one array must be a spark bolt array")

    @staticmethod
    def _argcheck(*args, **kwargs):
        """
        Check that arguments are consistent with spark array construction.

        Conditions are:
        (1) a positional argument is a SparkContext
        (2) keyword arg 'context' is a SparkContext
        (3) an argument is a BoltArraySpark, or
        (4) an argument is a nested list containing a BoltArraySpark
        """
        try:
            from pyspark import SparkContext
        except ImportError:
            return False

        cond1 = any([isinstance(arg, SparkContext) for arg in args])
        cond2 = isinstance(kwargs.get('context', None), SparkContext)
        cond3 = any([isinstance(arg, BoltArraySpark) for arg in args])
        cond4 = any([any([isinstance(sub, BoltArraySpark) for sub in arg])
                     if isinstance(arg, (tuple, list)) else False for arg in args])
        return cond1 or cond2 or cond3 or cond4

    @staticmethod
    def _format_axes(axes, shape):
        """
        Format target axes given an array shape
        """
        if isinstance(axes, int):
            axes = (axes,)
        elif isinstance(axes, list) or hasattr(axes, '__iter__'):
            axes = tuple(axes)
        if not isinstance(axes, tuple):
            raise ValueError("axes argument %s in the constructor not specified correctly" % str(axes))
        if min(axes) < 0 or max(axes) > len(shape) - 1:
            raise ValueError("invalid key axes %s given shape %s" % (str(axes), str(shape)))
        return axes

    @staticmethod
    def _wrap(func, shape, context=None, axis=(0,), dtype=None, npartitions=None):
        """
        Wrap an existing numpy constructor in a parallelized construction
        """
        if isinstance(shape, int):
            shape = (shape,)
        key_shape, value_shape = get_kv_shape(shape, ConstructSpark._format_axes(axis, shape))
        split = len(key_shape)

        # make the keys
        rdd = context.parallelize(list(product(*[arange(x) for x in key_shape])), npartitions)

        # use a map to make the arrays in parallel
        rdd = rdd.map(lambda x: (x, func(value_shape, dtype, order='C')))
        return BoltArraySpark(rdd, shape=shape, split=split, dtype=dtype)
