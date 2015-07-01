from numpy import float64, unravel_index, prod, arange, asarray

from itertools import product

from bolt.construct import ConstructBase
from bolt.spark.array import BoltArraySpark
from bolt.spark.utils import get_kv_shape, get_kv_axes


class ConstructSpark(ConstructBase):

    @staticmethod
    def array(arry, context=None, axes=(0,)):
        arry = asarray(arry)
        shape = arry.shape
        ndim = len(shape)

        # handle the axes specification and transpose if necessary
        axes = ConstructSpark._format_axes(axes, arry.shape)
        key_axes, value_axes = get_kv_axes(arry.shape, axes)
        permutation = key_axes + value_axes
        arry = arry.transpose(*permutation)
        split = len(axes)

        if split < 1:
            raise ValueError("Split axis must be greater than 0, got %g" % split)
        if split > len(shape):
            raise ValueError("Split axis must not exceed number of axes %g, got %g" % (ndim, split))

        key_shape = shape[:split]
        val_shape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(key_shape))), key_shape))
        vals = arry.reshape((prod(key_shape),) + val_shape)

        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd, shape=shape, split=split)

    @staticmethod
    def ones(shape, context=None, axes=(0,), dtype=float64, order='C'):
        from numpy import ones
        return ConstructSpark._wrap(ones, shape, context, axes, dtype, order)

    @staticmethod
    def zeros(shape, context=None, axes=(0,), dtype=float64, order='C'):
        from numpy import zeros
        return ConstructSpark._wrap(zeros, shape, context, axes, dtype, order)

    @staticmethod
    def argcheck(*args, **kwargs):

        try:
            from pyspark import SparkContext
        except ImportError:
            return False

        cond1 = any([isinstance(arg, SparkContext) for arg in args])
        cond2 = isinstance(kwargs.get('context', None), SparkContext)
        return cond1 or cond2

    @staticmethod
    def _format_axes(axes, shape):
        if isinstance(axes, int):
            axes = (axes,)
        elif isinstance(axes, list):
            axes = tuple(axes)
        if not isinstance(axes, tuple):
            raise ValueError("axes argument %s in the constructor not specified correctly" % str(axes))
        if min(axes) < 0 or max(axes) > len(shape) - 1:
            raise ValueError("invalid key axes %s given shape %s" % (str(axes), str(shape)))
        return axes

    @staticmethod
    def _wrap(func, shape, context=None, axes=(0,), dtype=float64, order='C'):

        key_shape, value_shape = get_kv_shape(shape, ConstructSpark._format_axes(axes, shape))
        split = len(key_shape)

        # make the keys
        rdd = context.parallelize(list(product(*[arange(x) for x in key_shape])))

        # use a map to make the arrays in parallel
        rdd = rdd.map(lambda x: (x, func(value_shape, dtype, order)))
        return BoltArraySpark(rdd, shape=shape, split=split)
