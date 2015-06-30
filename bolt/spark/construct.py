from numpy import float64, unravel_index, prod, arange, asarray

from itertools import product

from bolt.construct import ConstructBase
from bolt.spark.spark import BoltArraySpark

class ConstructSpark(ConstructBase):

    @staticmethod
    def array(arry, context=None, split=1):
        arry = asarray(arry)
        shape = arry.shape
        ndim = len(shape)

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
    def ones(shape, context=None, split=1, dtype=float64, order='C'):
        from numpy import ones
        return ConstructSpark._wrap(ones, shape, context, split, dtype, order)

    @staticmethod
    def zeros(shape, context=None, split=1, dtype=float64, order='C'):
        from numpy import zeros
        return ConstructSpark._wrap(zeros, shape, context, split, dtype, order)

    @staticmethod
    def _wrap(func, shape, context=None, split=1, dtype=float64, order='C'):

        # make the keys
        key_shape = shape[:split]
        val_shape = shape[split:]
        rdd = context.parallelize(list(product(*[arange(x) for x in key_shape])))

        # use a map to make the arrays in parallel
        rdd = rdd.map(lambda x: (x, func(val_shape, dtype, order)))
        return BoltArraySpark(rdd, shape=shape, split=split)

    @staticmethod
    def argcheck(*args, **kwargs):

        try:
            from pyspark import SparkContext
        except ImportError:
            return False

        cond1 = any([isinstance(arg, SparkContext) for arg in args])
        cond2 = isinstance(kwargs.get('context', None), SparkContext)
        return cond1 or cond2
