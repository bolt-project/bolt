from numpy import asarray, float64
from numpy import ones as npones
from numpy import zeros as npzeros
from numpy import empty as npempty
from numpy.random import random as nprandom

from itertools import product

from types import FunctionType

from bolt.local import BoltArrayLocal
from bolt.spark import BoltArraySpark

def _wrap(self, func_or_input, *args, **kwargs):
    context = kwargs.pop('context')
    split = kwargs.pop('split')

    if context is None:
        if isinstance(func_or_input, FunctionType): # this is a constructor function
            shape = args[0]
            dtype = kwargs.pop('dtype')
            order = kwargs.pop('order')
            return BoltArrayLocal(func_or_input(shape, order, dtype))
        else: # assume its an array and cast it
            return BoltArrayLocal(asarray(func_or_input))
    else:
        try:
            from pyspark import SparkContext
        except ImportError:
            print("Spark is not avaialble")
            return

        if isinstance(func_or_input, FunctionType): # this is a constructor function
            shape = args[0]
            dtype = kwargs.pop('dtype')
            order = kwargs.pop('order')

            # make the keys
            key_shape = shape[:split]
            val_shape = shape[split:]
            rdd = context.parallelize(list(product(*[range(x) for x in key_shape])))
            # use a map to make the arrays in parallel
            rdd = rdd.map(lambda x:(x, func_or_input(val_shape, dtype, order)))
            return BoltArraySpark(rdd, shape=shape, split=split)
        else: # assume its an array and cast it
            return BoltArraySpark.fromarray(asarray(func_or_input), context=context, split=split)

def array(input, context=None, split=1):
    return _wrap(input, context, split)

def ones(shape, context=None, split=1, dtype=float64, order='C'):
    return _wrap(npones, shape, context, split, dtype, order)

def zeros(shape, context=None, split=1, dtype=float64, order='C'):
    return _wrap(npzeros, shape, context, split, dtype, order)                

def random(shape, context=None, split=1, dtype=float64, order='C'):
    return _wrap(nprandom, shape, context, split, dtype, order)                

def empty(shape, context=None, split=1,  dtype=float64, order='C'):
    return _wrap(npempty, shape, context, split, dtype, order)
