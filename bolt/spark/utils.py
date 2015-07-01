"""
Spark-specific utility functions
"""

def get_kv_shape(shape, key_axes):
    func = lambda axis: shape[axis]
    return _get_kv_func(func, shape, key_axes)

def get_kv_axes(shape, key_axes):
    func = lambda axis: axis
    return _get_kv_func(func, shape, key_axes)

def _get_kv_func(func, shape, key_axes):
    key_res = [func(axis) for axis in key_axes]
    value_res = [func(axis) for axis in range(len(shape)) if axis not in key_axes]
    return key_res, value_res
