from bolt.utils import tupleize

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

def func_axes(b, axis, noswap):
    if axis is None:
        axis = 0
    axis = tupleize(axis)
    if noswap:
        key_axes = tuple(range(b.split))
        if axis != key_axes:
            raise ValueError("axis must match key axes if noswap == True")
    return sorted(axis)

def reducer_axes(b, axis):
    if axis is None:
        axis = range(len(b.shape))
    return tupleize(axis)

def extract_scalar(b):
    if b.shape == ():
        return b.toarray().reshape(1)[0]
    return b

def zip_with_index(rdd):
    """
    Alternate version of Spark's zipWithIndex that eagerly returns count.
    """
    starts = [0]
    count = None
    if rdd.getNumPartitions() > 1:
        nums = rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
        count = sum(nums)
        for i in range(len(nums) - 1):
            starts.append(starts[-1] + nums[i])

    def func(k, it):
        for i, v in enumerate(it, starts[k]):
            yield v, i

    return count, rdd.mapPartitionsWithIndex(func)
