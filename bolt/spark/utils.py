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

def zip_with_index(rdd):
    """
    Alternate version of Spark's zipWithIndex that eagerly returns count.
    """
    starts = [0]
    if rdd.getNumPartitions() > 1:
        nums = rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
        count = sum(nums)
        for i in range(len(nums) - 1):
            starts.append(starts[-1] + nums[i])
    else:
        count = rdd.count()

    def func(k, it):
        for i, v in enumerate(it, starts[k]):
            yield v, i

    return count, rdd.mapPartitionsWithIndex(func)
