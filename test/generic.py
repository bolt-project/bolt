"""
Generic tests for all BoltArrays
"""
from bolt.utils import allclose
import pytest

def map_suite(arr, b):
    """
    A set of tests for the map operator

    Parameters
    ----------
    arr: `ndarray`
        A 2D array used in the construction of `b` (used to check results)
    b: `BoltArray`
        The BoltArray to be used for testing
    """

    from numpy import ones
    import random
    random.seed(42)

    # A simple map should be equivalent to an element-wise multiplication
    func1 = lambda x: x * 2
    mapped = b.map(func1, axes=(0,))
    res = mapped.toarray()
    print "res.shape: %s" % str(res.shape)
    assert allclose(res, arr * 2)

    # More complicated maps can reshape elements so long as they do so consistently
    func2 = lambda x: ones(10)
    mapped = b.map(func2, axes=(0,))
    res = mapped.toarray()
    assert res.shape == (arr.shape[0], 10)

    # But the shape of the result will change if mapped over different axes
    mapped = b.map(func2, axes=(0,1))
    res = mapped.toarray()
    assert res.shape == (arr.shape[0], arr.shape[1], 10)

    # If a map is not applied uniformly, it should produce an error
    with pytest.raises(Exception):
        def nonuniform_map(x):
            x.flags.writeable = False
            random.seed(x.data)
            return random.random()
        func3 = lambda x: ones(10) if nonuniform_map(x) < 0.5 else ones(5)
        mapped = b.map(func3)
        res = mapped.toarray()

def reduce_suite(arr, b):
    """
    A set of tests for the reduce operator

    Parameters
    ----------
    arr: `ndarray`
        A 3D ndarray used in the construction of `b` (used to check results)
    b: `BoltArray`
        The BoltArray to be used for testing
    """

    from numpy import ones, sum
    from operator import add

    # Reduce over the first axis with an add
    reduced = b.reduce(add, axes=(0,))
    res = reduced.toarray()
    assert res.shape == (arr.shape[1], arr.shape[2])
    assert allclose(res, sum(arr, 0))

    # Reduce over multiple axes with an add
    reduced = b.reduce(add, axes=(0, 1))
    res = reduced.toarray()
    assert res.shape == (arr.shape[2],)
    assert allclose(res, sum(sum(arr, 0), 1))

def filter_suite(arr, b):
    """
    A set of tests for the filter operator

    Parameters
    ----------
    arr: `ndarray`
        A 3D ndarray used in the construction of `b` (used to check results)
    b: `BoltArray`
        The BoltArray to be used for testing
    """

    import random
    random.seed(42)

    # Filter all values over the first axis
    filtered = b.filter(lambda x: False)
    res = filtered.toarray()
    assert res.shape == (0,)

    # Filter no values over the first axis
    filtered = b.filter(lambda x: True)
    res = filtered.toarray()
    assert res.shape == b.shape

    # Filter out half of the values over the first axis
    def filter_half(x):
        x.flags.writeable = False
        random.seed(x.data)
        return random.random()

    filtered = b.filter(lambda x: filter_half(x) < 0.5)
    res = filtered.toarray()
    assert res.shape[1:] == b.shape[1:]
    assert res.shape[0] <= b.shape[0]
