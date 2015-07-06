from numpy import arange, repeat
from bolt import array
from bolt.utils import allclose
import generic


def test_map():

    import random
    random.seed(42)

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x)

    # Test all generic map functionality
    generic.map_suite(x, b)


def test_reduce():

    from numpy import asarray

    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x,area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    b = array(arr)

    # Test all generic reduce functionality
    generic.reduce_suite(arr, b)


def test_filter():

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x)

    # Test all generic filter functionality
    generic.filter_suite(x, b)

def test_ufuncs():

    x = arange(2*3*4*5).reshape(2, 3, 4, 5)
    b = array(x)

    # test a common ufunc (sum) over different dimensions
    assert allclose(x.sum(axis=0), b.sum(axis=0).toarray())
    assert allclose(x.sum(axis=(0, 1)), b.sum(axis=(0, 1)).toarray())
    assert allclose(x.sum(axis=(0, 1, 2)), b.sum(axis=(0, 1, 2)).toarray())
    assert x.sum() == b.sum()



