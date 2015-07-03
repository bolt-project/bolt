from numpy import arange
from bolt import array


def test_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.shape == x.shape

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc)
    assert b.shape == x.shape

def test_size(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.size == x.size

def test_split(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.split == 1

    b = array(x, sc, axes=(0, 1))
    assert b.split == 2

def test_ndim(sc):

    x = arange(2**5).reshape(2, 2, 2, 2, 2)
    b = array(x, sc, axes=[0, 1, 2])

    assert b.keys.ndim == 3
    assert b.values.ndim == 2
    assert b.ndim == 5

def test_mask(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.mask == (1, 0, 0)

    b = array(x, sc, axes=(0, 1))
    assert b.mask == (1, 1, 0)

    b = array(x, sc, axes=(0, 1, 2))
    assert b.mask == (1, 1, 1)

def test_cache(sc):
    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    b.cache()
    assert b._rdd.is_cached
    b.unpersist()
    assert not b._rdd.is_cached

