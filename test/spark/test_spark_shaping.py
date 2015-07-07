import pytest
from numpy import arange, prod
from itertools import permutations
from bolt import array
from bolt.utils import allclose

def test_value_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.values.shape == (3,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axis=0)
    assert b.values.shape == (3, 4)

def test_key_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.keys.shape == (2,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axis=(0, 1))
    assert b.keys.shape == (2, 3)

def test_reshape_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0, 1))
    c = b.keys.reshape((3, 2))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.reshape((3, 2, 4)))

    b = array(x, sc, axis=0)
    c = b.keys.reshape((2, 1))
    assert allclose(c.toarray(), x.reshape((2, 1, 3, 4)))

    b = array(x, sc, axis=(0,))
    c = b.keys.reshape((2,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axis=(0, 1))
    c = b.keys.reshape((2, 3))
    assert allclose(c.toarray(), x)

def test_reshape_keys_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0, 1))
    with pytest.raises(ValueError):
        b.keys.reshape((2, 3, 4))

def test_reshape_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0,))
    c = b.values.reshape((4, 3))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.reshape((2, 4, 3)))

    b = array(x, sc, axis=(0, 1))
    c = b.values.reshape((1, 4))
    assert allclose(c.toarray(), x.reshape((2, 3, 1, 4)))

    b = array(x, sc, axis=(0, 1))
    c = b.values.reshape((4,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axis=0)
    c = b.values.reshape((3, 4))
    assert allclose(c.toarray(), x)

def test_reshape_values_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0, 1))
    with pytest.raises(ValueError):
        b.values.reshape((2, 3, 4))

def test_transpose_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0, 1))
    c = b.keys.transpose((1, 0))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.transpose((1, 0, 2)))

    b = array(x, sc, axis=0)
    c = b.keys.transpose((0,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axis=(0, 1))
    c = b.keys.transpose((0, 1))
    assert allclose(c.toarray(), x)

def test_transpose_keys_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=(0, 1))
    with pytest.raises(ValueError):
        b.keys.transpose((0, 2))

    with pytest.raises(ValueError):
        b.keys.transpose((1, 1))

    with pytest.raises(ValueError):
        b.keys.transpose((0,))

def test_transpose_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=0)
    c = b.values.transpose((1, 0))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.transpose((0, 2, 1)))

    b = array(x, sc, axis=0)
    c = b.values.transpose((0, 1))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axis=(0, 1))
    c = b.values.transpose((0,))
    assert allclose(c.toarray(), x.reshape((2, 3, 4)))

def test_traspose_values_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axis=0)
    with pytest.raises(ValueError):
        b.values.transpose((0, 2))

    with pytest.raises(ValueError):
        b.values.transpose((1, 1))

    with pytest.raises(ValueError):
        b.values.transpose((0,))


def test_swap(sc):

    a = arange(2**8).reshape(*(8*[2]))
    b = array(a, sc, axis=(0, 1, 2, 3))

    bs = b.swap((1, 2), (0, 3), size=(2, 2))
    at = a.transpose((0, 3, 4, 7, 1, 2, 5, 6))
    assert allclose(at, bs.toarray())

    bs = b.swap((1, 2), (0, 3), size=50)
    at = a.transpose((0, 3, 4, 7, 1, 2, 5, 6))
    assert allclose(at, bs.toarray())

    bs = b.swap((1, 2), (0, 3))
    at = a.transpose((0, 3, 4, 7, 1, 2, 5, 6))
    assert allclose(at, bs.toarray())

    bs = b.swap((), (0, 1, 2, 3))
    at = a
    assert allclose(at, bs.toarray())

    bs = b.swap(0, 0)
    at = a.transpose((1, 2, 3, 4, 0, 5, 6, 7))
    assert allclose(at, bs.toarray())

    bs = b.swap([], 0)
    at = a.transpose((0, 1, 2, 3, 4, 5, 6, 7))
    assert allclose(at, bs.toarray())
    assert bs.split == 5

    bs = b.swap(0, [])
    at = a.transpose((1, 2, 3, 0, 4, 5, 6, 7))
    assert allclose(at, bs.toarray())
    assert bs.split == 3

    b = array(a, sc, axis=range(8))
    bs = b.swap([0,1], [])
    at = a.transpose((2, 3, 4, 5, 6, 7, 0, 1))
    assert allclose(at, bs.toarray())
    assert bs.split == 6


def test_transpose(sc):

    n = 4
    perms = list(permutations(range(n), n))

    a = arange(2*3*4*5).reshape((2, 3, 4, 5))

    b = array(a, sc, axis=(0, 1))
    for p in perms:
        assert allclose(b.transpose(p).toarray(), b.toarray().transpose(p))

    assert allclose(b.transpose(), b.toarray().transpose())

def test_t(sc):

    a = arange(2*3*4*5).reshape((2, 3, 4, 5))

    b = array(a, sc, axis=0)
    assert allclose(b.T.toarray(), b.toarray().T)

    b = array(a, sc, axis=(0, 1))
    assert allclose(b.T.toarray(), b.toarray().T)

def test_swapaxes(sc):

    a = arange(2*3*4*5).reshape((2, 3, 4, 5))

    b = array(a, sc, axis=(0, 1))
    assert allclose(b.swapaxes(1, 2).toarray(), b.toarray().swapaxes(1, 2))
    assert allclose(b.swapaxes(0, 1).toarray(), b.toarray().swapaxes(0, 1))
    assert allclose(b.swapaxes(2, 3).toarray(), b.toarray().swapaxes(2, 3))

def test_reshape(sc):

    old_shape = (6, 10, 4, 12)
    a = arange(prod(old_shape)).reshape(old_shape)
    b = array(a, sc, axis=(0, 1))

    # keys only
    new_shape = (15, 4, 4, 12)
    assert allclose(b.reshape(new_shape).toarray(), b.toarray().reshape(new_shape))
    # values only
    new_shape = (6, 10, 24, 2)
    assert allclose(b.reshape(new_shape).toarray(), b.toarray().reshape(new_shape))
    # keys and values, independent
    new_shape = (15, 4, 24, 2)
    assert allclose(b.reshape(new_shape).toarray(), b.toarray().reshape(new_shape))
    # keys and values, mixing
    new_shape = (6, 4, 10, 12)
    with pytest.raises(NotImplementedError):
        b.reshape(new_shape)
