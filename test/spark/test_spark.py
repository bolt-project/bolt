import pytest
from numpy import arange, allclose

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


def test_mask(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.mask == (1, 0, 0)

    b = array(x, sc, axes=(0, 1))
    assert b.mask == (1, 1, 0)

    b = array(x, sc, axes=(0, 1, 2))
    assert b.mask == (1, 1, 1)


def test_value_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.values.shape == (3,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.values.shape == (3, 4)


def test_key_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.keys.shape == (2,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0, 1))
    assert b.keys.shape == (2, 3)


def test_reshape_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    c = b.keys.reshape((3, 2))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.reshape((3, 2, 4)))

    b = array(x, sc, axes=(0,))
    c = b.keys.reshape((2, 1))
    assert allclose(c.toarray(), x.reshape((2, 1, 3, 4)))

    b = array(x, sc, axes=(0,))
    c = b.keys.reshape((2,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.keys.reshape((2, 3))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.keys.reshape((2, 3, 4))


def test_reshape_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    c = b.values.reshape((4, 3))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.reshape((2, 4, 3)))

    b = array(x, sc, axes=(0, 1))
    c = b.values.reshape((1, 4))
    assert allclose(c.toarray(), x.reshape((2, 3, 1, 4)))

    b = array(x, sc, axes=(0, 1))
    c = b.values.reshape((4,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0,))
    c = b.values.reshape((3, 4))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.values.reshape((2, 3, 4))


def test_transpose_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    c = b.keys.transpose((1, 0))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.transpose((1, 0, 2)))

    b = array(x, sc, axes=(0,))
    c = b.keys.transpose((0,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.keys.transpose((0, 1))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.keys.transpose((0, 2))

    with pytest.raises(ValueError):
        b.keys.transpose((1, 1))

    with pytest.raises(ValueError):
        b.keys.transpose((0,))


def test_transpose_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0,))
    c = b.values.transpose((1, 0))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.transpose((0, 2, 1)))

    b = array(x, sc, axes=(0,))
    c = b.values.transpose((0, 1))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.values.transpose((0,))
    assert allclose(c.toarray(), x.reshape((2, 3, 4)))

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.values.transpose((0, 2))

    with pytest.raises(ValueError):
        b.values.transpose((1, 1))

    with pytest.raises(ValueError):
        b.values.transpose((0,))


def test_getitem_slice(sc):

    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axes=(0,))
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

    b = array(x, sc, axes=(0, 1))
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

def test_getitem_int(sc):

    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axes=(0,))
    assert allclose(b[0, 0].toarray(), x[0, 0])
    assert allclose(b[0, 1].toarray(), x[0, 1])
    assert allclose(b[0, 0:1].toarray(), x[0, 0:1])
    assert allclose(b[1, 2].toarray(), x[1, 2])

    b = array(x, sc, axes=(0, 1))
    assert allclose(b[0, 0].toarray(), x[0, 0])
    assert allclose(b[0, 1].toarray(), x[0, 1])
    assert allclose(b[0, 0:1].toarray(), x[0, 0:1])
    assert allclose(b[1, 2].toarray(), x[1, 2])
