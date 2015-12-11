import pytest
from numpy import arange, split
from bolt import array, ones
from bolt.utils import allclose

def test_chunk(sc):
    
    x = arange(4*6).reshape(1, 4, 6)
    b = array(x, sc)
    
    k1, v1 = zip(*b.chunk((2,3))._rdd.sortByKey().collect())
    k2 = tuple(zip(((0,), (0,), (0,), (0,)), ((0, 0), (0, 1), (1, 0), (1, 1))))
    v2 = [s for m in split(x[0], (2,), axis=0) for s in split(m, (3,), axis=1)]
    assert k1 == k2
    assert all([allclose(m1, m2) for (m1, m2) in zip(v1, v2)])

    k1, v1 = zip(*b.chunk((3,4))._rdd.sortByKey().collect())
    k2 = tuple(zip(((0,), (0,), (0,), (0,)), ((0, 0), (0, 1), (1, 0), (1, 1))))
    v2 = [s for m in split(x[0], (3,), axis=0) for s in split(m, (4,), axis=1)]
    assert k1 == k2
    assert all([allclose(m1, m2) for (m1, m2) in zip(v1, v2)])

def test_unchunk(sc):

    x = arange(4*6).reshape(1, 4, 6)
    b = array(x, sc)

    assert allclose(b.chunk((2, 3)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((3, 4)).unchunk().toarray(), b.toarray())

    x = arange(4*5*10).reshape(1, 4, 5, 10)
    b = array(x, sc)

    assert allclose(b.chunk((4, 5, 10)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((1, 1, 1)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())

    x = arange(4*6).reshape(4, 6)
    b = array(x, sc, (0, 1))

    assert allclose(b.chunk(()).unchunk().toarray(), b.toarray())

def test_keystovalues(sc):

    x = arange(4*7*9*6).reshape(4, 7, 9, 6)
    b = array(x, sc, (0, 1))
    c = b.chunk((4, 2))

    assert allclose(x, c.keystovalues((0,)).unchunk().toarray().transpose(1, 0, 2, 3))
    assert allclose(x, c.keystovalues((1,)).unchunk().toarray())
    assert allclose(x, c.keystovalues((1,), size=(3,)).unchunk().toarray())
    assert allclose(x, c.keystovalues((0, 1)).unchunk().toarray())
    assert allclose(x, c.keystovalues((0, 1), size=(2, 3)).unchunk().toarray())
    assert allclose(x, c.keystovalues(()).unchunk().toarray())

    b = array(x, sc, range(4))
    c = b.chunk(())

    assert allclose(x, c.keystovalues((3,)).unchunk().toarray())
    assert allclose(x, c.keystovalues((0, 1)).unchunk().transpose(2, 3, 0, 1))

def test_valuestokeys(sc):

    x = arange(4*7*9*6).reshape(4, 7, 9, 6)
    b = array(x, sc, (0, 1))
    c = b.chunk((4, 2))

    assert allclose(x, c.valuestokeys((0,)).unchunk().toarray())
    assert allclose(x, c.valuestokeys((1,)).unchunk().toarray().transpose(0, 1, 3, 2))
    assert allclose(x, c.valuestokeys((0, 1)).unchunk().toarray())
    assert allclose(x, c.valuestokeys(()).unchunk().toarray())

def test_map(sc):

    x = arange(4*6).reshape(1, 4, 6)
    b1 = array(x, sc)

    c1 = b1.chunk(size=(2, 3))

    assert allclose(c1.map(lambda v: v * 2).unchunk().toarray(), x * 2)

    assert c1.map(lambda v: v[0:2, 0:2]).shape == (1, 4, 4)
    assert c1.map(lambda v: v[0:2, 0:2]).unchunk().toarray().shape == (1, 4, 4)

    x = arange(4*7).reshape(1, 4, 7)
    b = array(x, sc)

    with pytest.raises(NotImplementedError):
        b.chunk(size=(2, 3)).map(lambda v: v)

def test_map_drop_dim(sc):

    a = ones((2, 20, 10, 3), sc)
    c = a.chunk((10, 5, 3))

    assert c.map(lambda x: x[:, :, 0:2]).unchunk().toarray().shape == (2, 20, 10, 2)
    assert c.map(lambda x: x[:, :, 0]).unchunk().toarray().shape == (2, 20, 10, 1)
    assert c.map(lambda x: x[:, :, [0]]).unchunk().toarray().shape == (2, 20, 10, 1)

    a = ones((2, 20, 10), sc)
    c = a.chunk((10, 5))

    assert c.map(lambda x: x[:, 0:2]).unchunk().toarray().shape == (2, 20, 4)
    assert c.map(lambda x: x[:, 0]).unchunk().toarray().shape == (2, 20, 2)
    assert c.map(lambda x: x[:, [0]]).unchunk().toarray().shape == (2, 20, 2)

def test_properties(sc):

    x = arange(4*6).reshape(1, 4, 6)
    b = array(x, sc)

    assert b.chunk(size=(2, 3)).uniform is True
    assert b.chunk(size=(2, 4)).uniform is False

def test_args(sc):

    x = arange(4*6).reshape(1, 4, 6)
    b = array(x, sc)

    with pytest.raises(ValueError):
        b.chunk(size=(5, 6))
