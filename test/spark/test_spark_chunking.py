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
    assert allclose(b.chunk('0.1').unchunk().toarray(), b.toarray())
    assert allclose(b.chunk().unchunk().toarray(), b.toarray())

    x = arange(4*5*10).reshape(1, 4, 5, 10)
    b = array(x, sc)

    assert allclose(b.chunk((4, 5, 10)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((1, 1, 1)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())
    assert allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())

    x = arange(4*6).reshape(4, 6)
    b = array(x, sc, (0, 1))

    assert allclose(b.chunk(()).unchunk().toarray(), b.toarray())

    b = array(x, sc, (0,))

    assert allclose(b.chunk((2)).unchunk().toarray(), b.toarray())

def test_keys_to_values(sc):

    x = arange(4*7*9*6).reshape(4, 7, 9, 6)
    b = array(x, sc, (0, 1))
    c = b.chunk((4, 2))

    assert allclose(x, c.keys_to_values((0,)).unchunk().toarray().transpose(1, 0, 2, 3))
    assert allclose(x, c.keys_to_values((1,)).unchunk().toarray())
    assert allclose(x, c.keys_to_values((1,), size=(3,)).unchunk().toarray())
    assert allclose(x, c.keys_to_values((0, 1)).unchunk().toarray())
    assert allclose(x, c.keys_to_values((0, 1), size=(2, 3)).unchunk().toarray())
    assert allclose(x, c.keys_to_values(()).unchunk().toarray())

    b = array(x, sc, range(4))
    c = b.chunk(())

    assert allclose(x, c.keys_to_values((3,)).unchunk().toarray())
    assert allclose(x, c.keys_to_values((0, 1)).unchunk().toarray().transpose(2, 3, 0, 1))

    b = array(x, sc, (0,))
    c = b.chunk((2, 3, 4))

    assert allclose(x, c.keys_to_values((0,)).unchunk().toarray())

def test_values_to_keys(sc):

    x = arange(4*7*9*6).reshape(4, 7, 9, 6)
    b = array(x, sc, (0, 1))
    c = b.chunk((4, 2))

    assert allclose(x, c.values_to_keys((0,)).unchunk().toarray())
    assert allclose(x, c.values_to_keys((1,)).unchunk().toarray().transpose(0, 1, 3, 2))
    assert allclose(x, c.values_to_keys((0, 1)).unchunk().toarray())
    assert allclose(x, c.values_to_keys(()).unchunk().toarray())

    b = array(x, sc, (0,))
    c = b.chunk((2, 3, 4))

    assert allclose(x, c.values_to_keys((0,)).unchunk().toarray())
    assert allclose(x, c.values_to_keys((0, 1)).unchunk().toarray())


def test_padding(sc):

    x = arange(2*2*5*6).reshape(2, 2, 5, 6)
    b = array(x, sc, (0, 1))

    c = b.chunk((2, 2), padding=1)
    chunks = c.tordd().sortByKey().values().collect()
    assert allclose(chunks[0], array([[0, 1, 2], [6, 7, 8], [12, 13, 14]]))
    assert allclose(chunks[1], array([[1, 2, 3, 4], [7, 8, 9, 10], [13, 14, 15, 16]]))
    assert allclose(chunks[4], array([[7, 8, 9, 10], [13, 14, 15, 16], [19, 20, 21, 22], [25, 26, 27, 28]]))
    assert allclose(chunks[6], array([[18, 19, 20], [24, 25, 26]]))

    c = b.chunk((3, 3), padding=(1, 2))
    chunks = c.tordd().sortByKey().values().collect()
    assert allclose(chunks[0], array([[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16], [18, 19, 20, 21, 22]]))

    c = b.chunk((2,2), padding=1)
    assert allclose(x, c.unchunk().toarray())
    assert allclose(x, c.keys_to_values((1,)).unchunk().toarray())
    assert allclose(x, c.values_to_keys((0,)).unchunk().toarray())

def test_padding_errors(sc):

        x = arange(2*2*5*6).reshape(2, 2, 5, 6)
        b = array(x, sc, (0, 1))

        with pytest.raises(ValueError):
            c = b.chunk((2, 2), padding=(3, 1))

        with pytest.raises(ValueError):
            c = b.chunk((4, 4), padding=(2, 2))

        with pytest.raises(NotImplementedError):
            c = b.chunk((2, 2), padding=1)
            d = c.map(lambda x: x[:, 0])

def test_map(sc):

    x = arange(4*8*8).reshape(4, 8, 8)
    b = array(x, sc)

    c = b.chunk(size=(4, 8))

    # no change of shape
    def f(x):
        return 2*x

    assert allclose(c.map(f).unchunk().toarray(), f(x))
    assert allclose(c.map(f, value_shape=(4, 8)).unchunk().toarray(), f(x))

    # changing the size of an unchunked axis
    def f(x):
        return x[:, :4]
    def f_local(x):
        return x[:, :, :4]

    assert allclose(c.map(f).unchunk().toarray(), f_local(x))
    assert allclose(c.map(f, value_shape=(4, 4)).unchunk().toarray(), f_local(x))

def test_map_errors(sc):

    x = arange(4*8*8).reshape(4, 8, 8)
    b = array(x, sc)

    c = b.chunk(size=(4, 8))

    # changing the size of a chunked axis
    def f(x):
        return x[:2, :]

    with pytest.raises(ValueError):
        c.map(f)

    with pytest.raises(ValueError):
        c.map(f, value_shape=(2, 8))

    # dropping dimensions
    def f(x):
        return x[0, :]

    with pytest.raises(NotImplementedError):
        c.map(f)

    with pytest.raises(NotImplementedError):
        c.map(f, value_shape=(4,))

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
