import pytest
from numpy import arange, split
from bolt import array
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

    allclose(b.chunk((2, 3)).unchunk().toarray(), b.toarray())
    allclose(b.chunk((3, 4)).unchunk().toarray(), b.toarray())

    x = arange(4*5*10).reshape(1, 4, 5, 10)
    b = array(x, sc)

    allclose(b.chunk((4, 5, 10)).unchunk().toarray(), b.toarray())
    allclose(b.chunk((1, 1, 1)).unchunk().toarray(), b.toarray())
    allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())
    allclose(b.chunk((3, 3, 3)).unchunk().toarray(), b.toarray())
