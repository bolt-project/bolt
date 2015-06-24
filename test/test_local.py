from numpy import arange, allclose
from bolt import barray
from bolt.spark import BoltArraySpark


def test_construct():
    x = arange(2*3*4).reshape((2, 3, 4))
    b = barray(x)
    assert b.shape == (2, 3, 4)


def test_toarray():

    x = arange(2*3*4).reshape((2, 3, 4))
    b = barray(x)
    assert allclose(b.toarray(), x)


def test_tospark(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = barray(x)
    s = b.tospark(sc, split=1)
    assert isinstance(s, BoltArraySpark)
    assert s.shape == (2, 3, 4)
    assert allclose(s.toarray(), x)


def test_tordd(sc):

    from pyspark import RDD
    x = arange(2*3*4).reshape((2, 3, 4))

    b = barray(x)
    r = b.tordd(sc, split=1)
    assert isinstance(r, RDD)
    assert r.count() == 2

    r = b.tordd(sc, split=2)
    assert isinstance(r, RDD)
    assert r.count() == 2*3

    r = b.tordd(sc, split=3)
    assert isinstance(r, RDD)
    assert r.count() == 2*3*4
