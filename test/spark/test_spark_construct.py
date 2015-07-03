import pytest
from numpy import arange
from bolt import array, ones, zeros, concatenate
from bolt.utils import allclose
from bolt.spark.array import BoltArraySpark

def test_array(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc)
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

    b = array(x, sc, axes=(0,))
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

    b = array(x, sc, axes=(0, 1))
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

def test_array_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    with pytest.raises(ValueError):
        array(x, sc, axes=(-1,))

    with pytest.raises(ValueError):
        array(x, sc, axes=(0, 1, 2, 3))

def test_ones(sc):

    from numpy import ones as npones
    x = npones((2, 3, 4))
    b = ones((2, 3, 4), sc)
    assert allclose(x, b.toarray())

    x = npones(5)
    b = ones(5, sc)
    assert allclose(x, b.toarray())

def test_zeros(sc):

    from numpy import zeros as npzeros
    x = npzeros((2, 3, 4))
    b = zeros((2, 3, 4), sc)
    assert allclose(x, b.toarray())

    x = npzeros(5)
    b = zeros(5, sc)
    assert allclose(x, b.toarray())

def test_concatenate(sc):

    from numpy import concatenate as npconcatenate
    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=0)
    bb = concatenate((b, b), axis=0)
    assert allclose(npconcatenate((x, x), axis=0), bb.toarray())

    bb = concatenate((b, b), axis=1)
    assert allclose(npconcatenate((x, x), axis=1), bb.toarray())

    bb = concatenate((b, b), axis=2)
    assert allclose(npconcatenate((x, x), axis=2), bb.toarray())

    b = array(x, sc, axes=(0, 1))
    bb = concatenate((b, b), axis=0)
    assert allclose(npconcatenate((x, x), axis=0), bb.toarray())

    b = array(x, sc, axes=(0, 1))
    bb = concatenate((b, b), axis=1)
    assert allclose(npconcatenate((x, x), axis=1), bb.toarray())

    b = array(x, sc, axes=(0, 1))
    bb = concatenate((b, b), axis=2)
    assert allclose(npconcatenate((x, x), axis=2), bb.toarray())

def test_concatenate_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=0)

    with pytest.raises(ValueError):
        concatenate(b)

    with pytest.raises(NotImplementedError):
        concatenate((b, b, b))
