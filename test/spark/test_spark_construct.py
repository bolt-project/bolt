from numpy import arange

import pytest

from bolt import array, ones, zeros
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

    with pytest.raises(ValueError):
        array(x, sc, axes=(-1,))

    with pytest.raises(ValueError):
        array(x, sc, axes=(0, 1, 2, 3))


def test_ones(sc):
    from numpy import ones as npones
    x = npones((2, 3, 4))
    b = ones((2, 3, 4), sc)
    assert allclose(x, b.toarray())

def test_zeros(sc):
    from numpy import zeros as npzeros
    x = npzeros((2, 3, 4))
    b = zeros((2, 3, 4), sc)
    assert allclose(x, b.toarray())
