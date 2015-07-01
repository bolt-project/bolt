import pytest
from numpy import arange, ones, zeros
from bolt.common import allclose

import bolt as blt
from bolt.spark.spark import BoltArraySpark

def test_array(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = blt.array(x, sc)
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

    b = blt.array(x, sc, split=1)
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

    b = blt.array(x, sc, split=2)
    assert isinstance(b, BoltArraySpark)
    assert allclose(x, b.toarray())

    with pytest.raises(ValueError):
        blt.array(x, sc, split=0)

    with pytest.raises(ValueError):
        blt.array(x, sc, split=4)


def test_ones(sc):
    x = ones((2, 3, 4))
    b = blt.ones((2, 3, 4), sc)
    assert allclose(x, b.toarray())

def test_zeros(sc):
    x = zeros((2, 3, 4))
    b = blt.zeros((2, 3, 4), sc)
    assert allclose(x, b.toarray())
