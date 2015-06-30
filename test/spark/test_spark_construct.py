from numpy import arange, ones, zeros, random, allclose

import bolt as blt

def test_array(sc):
    x = arange(2*3*4).reshape((2, 3, 4))
    b = blt.array(x, sc)
    assert allclose(x, b.toarray())

def test_ones(sc):
    x = ones((2, 3, 4))
    b = blt.ones((2, 3, 4), sc)
    assert allclose(x, b.toarray())

def test_zeros(sc):
    x = zeros((2, 3, 4))
    b = blt.zeros((2, 3, 4), sc)
    assert allclose(x, b.toarray())