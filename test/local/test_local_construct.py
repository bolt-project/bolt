from numpy import arange, ones, zeros, random
from bolt.common import allclose

import bolt as blt

def test_array():
    x = arange(2*3*4).reshape((2, 3, 4))
    b = blt.array(x)
    assert allclose(x, b.toarray())

def test_ones():
    x = ones((2, 3, 4))
    b = blt.ones((2, 3, 4))
    assert allclose(x, b.toarray())

def test_zeros():
    x = zeros((2, 3, 4))
    b = blt.zeros((2, 3, 4))
    assert allclose(x, b.toarray())
