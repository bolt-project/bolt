from numpy import arange

from bolt import array, ones, zeros
from bolt.utils import allclose

def test_array():
    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x)
    assert allclose(x, b.toarray())

def test_ones():
    from numpy import ones as npones
    x = npones((2, 3, 4))
    b = ones((2, 3, 4))
    assert allclose(x, b.toarray())

def test_zeros():
    from numpy import zeros as npzeros
    x = npzeros((2, 3, 4))
    b = zeros((2, 3, 4))
    assert allclose(x, b.toarray())
