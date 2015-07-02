from numpy import arange, ones, zeros, random

import pytest

from bolt import array, ones, zeros, concatenate
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

def test_concatenate():

    from numpy import concatenate as npconcatenate
    x = arange(2*3*4).reshape((2, 3, 4))
    b = concatenate((x, x))
    assert allclose(npconcatenate((x, x)), b.toarray())

def test_concatenate_errors():

    x = arange(2*3*4).reshape((2, 3, 4))

    with pytest.raises(ValueError):
        concatenate(x)
