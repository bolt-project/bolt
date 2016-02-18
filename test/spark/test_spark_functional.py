import pytest
from numpy import arange, repeat
from bolt import array
from bolt.utils import allclose
import generic

def test_map(sc):
    import random
    random.seed(42)

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=0)

    # Test all map functionality when the base array is split after the first axis
    generic.map_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axis=(0, 1))
    generic.map_suite(x, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(x, sc, axis=(0, 1, 2))
    generic.map_suite(x, b)

def test_map_with_keys(sc):
    x = arange(2*3).reshape(2, 3)
    b = array(x, sc, axis=0)
    c = b.map(lambda (k, v): k + v, with_keys=True)
    assert allclose(b.toarray() + [[0, 0, 0], [1, 1, 1]], c.toarray())

def test_reduce(sc):
    from numpy import asarray

    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x, area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    b = array(arr, sc, axis=0)

    # Test all reduce functionality when the base array is split after the first axis
    generic.reduce_suite(arr, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(arr, sc, axis=(0, 1))
    generic.reduce_suite(arr, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(arr, sc, axis=(0, 1, 2))
    generic.reduce_suite(arr, b)

def test_filter(sc):

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=0)

    # Test all filter functionality when the base array is split after the first axis
    generic.filter_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axis=(0, 1))
    generic.filter_suite(x, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(x, sc, axis=(0, 1, 2))
    generic.filter_suite(x, b)

def test_mean(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.mean(), x.mean())
    assert allclose(b.mean(axis=0), x.mean(axis=0))
    assert allclose(b.mean(axis=(0, 1)), x.mean(axis=(0, 1)))
    assert b.mean(axis=(0, 1, 2)) == x.mean(axis=(0, 1, 2))

def test_std(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.std(), x.std())
    assert allclose(b.std(axis=0), x.std(axis=0))
    assert allclose(b.std(axis=(0, 1)), x.std(axis=(0, 1)))
    assert b.std(axis=(0, 1, 2)) == x.std(axis=(0, 1, 2))

def test_var(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.var(), x.var())
    assert allclose(b.var(axis=0), x.var(axis=0))
    assert allclose(b.var(axis=(0, 1)), x.var(axis=(0, 1)))
    assert b.var(axis=(0, 1, 2)) == x.var(axis=(0, 1, 2))

def test_sum(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.sum(), x.sum())
    assert allclose(b.sum(axis=0), x.sum(axis=0))
    assert allclose(b.sum(axis=(0, 1)), x.sum(axis=(0, 1)))
    assert b.sum(axis=(0, 1, 2)) == x.sum(axis=(0, 1, 2))

def test_min(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.min(), x.min())
    assert allclose(b.min(axis=0), x.min(axis=0))
    assert allclose(b.min(axis=(0, 1)), x.min(axis=(0, 1)))
    assert b.min(axis=(0, 1, 2)) == x.min(axis=(0, 1, 2))

def test_max(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.max(), x.max())
    assert allclose(b.max(axis=0), x.max(axis=0))
    assert allclose(b.max(axis=(0, 1)), x.max(axis=(0, 1)))
    assert b.max(axis=(0, 1, 2)) == x.max(axis=(0, 1, 2))
