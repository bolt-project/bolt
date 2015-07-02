from numpy import arange, squeeze, vstack, repeat, asarray, ones

import pytest

from bolt import array
from bolt.utils import allclose
from bolt.spark.array import BoltArraySpark

import generic

def test_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.shape == x.shape

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc)
    assert b.shape == x.shape


def test_size(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.size == x.size


def test_split(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.split == 1

    b = array(x, sc, axes=(0, 1))
    assert b.split == 2


def test_mask(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.mask == (1, 0, 0)

    b = array(x, sc, axes=(0, 1))
    assert b.mask == (1, 1, 0)

    b = array(x, sc, axes=(0, 1, 2))
    assert b.mask == (1, 1, 1)


def test_value_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.values.shape == (3,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0,))
    assert b.values.shape == (3, 4)


def test_key_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.keys.shape == (2,)

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axes=(0, 1))
    assert b.keys.shape == (2, 3)

def test_ndim(sc):

    x = arange(2**5).reshape(2, 2, 2, 2, 2)
    b = array(x, sc, axes=[0, 1, 2])

    assert b.keys.ndim == 3
    assert b.values.ndim == 2
    assert b.ndim == 5

def test_reshape_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    c = b.keys.reshape((3, 2))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.reshape((3, 2, 4)))

    b = array(x, sc, axes=(0,))
    c = b.keys.reshape((2, 1))
    assert allclose(c.toarray(), x.reshape((2, 1, 3, 4)))

    b = array(x, sc, axes=(0,))
    c = b.keys.reshape((2,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.keys.reshape((2, 3))
    assert allclose(c.toarray(), x)

def test_reshape_keys_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.keys.reshape((2, 3, 4))

def test_reshape_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0,))
    c = b.values.reshape((4, 3))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.reshape((2, 4, 3)))

    b = array(x, sc, axes=(0, 1))
    c = b.values.reshape((1, 4))
    assert allclose(c.toarray(), x.reshape((2, 3, 1, 4)))

    b = array(x, sc, axes=(0, 1))
    c = b.values.reshape((4,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0,))
    c = b.values.reshape((3, 4))
    assert allclose(c.toarray(), x)

def test_reshape_values_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.values.reshape((2, 3, 4))

def test_transpose_keys(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    c = b.keys.transpose((1, 0))
    assert c.keys.shape == (3, 2)
    assert allclose(c.toarray(), x.transpose((1, 0, 2)))

    b = array(x, sc, axes=(0,))
    c = b.keys.transpose((0,))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.keys.transpose((0, 1))
    assert allclose(c.toarray(), x)

def test_transpose_keys_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0, 1))
    with pytest.raises(ValueError):
        b.keys.transpose((0, 2))

    with pytest.raises(ValueError):
        b.keys.transpose((1, 1))

    with pytest.raises(ValueError):
        b.keys.transpose((0,))

def test_transpose_values(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0,))
    c = b.values.transpose((1, 0))
    assert c.values.shape == (4, 3)
    assert allclose(c.toarray(), x.transpose((0, 2, 1)))

    b = array(x, sc, axes=(0,))
    c = b.values.transpose((0, 1))
    assert allclose(c.toarray(), x)

    b = array(x, sc, axes=(0, 1))
    c = b.values.transpose((0,))
    assert allclose(c.toarray(), x.reshape((2, 3, 4)))

def test_traspose_values_errors(sc):

    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x, sc, axes=(0,))
    with pytest.raises(ValueError):
        b.values.transpose((0, 2))

    with pytest.raises(ValueError):
        b.values.transpose((1, 1))

    with pytest.raises(ValueError):
        b.values.transpose((0,))

"""
Stackable interface tests
"""

def _2D_stackable_preamble(sc, num_partitions=2):
    dims = (10, 10)
    arr = vstack([[x]*dims[1] for x in arange(dims[0])])
    barr = array(arr, sc, axes=(0,))
    barr = BoltArraySpark(barr._rdd.partitionBy(num_partitions),
            shape=barr.shape, split=barr.split)
    return barr

def _3D_stackable_preamble(sc, num_partitions=2):
    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x,area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    barr = array(arr, sc, axes=(0,))
    barr = BoltArraySpark(barr._rdd.partitionBy(num_partitions),
            shape=barr.shape, split=barr.split)
    return barr

def test_stack_2D(sc):

    barr = _2D_stackable_preamble(sc)

    # Without stack_size
    stacked = barr.stacked()
    first_partition = stacked._barray._rdd.first()[1]
    assert first_partition.shape == (5, 10)
    assert stacked._barray.shape == (10, 10)

    # With stack_size
    stacked = barr.stacked(stack_size=2)
    first_partition = stacked._barray._rdd.first()[1]
    assert first_partition.shape == (2, 10)

    # Invalid stack_size
    stacked = barr.stacked(stack_size=0)
    first_partition = stacked._barray._rdd.first()[1]
    assert first_partition.shape == (5, 10)

    # Unstacking
    unstacked = stacked.unstack()
    arr = unstacked.toarray()
    assert arr.shape == (10, 10)
    assert allclose(arr, barr.toarray())

def test_stack_3D(sc):

    barr = _3D_stackable_preamble(sc)

    # With stack_size
    stacked = barr.stacked(stack_size=2)
    first_partition = stacked._barray._rdd.first()[1]
    assert first_partition.shape == (2, 10, 10)

    # Invalid stack_size
    stacked = barr.stacked(stack_size=0)
    first_partition = stacked._barray._rdd.first()[1]
    assert first_partition.shape == (5, 10, 10)

    # Unstacking
    unstacked = stacked.unstack()
    arr = unstacked.toarray()
    assert arr.shape == (10, 10, 10)
    assert allclose(arr, barr.toarray())

def test_stacked_map(sc):

    barr = _2D_stackable_preamble(sc)

    map_func1 = lambda x: x * 2
    map_func2 = lambda x: ones(10)

    funcs = [map_func1, map_func2]

    for func in funcs:
        stacked = barr.stacked()
        stacked_map = stacked.map(func)
        normal_map = barr.map(func)
        unstacked = stacked_map.unstack()
        assert normal_map.shape == unstacked.shape
        assert normal_map.split == unstacked.split
        assert allclose(normal_map.toarray(), unstacked.toarray())

def test_stacked_reduce(sc):

    from numpy import max

    barr = _2D_stackable_preamble(sc)

    reduce_func1 = lambda x,y: max(x, y)

    funcs = [reduce_func1]

    for func in funcs:
        stacked = barr.stacked()
        stacked_reduce = stacked.reduce(func)
        normal_reduce = barr.reduce(func)
        unstacked = stacked_map.unstack()
        assert normal_map.shape == unstacked.shape
        assert normal_map.split == unstacked.split
        assert allclose(normal_map.toarray(), unstacked.toarray())

"""
Testing functional operators
"""

def test_map(sc):

    import random
    random.seed(42)

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axes=(0,))

    # Test all map functionality when the base array is split after the first axis
    generic.map_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axes=(0, 1))
    generic.map_suite(x, b)

    # Simple map should produce the same result even across multiple axes, though with a different
    # shape
    mapped = b.map(lambda x: x * 2, axes=(0,1))
    swapped = mapped.swap([1], [])
    swapped = mapped.toarray()
    assert allclose(swapped, x * 2)

def test_reduce(sc):

    from numpy import asarray

    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x,area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    b = array(arr, sc, axes=(0,))

    # Test all reduce functionality when the base array is split after the first axis
    generic.reduce_suite(arr, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(arr, sc, axes=(0,1))
    generic.reduce_suite(arr, b)

def test_filter(sc):

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axes=(0,))

    # Test all filter functionality when the base array is split after the first axis
    generic.filter_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axes=(0, 1))
    generic.filter_suite(x, b)

def test_getitem_slice(sc):

    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axes=(0,))
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

    b = array(x, sc, axes=(0, 1))
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

def test_getitem_slice_ragged(sc):

    x = arange(10*10*3).reshape((10, 10, 3))

    b = array(x, sc, axes=(0,1))
    assert allclose(b[0:5:2, 0:2].toarray(), x[0:5:2, 0:2])
    assert allclose(b[0:5:3, 0:2].toarray(), x[0:5:3, 0:2])
    assert allclose(b[0:9:3, 0:2].toarray(), x[0:9:3, 0:2])

def test_getitem_int(sc):

    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axes=(0,))
    assert allclose(b[0, 0].toarray(), x[0, 0])
    assert allclose(b[0, 1].toarray(), x[0, 1])
    assert allclose(b[0, 0:1].toarray(), x[0, 0:1])
    assert allclose(b[1, 2].toarray(), x[1, 2])
    assert allclose(b[[1], [2]].toarray(), x[[1], [2]])

    b = array(x, sc, axes=(0, 1))
    assert allclose(b[0, 0].toarray(), x[0, 0])
    assert allclose(b[0, 1].toarray(), x[0, 1])
    assert allclose(b[0, 0:1].toarray(), x[0, 0:1])
    assert allclose(b[1, 2].toarray(), x[1, 2])
    assert allclose(b[[1], [2]].toarray(), x[[1], [2]])

def test_getitem_list(sc):

    x = arange(3*3*4).reshape((3, 3, 4))

    b = array(x, sc, axes=(0,))
    assert allclose(b[[0, 1], [0, 1], [0, 2]].toarray(), x[[0, 1], [0, 1], [0, 2]])
    assert allclose(b[[0, 1], [0, 2], [0, 3]].toarray(), x[[0, 1], [0, 2], [0, 3]])
    assert allclose(b[[0, 1, 2], [0, 2, 1], [0, 3, 1]].toarray(), x[[0, 1, 2], [0, 2, 1], [0, 3, 1]])

    b = array(x, sc, axes=(0,1))
    assert allclose(b[[0, 1], [0, 1], [0, 2]].toarray(), x[[0, 1], [0, 1], [0, 2]])
    assert allclose(b[[0, 1], [0, 2], [0, 3]].toarray(), x[[0, 1], [0, 2], [0, 3]])
    assert allclose(b[[0, 1, 2], [0, 2, 1], [0, 3, 1]].toarray(), x[[0, 1, 2], [0, 2, 1], [0, 3, 1]])

def test_getitem_list_array(sc):

    x = arange(3*3*4).reshape((3, 3, 4))

    rows = [[0, 0], [1, 1]]
    cols = [[0, 2], [0, 2]]
    dept = [[0, 3], [0, 3]]

    b = array(x, sc, axes=(0,))
    assert allclose(b[rows, cols, dept].toarray(), x[rows, cols, dept])

    b = array(x, sc, axes=(0,1))
    assert allclose(b[rows, cols, dept].toarray(), x[rows, cols, dept])

def test_swap(sc):
   
   a = arange(2**8).reshape(*(8*[2]))
   b = array(a, sc, axes=(0, 1, 2, 3))

   bT = b.swap([1,2],[0,3], size=(2,2)).toarray()
   aT = a.transpose([0,3,4,7,1,2,5,6])

   assert allclose(aT, bT)
