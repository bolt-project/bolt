import pytest
from numpy import arange, repeat, asarray, vstack, tile
from bolt import array, ones
from bolt.utils import allclose
from bolt.spark.array import BoltArraySpark


def _2D_stackable_preamble(sc, num_partitions=2):

    dims = (10, 10)
    arr = vstack([[x]*dims[1] for x in arange(dims[0])])
    barr = array(arr, sc, axis=0)
    barr = BoltArraySpark(barr._rdd.partitionBy(num_partitions),
                          shape=barr.shape, split=barr.split)
    return barr

def _3D_stackable_preamble(sc, num_partitions=2):

    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x, area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    barr = array(arr, sc, axis=0)
    barr = BoltArraySpark(barr._rdd.partitionBy(num_partitions),
                          shape=barr.shape, split=barr.split)
    return barr

def test_stack_2D(sc):

    barr = _2D_stackable_preamble(sc)

    # without stack_size
    stacked = barr.stack()
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (5, 10)
    assert stacked.shape == (10, 10)

    # with stack_size
    stacked = barr.stack(size=2)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (2, 10)

    # invalid stack_size
    stacked = barr.stack(size=0)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (5, 10)

    # unstacking
    unstacked = stacked.unstack()
    arr = unstacked.toarray()
    assert arr.shape == (10, 10)
    assert allclose(arr, barr.toarray())

def test_stack_3D(sc):

    barr = _3D_stackable_preamble(sc)

    # with stack_size
    stacked = barr.stack(size=2)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (2, 10, 10)

    # invalid stack_size
    stacked = barr.stack(size=0)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (5, 10, 10)

    # unstacking
    unstacked = stacked.unstack()
    arr = unstacked.toarray()
    assert arr.shape == (10, 10, 10)
    assert allclose(arr, barr.toarray())

def test_stacked_map(sc):

    barr = _2D_stackable_preamble(sc)

    map_func1 = lambda x: x * 2

    funcs = [map_func1]

    for func in funcs:
        stacked = barr.stack()
        stacked_map = stacked.map(func)
        normal_map = barr.map(func)
        unstacked = stacked_map.unstack()
        assert normal_map.shape == unstacked.shape
        assert normal_map.split == unstacked.split
        assert allclose(normal_map.toarray(), unstacked.toarray())

def test_stacked_shape_inference(sc):

    from numpy import ones as npones

    a = ones((100, 2), sc)
    a._rdd = a._rdd.partitionBy(2)
    s = a.stack(5)
    n = s.nrecords

    # operations that preserve keys
    assert s.map(lambda x: x * 2).unstack().shape == (100, 2)
    assert s.map(lambda x: x.sum(axis=1)).unstack().shape == (100,)
    assert s.map(lambda x: tile(x, (1, 2))).unstack().shape == (100, 4)

    # operations that create new keys
    assert s.map(lambda x: npones((2, 2))).unstack().shape == (n, 2, 2)
    assert s.map(lambda x: x.sum(axis=0)).unstack().shape == (n, 2)
    assert s.map(lambda x: asarray([2])).unstack().toarray().shape == (n, 1)
    assert s.map(lambda x: asarray(2)).unstack().toarray().shape == (n,)

    # composing functions works
    assert s.map(lambda x: x * 2).map(lambda x: x * 2).unstack().shape == (100, 2)
    assert s.map(lambda x: x * 2).map(lambda x: npones((2, 2))).unstack().shape == (n, 2, 2)
    assert s.map(lambda x: npones((2, 2))).map(lambda x: x * 2).unstack().shape == (n, 2, 2)

    # check the result
    assert allclose(s.map(lambda x: x.sum(axis=1)).unstack().toarray(), npones(100) * 2)
    assert allclose(s.map(lambda x: tile(x, (1, 2))).unstack().toarray(), npones((100, 4)))

    with pytest.raises(ValueError):
        s.map(lambda x: 2)

    with pytest.raises(ValueError):
        s.map(lambda x: None)

    with pytest.raises(RuntimeError):
        s.map(lambda x: 1/0)

def test_stacked_nrecords(sc):

    a = ones((100, 2), sc)
    a._rdd = a._rdd.partitionBy(2)

    assert a.stack(1).nrecords == 100
    assert a.stack(5).nrecords == 20
    assert a.stack(10).nrecords == 10

def test_stacked_conversion(sc):

    from pyspark import RDD
    barr = _2D_stackable_preamble(sc)
    k1 = barr.tordd().keys()
    assert isinstance(k1, RDD)