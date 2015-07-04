from numpy import arange, repeat
from bolt import array
from bolt.utils import allclose
from bolt.spark.array import BoltArraySpark


def _2D_stackable_preamble(sc, num_partitions=2):
    from numpy import vstack

    dims = (10, 10)
    arr = vstack([[x]*dims[1] for x in arange(dims[0])])
    barr = array(arr, sc, axis=0)
    barr = BoltArraySpark(barr._rdd.partitionBy(num_partitions),
                          shape=barr.shape, split=barr.split)
    return barr

def _3D_stackable_preamble(sc, num_partitions=2):
    from numpy import asarray

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
    stacked = barr.stack(stack_size=2)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (2, 10)

    # invalid stack_size
    stacked = barr.stack(stack_size=0)
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
    stacked = barr.stack(stack_size=2)
    first_partition = stacked._rdd.first()[1]
    assert first_partition.shape == (2, 10, 10)

    # invalid stack_size
    stacked = barr.stack(stack_size=0)
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
