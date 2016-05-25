from numpy import arange, dtype, int64, float64
from bolt import array, ones
from bolt.utils import allclose

def test_shape(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    assert b.shape == x.shape

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc)
    assert b.shape == x.shape

def test_size(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axis=0)
    assert b.size == x.size

def test_split(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axis=0)
    assert b.split == 1

    b = array(x, sc, axis=(0, 1))
    assert b.split == 2

def test_ndim(sc):

    x = arange(2**5).reshape(2, 2, 2, 2, 2)
    b = array(x, sc, axis=(0, 1, 2))

    assert b.keys.ndim == 3
    assert b.values.ndim == 2
    assert b.ndim == 5

def test_mask(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x, sc, axis=0)
    assert b.mask == (1, 0, 0)

    b = array(x, sc, axis=(0, 1))
    assert b.mask == (1, 1, 0)

    b = array(x, sc, axis=(0, 1, 2))
    assert b.mask == (1, 1, 1)

def test_cache(sc):

    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    b.cache()
    assert b._rdd.is_cached
    b.unpersist()
    assert not b._rdd.is_cached

def test_repartition(sc):
    x = arange(2 * 3).reshape((2, 3))
    b = array(x, sc)
    assert b._ordered
    b = b.repartition(10)
    assert not b._ordered
    assert b._rdd.getNumPartitions() == 10

def test_concatenate(sc):

    from numpy import concatenate
    x = arange(2*3).reshape((2, 3))
    b = array(x, sc)
    c = array(x)
    assert allclose(b.concatenate(x).toarray(), concatenate((x, x)))
    assert allclose(b.concatenate(b).toarray(), concatenate((x, x)))
    assert allclose(b.concatenate(c).toarray(), concatenate((x, x)))

def test_dtype(sc):

    a = arange(2**8, dtype=int64)
    b = array(a, sc, dtype=int64)
    assert a.dtype == b.dtype
    assert b.dtype == dtype(int64)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(int64)
    
    a = arange(2.0**8)
    b = array(a, sc)
    assert a.dtype == b.dtype
    assert b.dtype == dtype(float64)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(float64)

    a = arange(2**8)
    b = array(a, sc)
    assert a.dtype == b.dtype
    assert b.dtype == dtype(int64)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(int64)

    from numpy import ones as npones
    a = npones(2**8, dtype=bool)
    b = array(a, sc)
    assert a.dtype == b.dtype
    assert b.dtype == dtype(bool)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(bool)

    b = ones(2**8, sc)
    assert b.dtype == dtype(float64)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(float64)

    b = ones(2**8, sc, dtype=bool)
    assert b.dtype == dtype(bool)
    dtypes = b._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(bool)

def test_astype(sc):
    
    from numpy import ones as npones
    
    a = npones(2**8, dtype=int64)
    b = array(a, sc, dtype=int64)
    c = b.astype(bool)
    assert c.dtype == dtype(bool)
    dtypes = c._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(bool)
        
    b = ones((100, 100), sc, dtype=int64)
    c = b.astype(bool)
    assert c.dtype == dtype(bool)
    dtypes = c._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(bool)

    b = ones((100, 100), sc)
    c = b.astype(bool)
    assert c.dtype == dtype(bool)
    dtypes = c._rdd.map(lambda x: x[1].dtype).collect()
    for dt in dtypes:
        assert dt == dtype(bool)

def test_clip(sc):

    from numpy import arange

    a = arange(4).reshape(2, 2)
    b = array(a, sc)
    assert allclose(b.clip(0).toarray(), a.clip(0))
    assert allclose(b.clip(2).toarray(), a.clip(2))
    assert allclose(b.clip(1, 2).toarray(), a.clip(1, 2))