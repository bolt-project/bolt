"""
Generic tests for all BoltArrays
"""

def map_suite(arr, b):
    """
    A set of tests for the map operator

    Parameters
    ----------
    arr: `ndarray`
        A 2D array used in the construction of `b` (used to check results)
    b: `BoltArray`
        The BoltArray to be used for testing

    """

    from numpy import ones

    # A simple map should be equivalent to an element-wise multiplication
    func1 = lambda x: x * 2
    mapped = b.map(func1, axes=(0,))
    res = mapped.toarray()
    assert allclose(res, arr * 2)

    # Simple map should produce the same result even across multiple axes
    mapped = b.map(func1, axes=(0,1))
    res = mapped.toarray()
    assert allclose(res, arr * 2)

    # More complicated maps can reshape elements so long as they do so consistently
    func2 = lambda x: ones(10)
    mapped = b.map(func2, axes=(0,))
    res = mapped.toarray()
    assert res.shape == (arr.shape[0], 10)

    # But the shape of the result will change if mapped over different axes
    mapped = b.map(func2, axes=(0,1))
    res = mapped.toarray()
    assert res.shape == (arr.shape[0], arr.shape[1], 10)

    # If a map is not applied uniformly, it should lazily produce an error
    func3 = lambda x: ones(10) if random.random() < 0.5 else ones(5)
    mapped = b.map(func3)
    with pytest.raises(Exception):
        res = mapped.toarray()

def reduce_suite(arr, b):
    """
    A set of tests for the reduce operator

    Parameters
    ----------
    arr: `ndarray`
        A 3D ndarray used in the construction of `b` (used to check results) 
    b: `BoltArray`
        The BoltArray to be used for testing
    """

    from numpy import sum, ones

    # Reduce over the first axis with an add
    reduced = b.reduce(sum, axes=(0,))
    res = reduced.toarray()
    assert res.shape == (arr.shape[1], arr.shape[2])
    assert allclose(res, sum(arr, axis=0))

    # Reduce over multiple axes with an add
    reduced = b.reduce(sum, axes=(0, 1))
    res = reduced.toarray()
    assert res.shape == (arr.shape[2],)
    assert allclose(res, sum(sum(arr, axis=0), axis=1))

    # A reduce operation that yields a result with an invalid shape should lazily error out
    reduced = b.reduce(lambda x,y: ones(5))
    with pytest.raises(Exception):
        res = reduced.toarray()


