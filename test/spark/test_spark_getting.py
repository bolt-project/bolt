from numpy import arange
from bolt import array, ones
from bolt.utils import allclose


def test_getitem_slice(sc):
    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axis=0)
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

    b = array(x, sc, axis=(0, 1))
    assert allclose(b[0:1, 0:1].toarray(), x[0:1, 0:1])
    assert allclose(b[0:2, 0:2].toarray(), x[0:2, 0:2])
    assert allclose(b[0:2, 0:3].toarray(), x[0:2, 0:3])
    assert allclose(b[0:2, 0:3:2].toarray(), x[0:2, 0:3:2])
    assert allclose(b[:2, :2].toarray(), x[:2, :2])
    assert allclose(b[1:, 1:].toarray(), x[1:, 1:])

def test_getitem_slice_ragged(sc):

    x = arange(10*10*3).reshape((10, 10, 3))

    b = array(x, sc, axis=(0,1))
    assert allclose(b[0:5:2, 0:2].toarray(), x[0:5:2, 0:2])
    assert allclose(b[0:5:3, 0:2].toarray(), x[0:5:3, 0:2])
    assert allclose(b[0:9:3, 0:2].toarray(), x[0:9:3, 0:2])

def test_getitem_int(sc):

    x = arange(2*3).reshape((2, 3))

    b = array(x, sc, axis=0)
    assert allclose(b[0, 0], x[0, 0])
    assert allclose(b[0, 1], x[0, 1])
    assert allclose(b[0, 0:1], x[0, 0:1])
    assert allclose(b[1, 2], x[1, 2])
    assert allclose(b[[1], [2]].toarray(), x[[1], [2]])
    assert allclose(b[[1], 2].toarray(), x[[1], 2])

    b = array(x, sc, axis=(0, 1))
    assert allclose(b[0, 0], x[0, 0])
    assert allclose(b[0, 1], x[0, 1])
    assert allclose(b[0, 0:1], x[0, 0:1])
    assert allclose(b[1, 2], x[1, 2])
    assert allclose(b[[1], [2]].toarray(), x[[1], [2]])
    assert allclose(b[[1], 2].toarray(), x[[1], 2])

def test_getitem_list(sc):

    x = arange(3*3*4).reshape((3, 3, 4))

    b = array(x, sc, axis=0)
    assert allclose(b[[0, 1], [0, 1], [0, 2]].toarray(), x[[0, 1], [0, 1], [0, 2]])
    assert allclose(b[[0, 1], [0, 2], [0, 3]].toarray(), x[[0, 1], [0, 2], [0, 3]])
    assert allclose(b[[0, 1, 2], [0, 2, 1], [0, 3, 1]].toarray(), x[[0, 1, 2], [0, 2, 1], [0, 3, 1]])

    b = array(x, sc, axis=(0,1))
    assert allclose(b[[0, 1], [0, 1], [0, 2]].toarray(), x[[0, 1], [0, 1], [0, 2]])
    assert allclose(b[[0, 1], [0, 2], [0, 3]].toarray(), x[[0, 1], [0, 2], [0, 3]])
    assert allclose(b[[0, 1, 2], [0, 2, 1], [0, 3, 1]].toarray(), x[[0, 1, 2], [0, 2, 1], [0, 3, 1]])

def test_getitem_list_array(sc):

    x = arange(3*3*4).reshape((3, 3, 4))

    rows = [[0, 0], [1, 1]]
    cols = [[0, 2], [0, 2]]
    dept = [[0, 3], [0, 3]]

    b = array(x, sc, axis=0)
    assert allclose(b[rows, cols, dept].toarray(), x[rows, cols, dept])

    b = array(x, sc, axis=(0, 1))
    assert allclose(b[rows, cols, dept].toarray(), x[rows, cols, dept])

def test_squeeze(sc):

    from numpy import ones as npones

    x = npones((1, 2, 1, 4))
    b = ones((1, 2, 1, 4), sc, axis=0)
    assert allclose(b.squeeze().toarray(), x.squeeze())
    assert allclose(b.squeeze((0, 2)).toarray(), x.squeeze((0, 2)))
    assert allclose(b.squeeze(0).toarray(), x.squeeze(0))
    assert allclose(b.squeeze(2).toarray(), x.squeeze(2))
    assert b.squeeze().split == 0
    assert b.squeeze((0, 2)).split == 0
    assert b.squeeze(2).split == 1

    x = npones((1, 2, 1, 4))
    b = ones((1, 2, 1, 4), sc, axis=(0, 1))
    assert allclose(b.squeeze().toarray(), x.squeeze())
    assert allclose(b.squeeze((0, 2)).toarray(), x.squeeze((0, 2)))
    assert allclose(b.squeeze(0).toarray(), x.squeeze(0))
    assert allclose(b.squeeze(2).toarray(), x.squeeze(2))
    assert b.squeeze().split == 1
    assert b.squeeze((0, 2)).split == 1
    assert b.squeeze(2).split == 2

    x = npones((1, 1, 1, 1))
    b = ones((1, 1, 1, 1), sc, axis=(0, 1))
    assert allclose(b.squeeze().toarray(), x.squeeze())
