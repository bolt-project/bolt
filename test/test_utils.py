from bolt.utils import tupleize, argpack


def test_tupleize():

    assert tupleize([1, 2, 3]) == (1, 2, 3)
    assert tupleize((1, 2, 3)) == (1, 2, 3)
    assert tupleize((1,)) == (1,)
    assert tupleize(1) == (1,)


def test_argpack():

    assert argpack(((1, 2),)) == (1, 2)
    assert argpack((1, 2)) == (1, 2)
    assert argpack(([0, 1],)) == (0, 1)