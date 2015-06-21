from bolt.base import BoltArray
from bolt.constructor import barray
from numpy import sum

def test(sc):
    foo = barray([1, 2, 3])
    print sum(foo)

    foo = barray([1, 2, 3], sc)
    print sum(foo)