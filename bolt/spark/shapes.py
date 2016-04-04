from numpy import unravel_index, ravel_multi_index

from bolt.utils import argpack, istransposeable, isreshapeable
from bolt.spark.array import BoltArraySpark


class Shapes(object):
    """
    Base Shape class. These classes wrap a BoltArraySpark in their
    entirity, but implement the following attributes and methods as if
    they were only working on the keys or the values, depending which
    subclass is used.
    """
    @property
    def shape(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError

class Keys(Shapes):
    """
    This class implements all the base shape attributes and methods
    for the keys of a BoltArraySpark.
    """
    def __init__(self, barray):
        self._barray = barray

    @property
    def shape(self):
        return self._barray.shape[:self._barray.split]

    def reshape(self, *shape):
        """
        Reshape just the keys of a BoltArraySpark, returning a
        new BoltArraySpark.

        Parameters
        ----------
        shape : tuple
              New proposed axes.
        """
        new = argpack(shape)
        old = self.shape
        isreshapeable(new, old)

        if new == old:
            return self._barray

        def f(k):
            return unravel_index(ravel_multi_index(k, old), new)

        newrdd = self._barray._rdd.map(lambda kv: (f(kv[0]), kv[1]))
        newsplit = len(new)
        newshape = new + self._barray.values.shape

        return BoltArraySpark(newrdd, shape=newshape, split=newsplit).__finalize__(self._barray)

    def transpose(self, *axes):
        """
        Transpose just the keys of a BoltArraySpark, returning a
        new BoltArraySpark.

        Parameters
        ----------
        axes : tuple
             New proposed axes.
        """
        new = argpack(axes)
        old = range(self.ndim)
        istransposeable(new, old)

        if new == old:
            return self._barray

        def f(k):
            return tuple(k[i] for i in new)

        newrdd = self._barray._rdd.map(lambda kv: (f(kv[0]), kv[1]))
        newshape = tuple(self.shape[i] for i in new) + self._barray.values.shape

        return BoltArraySpark(newrdd, shape=newshape, ordered=False).__finalize__(self._barray)

    def __str__(self):
        s = "BoltArray Keys\n"
        s += "shape: %s" % str(self.shape)
        return s

    def __repr__(self):
        return str(self)

class Values(Shapes):
    """
    This class implements all the base shape attributes and methods
    for the values of a BoltArraySpark.
    """
    def __init__(self, barray):
        self._barray = barray

    @property
    def shape(self):
        return self._barray.shape[self._barray.split:]

    def reshape(self, *shape):
        """
        Reshape just the values of a BoltArraySpark, returning a
        new BoltArraySpark.

        Parameters
        ----------
        shape : tuple
              New proposed axes.
        """
        new = argpack(shape)
        old = self.shape
        isreshapeable(new, old)

        if new == old:
            return self._barray

        def f(v):
            return v.reshape(new)

        newrdd = self._barray._rdd.mapValues(f)
        newshape = self._barray.keys.shape + new

        return BoltArraySpark(newrdd, shape=newshape).__finalize__(self._barray)

    def transpose(self, *axes):
        """
        Transpose just the values of a BoltArraySpark, returning a
        new BoltArraySpark.

        Parameters
        ----------
        axes : tuple
             New proposed axes.
        """
        new = argpack(axes)
        old = range(self.ndim)
        istransposeable(new, old)

        if new == old:
            return self._barray

        def f(v):
            return v.transpose(new)

        newrdd = self._barray._rdd.mapValues(f)
        newshape = self._barray.keys.shape + tuple(self.shape[i] for i in new)

        return BoltArraySpark(newrdd, shape=newshape).__finalize__(self._barray)

    def __str__(self):
        s = "BoltArray Values\n"
        s += "shape: %s" % str(self.shape)
        return s

    def __repr__(self):
        return str(self)
