
class Stackable(object):
    """
    Interface for objects that can be converted into a stacked representation
    """

    def stack(self, stack_size=None):
        """
        The primary entry point for the StackedBoltArray interface.

        Parameters
        ----------
        stack_size: int, optional, default=None
            The maximum size for each stack (number of original records per stack)

        Returns
        -------
        a StackedBoltArray

        """
        return StackedBoltArray(self._stack(stack_size), stack_size)

    def _stack(self, stack_size=None):
        """
        Converts a non-stacked stackable into a stacked Stackable

        Returns
        -------
        a Stackable whose underlying representation has been converted into stacks
        """
        raise NotImplementedError

    def _unstack(self):
        """
        Splits apart the underlying representation of a stacked Stackable such that the new number
        of records is equal to that of the original.

        Returns
        -------
        a Stackable whose underlying stacked representation has been unstacked
        """
        raise NotImplementedError


class StackedBoltArray(object):
    """
    Wraps a BoltArray and provides an interface for performing Stacked operations (operations
    on whole subarrays). Many BoltArray methods will be restricted or forbidden until the
    StackedBoltArray is unstacked.
    """

    def __init__(self, barray, stack_size=None):
        self._barray = barray
        self.stack_size = stack_size

    def __finalize__(self, other):
        other.stack_size = self.stack_size
        return self

    @property
    def shape(self):
        return self._barray.shape

    @property
    def _constructor(self):
        return StackedBoltArray

    def unstack(self):
        return self._barray._unstack()

    def map(self, func):
        rdd = self._barray._rdd.map(lambda keys_arrs: (keys_arrs[0], func(keys_arrs[1])))
        split = self._barray.split
        arr = self._barray._constructor(rdd, shape=self.shape, split=split)
        return self._constructor(arr).__finalize__(self)

    def reduce(self, func):
        return self._barray.reduce(func)

    def __str__(self):
        s = "Stacked BoltArray\n"
        s += "mode: %s\n" % str(self._barray._mode)
        s += "shape: %s\n" % str(self.shape)
        s += "stack size: %s" % str(self.stack_size)
        return s

    def __repr__(self):
        return str(self)