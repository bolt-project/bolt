
class Stackable(object):
    """
    Interface for objects that can be converted into a stacked representation
    """

    def stacked(self, stack_size=None):
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

    @property
    def _constructor(self):
        return StackedBoltArray

    def unstack(self):
        return self._barray._unstack()

    """
    StackedBoltArray operations
    """

    def map(self, func):
        # TODO should StackedBoltArray.map accept an axes argument?
        return self._constructor(self._barray.map(func))

    def reduce(self, func):
        # TODO should StackedBoltArray.reduce accept an axes argument?
        return self._constructor(self._barray.reduce(func))

