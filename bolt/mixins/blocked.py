
class Blockable(object):
    """
    Interface for objects that can be converted into a blocked representation
    """

    def blocked(self, block_size=None):
        """
        The primary entry point for the BlockedBoltArray interface.

        Parameters
        ----------
        block_size: int, optional, default=None
            The maximum size for each block (number of original records per block)

        Returns
        -------
        a BlockedBoltArray

        """
        return BlockedBoltArray(self._block(block_size), block_size)

    def _block(self, block_size=None):
        """
        Converts a non-blocked Blockable into a blocked Blockable

        Returns
        -------
        a Blockable whose underlying representation has been converted into blocks
        """
        raise NotImplementedError

    def _unblock(self):
        """
        Splits apart the underlying representation of a blocked Blockable such that the new number
        of records is equal to that of the original.

        Returns
        -------
        a Blockable whose underlying blocked representation has been unblocked
        """
        raise NotImplementedError


class BlockedBoltArray(object):
    """
    Wraps a BoltArray and provides an interface for performing blocked operations (operations
    on whole subarrays). Many BoltArray methods will be restricted or forbidden until the
    BlockedBoltArray is unblocked.
    """

    def __init__(self, barray, block_size=None):
        self._barray = barray
        self.block_size = block_size

    @property
    def _constructor(self):
        return BlockedBoltArray

    def unblock(self):
        return self._barray._unblock()

    """
    BlockedBoltArray operations
    """

    def map(self, func):
        # TODO should BlockedBoltArray.map accept an axes argument?
        return self._constructor(self._barray.map(func))

    def reduce(self, func):
        # TODO should BlockedBoltArray.reduce accept an axes argument?
        return self._constructor(self._barray.reduce(func))

