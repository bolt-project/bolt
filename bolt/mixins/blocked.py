
class Blockable(object):
    """
    Interface for objects that can be converted into a blocked representation
    """

    def blocked(self, block_size=None):
        return BlockedBoltArray(self, block_size).block()

    @classmethod
    def _block(cls, to_block):
        raise NotImplementedError

    @classmethod
    def _unblock(self, to_unblock):
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

    def block():
        self._barray = self._barray._block(self.block_size)
        return self

    def unblock(self):
        return self._barray._unblock(self._barray)

    """
    BlockedBoltArray operations
    """

    def map(self, func):
        # TODO should BlockedBoltArray.map accept an axes argument?
        return BlockedBoltArray(self._barray.map(func))

    def reduce(self, func):
        # TODO should BlockedBoltArray.reduce accept an axes argument?
        return BlockedBoltArray(self._barray.reduce(func))
