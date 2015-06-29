
class Blockable(object):
    """
    Interface for objects that can be converted into a blocked representation
    """

    @property
    def blocked(self):
        return BlockedBoltArray(self).block()

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

    def __init__(self, barray):
        self._barray = barray

    def block():
        self._barray = self._barray._block()
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
