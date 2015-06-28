class BoltArray(object):

    _mode = None
    _metadata = ['_mode']

    @property
    def mode(self):
        return self._mode

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def _constructor(self):
        return None

    def __finalize__(self, other):
        if isinstance(other, BoltArray):
            for name in self._metadata:
                otherAttr = getattr(other, name, None)
                if (otherAttr is not None) and (getattr(self, name, None) is None):
                    object.__setattr__(self, name, otherAttr)
        return self

    def sum(self, axis):
        raise NotImplementedError

    def map(self, func):
        raise NotImplementedError

    def reduce(self, func):
        raise NotImplementedError

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += "shape: %s\n" % str(self.shape)
        return s


class ChunkedBoltArray(object):
    """
    Wraps a BoltArray and provides an interface for performing chunked operations (operations
    on whole subarrays). Many BoltArray methods will be restricted or forbidden until the
    ChunkedBoltArray is unchunked.
    """

    def __init__(self, barray, shape=None, split=None):
        self._shape = shape if shape else barray.shape
        self._split = split if split else barray.split
        self._barray = barray

    def chunk():
        self._barray = self._barray._chunk()
        return self

    def unchunk(self):
        return self._barray._unchunk(self._barray, self._shpae, self._split)

    """
    ChunkedBoltArray operations
    """

    def map(self, func):
        # TODO should ChunkedBoltArray.map accept an axes argument?
        return ChunkedBoltArray(self._barray.map(func), self._shape, self._split)



