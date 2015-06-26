class BoltArray(object):

    _mode = None
    _metadata = ['_mode']

    @property
    def mode(self):
        return self._mode

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

    # TODO: where should we put this method?
    def _checkKeyAxes(self, keyAxes):
        for axis in keyAxes:
            if axis > len(self.shape) - 1:
                raise ValueError("Axes not valid for a BoltArray of shape: %s" % str(self.shape))

    def sum(self, axis):
        raise NotImplementedError

    """
    Functional operators
    """

    def map(self, func, axes=(0,)):
        raise NotImplementedError

    def reduce(self, func, axes=(0,)):
        raise NotImplementedError

    def filter(self, func, axes=(0,):
        raise NotImplementedError

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += str(self)
        return s
