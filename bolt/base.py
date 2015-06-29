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
                other_attr = getattr(other, name, None)
                if (other_attr is not None) and (getattr(self, name, None) is None):
                    object.__setattr__(self, name, other_attr)
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