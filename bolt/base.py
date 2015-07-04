class BoltArray(object):

    _mode = None
    _metadata = ['_mode']

    def __finalize__(self, other):
        if isinstance(other, BoltArray):
            for name in self._metadata:
                other_attr = getattr(other, name, None)
                if (other_attr is not None) and (getattr(self, name, None) is None):
                    object.__setattr__(self, name, other_attr)
        return self

    @property
    def mode(self):
        return self._mode

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    @property
    def _constructor(self):
        return None

    def sum(self, axis):
        raise NotImplementedError

    def mean(self, axis):
        raise NotImplementedError

    def var(self, axis):
        raise NotImplementedError

    def std(self, axis):
        raise NotImplementedError

    def min(self, axis):
        raise NotImplementedError

    def max(self, axis):
        raise NotImplementedError

    def concatenate(self, arry, axis):
        raise NotImplementedError

    def transpose(self, axes):
        raise NotImplementedError

    def squeeze(self, axis):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def map(self, func, axes):
        raise NotImplementedError

    def reduce(self, func, axes):
        raise NotImplementedError

    def filter(self, func, axes):
        raise NotImplementedError

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += "shape: %s\n" % str(self.shape)
        return s
