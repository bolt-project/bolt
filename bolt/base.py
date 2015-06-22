class BoltArray(object):

    _mode = None

    @property
    def mode(self):
        return self._mode

    @property
    def _constructor(self):
        return None

    def sum(self, axis):
        raise NotImplementedError

    def map(self, func):
        raise NotImplementedError

    def reduce(self, func):
        raise NotImplementedError

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += "value: %s\n" % str(self)
        return s