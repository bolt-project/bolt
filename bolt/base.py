class BoltArray(object):

    _mode = None

    @property
    def mode(self):
        return self._mode

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += "value: %s\n" % str(self)
        return s