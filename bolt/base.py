class BoltArray(object):

    _mode = None
    _metadata = {}

    def __finalize__(self, other):
        if isinstance(other, BoltArray):
            for name in self._metadata:
                other_attr = getattr(other, name, None)
                if (other_attr is not self._metadata[name]) \
                        and (getattr(self, name, None) is self._metadata[name]):
                    object.__setattr__(self, name, other_attr)
        return self

    @property
    def mode(self):
        return self._mode

    @property
    def shape(self):
        """
        Size of each dimension.
        """
        raise NotImplementedError

    @property
    def size(self):
        """
        Total number of elements.
        """
        raise NotImplementedError

    @property
    def ndim(self):
        """
        Number of dimensions.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        """
        Data-type of array.
        """
        raise NotImplementedError

    @property
    def _constructor(self):
        return None

    def sum(self, axis):
        """
        Return the sum of the array elements over the given axis.
        """
        raise NotImplementedError

    def mean(self, axis):
        """
        Return the mean of the array elements over the given axis.
        """
        raise NotImplementedError

    def var(self, axis):
        """
        Return the variance of the array elements over the given axis.
        """
        raise NotImplementedError

    def std(self, axis):
        """
        Return the standard deviation of the array elements over the given axis.
        """
        raise NotImplementedError

    def min(self, axis):
        """
        Return the minimum of the array elements over the given axis or axes.
        """
        raise NotImplementedError

    def max(self, axis):
        """
        Return the maximum of the array elements over the given axis or axes.
        """
        raise NotImplementedError

    def concatenate(self, arry, axis):
        raise NotImplementedError

    def transpose(self, axis):
        """
        Return an array with the axes transposed.
        """
        raise NotImplementedError

    @property
    def T(self):
        """
        Transpose by reversing the order of the axes.
        """
        raise NotImplementedError

    def reshape(self, axis):
        """
        Return an array with the same data but a new shape.
        """
        raise NotImplementedError

    def squeeze(self, axis):
        """
        Remove one or more single-dimensional axes from the array.
        """
        raise NotImplementedError

    def swapaxes(self, axis1, axis2):
        """
        Return an array with two axes interchanged.
        """
        raise NotImplementedError

    def astype(self, dtype, casting):
        """
        Cast the array to a specified type.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def map(self, func, axis):
        """
        Apply a function across one or more axes.
        """
        raise NotImplementedError

    def reduce(self, func, axis, keepdims):
        """
        Reduce an array across one or more axes.
        """
        raise NotImplementedError

    def filter(self, func, axis):
        """
        Filter an array across one or more axes.
        """
        raise NotImplementedError

    def first(self):
        """
        Return the first element of the array
        """
        raise NotImplementedError

    def __repr__(self):
        s = "BoltArray\n"
        s += "mode: %s\n" % self._mode
        s += "shape: %s\n" % str(self.shape)
        return s
