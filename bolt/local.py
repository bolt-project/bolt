from numpy import ndarray, asarray, apply_over_axes, ufunc, prod
from bolt.base import BoltArray


class BoltArrayLocal(ndarray, BoltArray):

    def __new__(cls, array):
        obj = asarray(array).view(cls)
        obj._mode = 'local'
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._mode = getattr(obj, 'mode', None)

    @property
    def _constructor(self):
        return BoltArrayLocal

    """
    Functional operators
    """

    def _functionalReshape(self, axes, key_shape=None):
        # Compute the set of dimensions/axes that will be used to reshape
        remaining = [dim for dim in range(len(self.shape)) if dim not in axes]
        key_shape = key_shape if key_shape else [self.shape[axis] for axis in axes]
        remaining_shape = [self.shape[axis] for axis in remaining]
        linearized_shape = [prod(key_shape)] + remaining_shape

        # Compute the transpose permutation
        transpose_order = axes + remaining

        # Transpose the array so that the keys being mapped over come first, then linearize the keys
        reshaped = self.transpose(*transpose_order).reshape(*linearized_shape)

        return reshaped

    def filter(self, func, axes=(0,)):
        """
        """
        axes = sorted(axes)
        self._checkKeyAxes(axes)

        reshaped = self._functionalReshape(axes)

        filtered = filter(func, reshaped)

        return self._constructor(filtered)

    def map(self, func, axes=(0,)):
        """
        shape = (10,20,30,40)
        axes = (2,3)
        remaining = (0,1)
        transpose_order = [2,3,0,1]
        linearized_shape = [30,40,10,20]
        transpose_order_inv = [0,1,2,3]
        linearized_shape_inv = [10,20,30,40]

        (x, y, a, b)
        map(x, a)
        (x, a, y, b)
        (x, a, y, b)
        (x, y, a, b)
        """

        axes = sorted(axes)
        self._checkKeyAxes(axes)
        key_shape = [self.shape[axis] for axis in axes]

        reshaped = self._functionalReshape(axes, key_shape=key_shape)

        mapped = asarray(map(func, reshaped))
        elem_shape = mapped[0].shape

        # Invert the previous reshape operation, using the shape of the map result
        linearized_shape_inv = key_shape + list(elem_shape)
        reordered = mapped.reshape(*linearized_shape_inv)

        return self._constructor(reordered)

    def reduce(self, func, axes=(0,)):
        """
        (10, 20, 30, 40)
        reduce(func, axes(2,))
        (30, 10, 20, 40)
        (30, 20, 10, 40)
        """

        axes = sorted(axes)
        self._checkKeyAxes(axes)

        reduced = None
        # If the function is a ufunc, it can automatically handle reducing over multiple axes
        if isinstance(func, ufunc):
            reduced = func.reduce(self, axis=tuple(axes))
        else:
            reshaped = self._functionalReshape(axes)
            reduced = reduce(func, reshaped)

        return self._constructor(reduced)

    """
    Conversions
    """

    def tospark(self, sc, split=1):
        from bolt.spark import BoltArraySpark
        return BoltArraySpark.fromarray(self.toarray(), sc, split)

    def tordd(self, sc, split=1):
        from bolt.spark import BoltArraySpark
        return BoltArraySpark.fromarray(self.toarray(), sc, split).tordd()

    def toarray(self):
        return asarray(self)

    def display(self):
        return str(self)

    def __repr__(self):
        return BoltArray.__repr__(self)