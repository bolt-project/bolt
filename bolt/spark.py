from numpy import asarray, unravel_index, ravel_multi_index, arange, prod, random
from bolt.common import tupleize
from bolt.base import BoltArray


class BoltArraySpark(BoltArray):

    _metadata = BoltArray._metadata + ['_shape', '_split']

    def __init__(self, rdd, shape=None, split=None):
        self._rdd = rdd
        self._shape = shape
        self._split = split
        self._mode = 'spark'

    @property
    def _constructor(self):
        return BoltArraySpark

    @staticmethod
    def fromarray(arry, context, split=1):

        shape = arry.shape
        ndim = len(shape)

        if split < 1:
            raise ValueError("Split axis must be greater than 0, got %g" % split)
        if split > len(shape):
            raise ValueError("Split axis must not exceed number of axes %g, got %g" % (ndim, split))

        keyShape = shape[:split]
        valShape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(keyShape))), keyShape))
        vals = arry.reshape((prod(keyShape),) + valShape)

        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd, shape=shape, split=split)

    """
    Functional operators
    """

    # TODO handle shape changes
    # TODO add axes
    def map(self, func):
        return self._constructor(self._rdd.mapValues(func)).__finalize__(self)

    # TODO add axes
    def reduce(self, func):
        return self._constructor(self._rdd.values().reduce(func)).__finalize__(self)

    """
    Reductions
    """

    # TODO add axes
    def sum(self, axis=0):
        return self._constructor(self._rdd.sum()).__finalize__(self)

    """
    Slicing and indexing
    """

    def __getitem__(self):
        pass

    """
    Shaping operators
    """

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return prod(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def split(self):
        return self._split

    @property
    def mask(self):
        return tuple([1] * len(self.keyShape) + [0] * len(self.valueShape))

    @property
    def keyShape(self):
        return self.shape[:self.split]

    @property
    def valueShape(self):
        return self.shape[self.split:]

    def reshapeKeys(self, *new):

        new = tupleize(new)
        old = self.keyShape

        if new == old:
            return self

        if not prod(new) == prod(old):
            raise ValueError("Total size of new keys must remain unchanged")

        def f(k):
            return unravel_index(ravel_multi_index(k, old), new)

        newrdd = self._rdd.map(lambda (k, v): (f(k), v))
        newsplit = len(new)
        newshape = new + self.valueShape

        return self._constructor(newrdd, shape=newshape, split=newsplit)

    def reshapeValues(self, *new):

        new = tupleize(new)
        old = self.valueShape

        if new == old:
            return self

        if not prod(new) == prod(old):
            raise ValueError("Total size of new values must remain unchanged")

        def f(v):
            return v.reshape(new)

        newrdd = self._rdd.mapValues(f)
        newshape = self.keyShape + new

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    def transposeKeys(self, *new):

        new = tupleize(new)
        old = self.keyShape
        self._checkTranspose(new, old)

        if new == range(0, len(old)):
            return self

        def f(k):
            return tuple(k[i] for i in new)

        newrdd = self._rdd.map(lambda (k, v): (f(k), v))
        newshape = tuple(old[i] for i in new) + self.valueShape

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    def transposeValues(self, *new):

        new = tupleize(new)
        old = self.valueShape
        self._checkTranspose(new, old)

        if new == range(0, len(old)):
            return self

        if not len(new) == len(old):
            raise ValueError("Axes do not match axes of values")

        def f(v):
            return v.transpose(new)

        newrdd = self._rdd.mapValues(f)
        newshape = self.keyShape + tuple(old[i] for i in new)

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    def _exchange(self, to_values, to_keys):
        raise NotImplementedError

    def _configureKeyAxes(self, keyAxes):
        axis_set = set(keyAxes)
        to_keys = [a for a in keyAxes if self.parts[a] == 1]
        to_values = [a for a in range(len(self.shape)) if self.parts[a] == 0 and a not in axis_set]
        if to_keys or to_values:
            self._exhange(to_values, to_keys)

    def _checkKeyAxes(self, keyAxes):
        for axis in keyAxes:
            if axis > len(self.shape) - 1:
                raise ValueError("Axes not valid for an ndarray of shape: %s" % str(self.shape))

    def map(self, func, axes=(0,)):
        """
        Applies a function to every element across the specified axis.

        What about the scenario when a map returns an ndarray per mapped element??

        TODO: Better docstring
        """

        axes = sorted(axes)

        # Ensure that the specified axes are valid
        self._checkKeyAxes(axes)

        # Check if an exchange is necessary
        self._configureKeyAxes(axes)

        newrdd = self._rdd.map(func)

        # Try to compute the size of each mapped element by applying func to a random array
        element_shape = None
        try:
            element_shape = func(random.randn(*[self._shape[axis] for axis in axes])).shape
        except Exception:
            print "Failed to compute the shape of the result using a test array. Retrying with Spark job."
            first_elem = newrdd.first()
            if first_elem:
                element_shape = first_elem[1].shape

        # Reshaping will fail if the elements aren't uniformly shaped (is this necessary?)
        newrdd = newrdd.map(lambda v: v.reshape(element_shape))
        newshape = tuple([self._shape[axis] for axis in axes] + list(element_shape))

        return self._constructor(newrdd, newshape).__finalize__(self)

    def reduce(self, func, axes=(0,)):
        """

        Simple case:
        - (x, y, a, b) -> (5, 10, 15, 20)
        - reduce(func, axes=(0, 1))
        - (1, a, b) -> (1, 15, 20)

        Complicated case:
        - (x, y, a, b) -> (5, 10, 15, 20)
        - reduce(func, axes=(0, 2))
        - (x, y, a, b) -> (x, a, y, b)
        - (1, y, b) -> (1, 10, 20)

        newshape = (1, (shape of non-reduced axes))
        The ordering of the non-reduced axes is maintained after the reduce

        TODO: Better docstring
        """

        axes = sorted(axes)

        # Ensure that the specified axes are valid
        for axis in axes:
            if axis > len(self.shape) - 1:
                raise ValueError("Axes not valid for an ndarray of shape: %s" % str(self.shape))

        # Do an exchange if necessary
        self._configureKeyAxes()

        newrdd = self._rdd.reduce(func)
        newshape = tuple([self.shape[axis] for axis in axes])

        return self._constructor(newrdd, shape=newshape).__finalize__(self)

    @staticmethod
    def _checkTranspose(new, old):

        if not len(new) == len(old):
            raise ValueError("Axes do not match axes of keys")

        if not len(set(new)) == len(set(old)):
            raise ValueError("Repeated axes")

        if any(n < 0 for n in new) or max(new) > len(old) - 1:
            raise ValueError("Invalid axes")

    """
    Conversions
    """

    def tolocal(self):
        from bolt.local import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.sortByKey().values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        return self._rdd

    def display(self):
        return str(asarray(self._rdd.take(10)))

    def __str__(self):
        return str(asarray(self._rdd.collect()))
