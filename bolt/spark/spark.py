from numpy import asarray, unravel_index, ravel_multi_index, arange, prod, mod, divide, random
from bolt.common import tupleize, slicify
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

        key_shape = shape[:split]
        val_shape = shape[split:]

        keys = zip(*unravel_index(arange(0, int(prod(key_shape))), key_shape))
        vals = arry.reshape((prod(key_shape),) + val_shape)

        rdd = context.parallelize(zip(keys, vals))
        return BoltArraySpark(rdd, shape=shape, split=split)

    """
    Functional operators
    """

    def _configureKeyAxes(self, keyAxes):
        """
        Say the axes are (a, b, c, d) and split=1 so (a) -> (b, c, d)
            map(func, axes=(1,2)) -> keyAxes = (1,2)
            want new distribution to be (b, c) -> (a, d)
        """
        axis_set = set(keyAxes)

        # Find the value axes that should be moved into the keys (axis >= split)
        to_keys = [a for a in keyAxes if a >= self.split]

        # Find the key axes that should be moved into the values (axis < split)
        to_values = [a for a in range(split - 1) if a not in axis_set]

        if to_keys or to_values:
            self._swap(to_values, to_keys)

    def _functionalReshape(self, axes):
        """
        The common prefix for functional operations:
        1) Ensures that the specified axes are valid
        2) Swaps key/value axes if necessary so that the underlying RDD operation is applied to the correct records
        """
        # Ensure that the specified axes are valid
        checkKeyAxes(self, axes)

        # Check if an exchange is necessary
        self._configureKeyAxes(axes)

    def map(self, func, axes=(0,)):
        """
        Applies a function to every element across the specified axis.

        What about the scenario when a map returns an ndarray per mapped element??

        TODO: Better docstring
        """

        axes = sorted(axes)
        self._functionalPrefix(axes)

        newrdd = self._rdd.mapValues(func)

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
        def checkShape(v):
            if v.shape != element_shape:
                raise Exception("Map operation did not produce values of uniform shape.")
            return v
        newrdd = newrdd.mapValues(lambda v: checkShape(v))
        newshape = tuple([self._shape[axis] for axis in axes] + list(element_shape))

        return self._constructor(newrdd, shape=newshape, split=self.split).__finalize__(self)

    def filter(self, func, axes=(0,)):
        """

        Filter must do a count in order to get the shape, followed by a re-keying

        (x, y) -> (a, b)
        filter(func, axes=(0,2))
        (x, a) -> (y, b)

        Since arbitrary rows can be filtered out, the keys are just linearized after the filter.

        TODO: Better docstring
        """

        if len(axes) != 1:
            print "Filtering over multiple axes will not be supported until SparseBoltArray is implemented."
            return self

        axes = sorted(axes)
        self._functionalPrefix(axes)

        newrdd = self._rdd.values().filter(func)

        # Count the resulting array in order to reindex (linearize) the keys
        count = newrdd.count()

        remaining = [dim for dim in self.shape if dim not in axes]
        newshape = tuple([count] + remaining)

        return self._constructor(newrdd, shape=newshape, split=self.split).__finalize__(self)

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

        self._functionalPrefix(axes)

        newrdd = self._rdd.values().reduce(func)
        remaining = [dim for dim in self.shape if dim not in axes]
        newshape = tuple(remaining)

        return self._constructor(newrdd, shape=newshape, split=self.split).__finalize__(self)

    """
    Reductions
    """

    def sum(self, axes=(0,)):
        from numpy import sum
        return self._constructor(self.reduce(sum, axes)).__finalize__(self)

    def mean(self, axes=(0,)):
        from numpy import mean
        return self._constructor(self.reduce(mean, axes)).__finalize__(self)

    def max(self, axes=(0,)):
        from numpy import max
        return self._constructor(self.reduce(max, axes)).__finalize__(self)

    def min(self, axes=(0,)):
        from numpy import min
        return self._constructor(self.reduce(min, axes)).__finalize__(self)

    """
    Slicing and indexing
    """

    def __getitem__(self, index):

        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > self.ndim:
            raise ValueError("Too many indices for array")

        if len(index) < self.ndim:
            index += tuple([slice(0, None, None) for _ in range(self.ndim - len(index))])

        index = tuple([slicify(s, d) for (s, d) in zip(index, self.shape)])

        key_slices = index[0:self.split]
        value_slices = index[self.split:]

        def key_check(key):
            check = lambda kk, ss: ss.start <= kk < ss.stop and mod(kk - ss.start, ss.step) == 0
            out = [check(k, s) for k, s in zip(key, key_slices)]
            return all(out)

        def key_func(key):
            return tuple([k - s.start for k, s in zip(key, key_slices)])

        def value_func(value):
            return value[value_slices]

        filtered = self._rdd.filter(lambda (k, v): key_check(k))
        mapped = filtered.map(lambda (k, v): (key_func(k), value_func(v)))

        print(s)

        shape = tuple([divide(s.stop - s.start, s.step) + mod(s.stop - s.start, s.step)
                       for s in key_slices + value_slices])

        return self._constructor(mapped, shape=shape).__finalize__(self)

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
        return tuple([1] * len(self.keys.shape) + [0] * len(self.values.shape))

    @property
    def keys(self):
        return BoltArraySpark._Keys(self)

    @property
    def values(self):
        return BoltArraySpark._Values(self)

    class _Shapes(object):

        @property
        def shape(self):
            raise NotImplementedError

        def reshape(self):
            raise NotImplementedError

        def transpose(self):
            raise NotImplementedError

        @staticmethod
        def _istransposeable(new, old):

            if not len(new) == len(old):
                raise ValueError("Axes do not match axes of keys")

            if not len(set(new)) == len(set(old)):
                raise ValueError("Repeated axes")

            if any(n < 0 for n in new) or max(new) > len(old) - 1:
                raise ValueError("Invalid axes")

        @staticmethod
        def _isreshapable(new, old):

            if not prod(new) == prod(old):
                raise ValueError("Total size of new keys must remain unchanged")

    class _Keys(_Shapes):

        def __init__(self, barray):
            self._barray = barray

        @property
        def shape(self):
            return self._barray.shape[:self._barray.split]

        def reshape(self, *new):

            new = tupleize(new)
            old = self.shape
            self._isreshapable(new, old)

            if new == old:
                return self._barray

            def f(k):
                return unravel_index(ravel_multi_index(k, old), new)

            newrdd = self._barray._rdd.map(lambda (k, v): (f(k), v))
            newsplit = len(new)
            newshape = new + self._barray.values.shape

            return BoltArraySpark(newrdd, shape=newshape, split=newsplit)

        def transpose(self, *new):

            new = tupleize(new)
            old = self.shape
            self._istransposeable(new, old)

            if new == range(0, len(old)):
                return self._barray

            def f(k):
                return tuple(k[i] for i in new)

            newrdd = self._barray._rdd.map(lambda (k, v): (f(k), v))
            newshape = tuple(old[i] for i in new) + self._barray.values.shape

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self._barray)

        def __str__(self):
            s = "BoltArray Keys\n"
            s += "shape: %s" % str(self.shape)
            return s

        def __repr__(self):
            return str(self)

    class _Values(_Shapes):

        def __init__(self, barray):
            self._barray = barray

        @property
        def shape(self):
            return self._barray.shape[self._barray.split:]

        def reshape(self, *new):

            new = tupleize(new)
            old = self.shape
            self._isreshapable(new, old)

            if new == old:
                return self._barray

            def f(v):
                return v.reshape(new)

            newrdd = self._barray._rdd.mapValues(f)
            newshape = self._barray.keys.shape + new

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self._barray)

        def transpose(self, *new):

            new = tupleize(new)
            old = self.shape
            self._istransposeable(new, old)

            if new == range(0, len(old)):
                return self._barray

            def f(v):
                return v.transpose(new)

            newrdd = self._barray._rdd.mapValues(f)
            newshape = self._barray.keys.shape + tuple(old[i] for i in new)

            return BoltArraySpark(newrdd, shape=newshape).__finalize__(self._barray)

        def __str__(self):
            s = "BoltArray Values\n"
            s += "shape: %s" % str(self.shape)
            return s

        def __repr__(self):
            return str(self)

    """
    Conversions
    """

    def tolocal(self):
        from bolt.local.local import BoltArrayLocal
        return BoltArrayLocal(self.toarray())

    def toarray(self):
        x = self._rdd.sortByKey().values().collect()
        return asarray(x).reshape(self.shape)

    def tordd(self):
        return self._rdd

    def display(self):
        for x in self._rdd.take(10):
            print x
