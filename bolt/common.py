def tupleize(args):

    if isinstance(args[0], tuple):
        return args[0]
    else:
        return tuple(args)

def slicify(slc, dim):

    if isinstance(slc, slice):

        if slc.start is None and slc.stop is None and slc.step is None:
            return slice(0, dim, 1)

        elif slc.start is None and slc.step is None:
            return slice(0, slc.stop, 1)

        elif slc.stop is None and slc.step is None:
            return slice(slc.start, dim, 1)

        elif slc.step is None:
            return slice(slc.start, slc.stop, 1)

        else:
            return slc

    elif isinstance(slc, int):

        return slice(slc, slc+1, 1)

    else:
        raise ValueError("Type for slice %s not recongized" % type(slc))

def check_key_axes(barray, axes):
    """
    Checks to see if a list of axes are valid axes to iterate over during a functional operation.
    i.e. map(func, axes=(1,2)) only makes sense if the BoltArray's shape is >= 3
    """
    for axis in axes:
        if (axis > len(barray.shape) - 1) or (axis < 0):
            raise ValueError("Axes not valid for an ndarray of shape: %s" % str(self.shape))

"""
Functions used in tests
"""

def allclose(a, b):
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)
