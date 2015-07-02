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

def check_axes(shape, axes):
    """
    Checks to see if a list of axes are contained within the shape of a BoltArray

    Throws a ValueError if the axes are not valid for the given shape.

    Parameters
    ----------
    shape: tuple[int]
        the shape of a BoltArray
    axes: tuple[int]
        the axes to check against shape
    """
    valid = all([(axis < len(shape) - 1) and (axis >= 0) for axis in axes])
    if not valid:
        raise ValueError("axes not valid for an ndarray of shape: %s" % str(shape))

"""
Functions used in tests
"""

def allclose(a, b):
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)
