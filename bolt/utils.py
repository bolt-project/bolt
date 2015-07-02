def tupleize(args):
    """
    Coerce a list of arguments to a tuple
    """
    if isinstance(args[0], tuple):
        return args[0]
    else:
        return tuple(args)

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

def allclose(a, b):
    """
    Test that a and b are close and match in shape
    """
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)

def tuplesort(seq):

    return sorted(range(len(seq)), key=seq.__getitem__)
