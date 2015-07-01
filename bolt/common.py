from numpy import any, asarray

def tupleize(args):
    """
    Coerce a list of arguments to a tuple
    """
    if isinstance(args[0], tuple):
        return args[0]
    else:
        return tuple(args)

def listify(lst, dim):
    """
    Flatten lists of indices and ensure bounded by a known dim
    """
    if not all([l.dtype == int for l in lst]):
        raise ValueError("indices must be integers")

    if any(asarray(lst) >= dim):
        raise ValueError("indices out of bounds for axis with size %s" % dim)

    return lst.flatten()

def slicify(slc, dim):
    """
    Force a slice to have defined start, stop, and step from a known dim
    """
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
