from numpy import ndarray, asarray, prod, concatenate, expand_dims
from numpy import any as npany
from collections import Iterable

def tupleize(arg):
    """
    Coerce singletons and lists and ndarrays to tuples.

    Parameters
    ----------
    arg : tuple, list, ndarray, or singleton
        Item to coerce
    """
    if arg is None:
        return None
    if not isinstance(arg, (tuple, list, ndarray, Iterable)):
        return tuple((arg,))
    elif isinstance(arg, (list, ndarray)):
        return tuple(arg)
    elif isinstance(arg, Iterable) and not isinstance(arg, str):
        return tuple(arg)
    else:
        return arg

def argpack(args):
    """
    Coerce a list of arguments to a tuple.

    Parameters
    ----------
    args : tuple or nested tuple
        Pack arguments into a tuple, converting ((,...),) or (,) -> (,)
    """
    if isinstance(args[0], (tuple, list, ndarray)):
        return tupleize(args[0])
    elif isinstance(args[0], Iterable) and not isinstance(args[0], str):
        # coerce any iterable into a list before calling tupleize (Python 3 compatibility)
        return tupleize(list(args[0]))
    else:
        return tuple(args)

def inshape(shape, axes):
    """
    Checks to see if a list of axes are contained within an array shape.

    Parameters
    ----------
    shape : tuple[int]
        the shape of a BoltArray

    axes : tuple[int]
        the axes to check against shape
    """
    valid = all([(axis < len(shape)) and (axis >= 0) for axis in axes])
    if not valid:
        raise ValueError("axes not valid for an ndarray of shape: %s" % str(shape))

def allclose(a, b):
    """
    Test that a and b are close and match in shape.

    Parameters
    ----------
    a : ndarray
        First array to check

    b : ndarray
        First array to check
    """
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)

def tuplesort(seq):
    """
    Sort a list by a sequence.

    Parameters
    ----------
    seq : tuple
        Sequence to sort by
    """

    return sorted(range(len(seq)), key=seq.__getitem__)

def listify(lst, dim):
    """
    Flatten lists of indices and ensure bounded by a known dim.

    Parameters
    ----------
    lst : list
        List of integer indices

    dim : tuple
        Bounds for indices
    """
    if not all([l.dtype == int for l in lst]):
        raise ValueError("indices must be integers")

    if npany(asarray(lst) >= dim):
        raise ValueError("indices out of bounds for axis with size %s" % dim)

    return lst.flatten()

def slicify(slc, dim):
    """
    Force a slice to have defined start, stop, and step from a known dim.
    Start and stop will always be positive. Step may be negative.

    There is an exception where a negative step overflows the stop needs to have
    the default value set to -1. This is the only case of a negative start/stop
    value.

    Parameters
    ----------
    slc : slice or int
        The slice to modify, or int to convert to a slice

    dim : tuple
        Bound for slice
    """
    if isinstance(slc, slice):

        # default limits
        start = 0 if slc.start is None else slc.start
        stop = dim if slc.stop is None else slc.stop
        step = 1 if slc.step is None else slc.step
        # account for negative indices
        if start < 0: start += dim
        if stop < 0: stop += dim
        # account for over-flowing the bounds
        if step > 0:
            if start < 0: start = 0
            if stop > dim: stop = dim
        else:
            if stop < 0: stop = -1
            if start > dim: start = dim-1

        return slice(start, stop, step)

    elif isinstance(slc, int):
        if slc < 0:
            slc += dim
        return slice(slc, slc+1, 1)

    else:
        raise ValueError("Type for slice %s not recongized" % type(slc))

def istransposeable(new, old):
    """
    Check to see if a proposed tuple of axes is a valid permutation
    of an old set of axes. Checks length, axis repetion, and bounds.

    Parameters
    ----------
    new : tuple
        tuple of proposed axes

    old : tuple
        tuple of old axes
    """

    new, old = tupleize(new), tupleize(old)

    if not len(new) == len(old):
        raise ValueError("Axes do not match axes of keys")

    if not len(set(new)) == len(set(old)):
        raise ValueError("Repeated axes")

    if any(n < 0 for n in new) or max(new) > len(old) - 1:
        raise ValueError("Invalid axes")

def isreshapeable(new, old):
    """
    Check to see if a proposed tuple of axes is a valid reshaping of
    the old axes by ensuring that they can be factored.

    Parameters
    ----------
    new : tuple
        tuple of proposed axes

    old : tuple
        tuple of old axes
    """

    new, old = tupleize(new), tupleize(old)

    if not prod(new) == prod(old):
        raise ValueError("Total size of new keys must remain unchanged")

def allstack(vals, depth=0):
    """
    If an ndarray has been split into multiple chunks by splitting it along
    each axis at a number of locations, this function rebuilds the
    original array from chunks.

    Parameters
    ----------
    vals : nested lists of ndarrays
        each level of nesting of the lists representing a dimension of
        the original array.
    """
    if type(vals[0]) is ndarray:
        return concatenate(vals, axis=depth)
    else:
        return concatenate([allstack(x, depth+1) for x in vals], axis=depth)

def iterexpand(arry, extra):
    """
    Expand dimensions by iteratively append empty axes.

    Parameters
    ----------
    arry : ndarray
        The original array

    extra : int
        The number of empty axes to append
    """
    for d in range(arry.ndim, arry.ndim+extra):
        arry = expand_dims(arry, axis=d)
    return arry
