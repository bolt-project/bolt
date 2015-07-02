def tupleize(args):
    """
    Coerce a list of arguments to a tuple
    """
    if isinstance(args[0], tuple):
        return args[0]
    else:
        return tuple(args)

def allclose(a, b):
    """
    Test that a and b are close and match in shape
    """
    from numpy import allclose
    return (a.shape == b.shape) and allclose(a, b)

def tuplesort(seq):

    return sorted(range(len(seq)), key=seq.__getitem__)
