from numpy import ndarray, argwhere

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

    elif isinstance(slc, list):
        if len(slc) > dim:
            raise ValueError("List cannot be larger than the dimension")

        # make sure they're all ints
        for item in slc:
            if type(item) != int:
                raise ValueError("If a list, all members must be integers.")

        return set(slc)
    
    elif isinstance(slc, ndarray):
        if slc.shape[0] > dim:
            raise ValueError("Boolean indexing array size much match dimension.")
        
        if slc.dtype == bool:
            return set(argwhere(slc).flatten().tolist())
        else:
            return set(slc.astype(int).tolist())

    elif isinstance(slc, set):
        return slc

    else:
        raise ValueError("Type for slice %s not recongized" % type(slc))

