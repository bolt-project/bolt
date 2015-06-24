def tupleize(args):

    if isinstance(args[0], tuple):
        return args[0]
    else:
        return tuple(args)