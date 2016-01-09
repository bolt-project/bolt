from bolt.local.construct import ConstructLocal
from bolt.spark.construct import ConstructSpark

constructors = [
    ('local', ConstructLocal),
    ('spark', ConstructSpark)
]

def wrapped(f):
    """
    Decorator to append routed docstrings
    """
    import inspect

    def extract(func):
        append = ""
        args = inspect.getargspec(func)
        for i, a in enumerate(args.args):
            if i < (len(args) - len(args.defaults)):
                append += str(a) + ", "
            else:
                default = args.defaults[i-len(args.defaults)]
                if hasattr(default, "__name__"):
                    default = default.__name__
                else:
                    default = str(default)
                append += str(a) + "=" + default + ", "
        append = append[:-2] + ")"
        return append

    doc = f.__doc__ + "\n"
    doc += "    local -> array(" + extract(getattr(ConstructLocal, f.__name__)) + "\n"
    doc += "    spark -> array(" + extract(getattr(ConstructSpark, f.__name__)) + "\n"
    f.__doc__ = doc
    return f

def lookup(*args, **kwargs):
    """
    Use arguments to route constructor.

    Applies a series of checks on arguments to identify constructor,
    starting with known keyword arguments, and then applying
    constructor-specific checks
    """
    if 'mode' in kwargs:
        mode = kwargs['mode']
        if mode not in constructors:
            raise ValueError('Mode %s not supported' % mode)
        del kwargs['mode']
        return constructors[mode]
    else:
        for mode, constructor in constructors:
            if constructor._argcheck(*args, **kwargs):
                return constructor
    return ConstructLocal

@wrapped
def array(*args, **kwargs):
    """
    Create a bolt array.
    """
    return lookup(*args, **kwargs).dispatch('array', *args, **kwargs)

@wrapped
def ones(*args, **kwargs):
    """
    Create a bolt array of ones.
    """
    return lookup(*args, **kwargs).dispatch('ones', *args, **kwargs)

@wrapped
def zeros(*args, **kwargs):
    """
    Create a bolt array of zeros.
    """
    return lookup(*args, **kwargs).dispatch('zeros', *args, **kwargs)

@wrapped
def concatenate(*args, **kwargs):
    """
    Create a bolt array of ones.
    """
    return lookup(*args, **kwargs).dispatch('concatenate', *args, **kwargs)