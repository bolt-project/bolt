from bolt.local.construct import ConstructLocal
from bolt.spark.construct import ConstructSpark

constructors = [
    ('local', ConstructLocal),
    ('spark', ConstructSpark)
]

def lookup(*args, **kwargs):
    if 'mode' in kwargs:
        mode = kwargs['mode']
        if mode not in constructors:
            raise ValueError('Mode %s not supported' % mode)
        del kwargs['mode']
        return constructors[mode]
    else:
        for mode, constructor in constructors:
            if constructor.argcheck(*args, **kwargs):
                return constructor
    return ConstructLocal


def array(*args, **kwargs):
    return lookup(*args, **kwargs).dispatch('array', *args, **kwargs)


def ones(*args, **kwargs):
    return lookup(*args, **kwargs).dispatch('ones', *args, **kwargs)


def zeros(*args, **kwargs):
    return lookup(*args, **kwargs).dispatch('zeros', *args, **kwargs)


def randn(*args, **kwargs):
    return lookup(*args, **kwargs).dispatch('randn', *args, **kwargs)