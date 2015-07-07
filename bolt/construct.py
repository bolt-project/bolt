class ConstructBase(object):

    @classmethod
    def dispatch(cls, method, *args, **kwargs):
        if method in cls.__dict__:
            return cls.__dict__[method].__func__(*args, **kwargs)
        else:
            raise NotImplementedError("Method %s not implemented on %s" % (method, cls.__name__))

    @staticmethod
    def _argcheck(*args, **kwargs):
        return False
