"""
Functions commonly used in BoltArray methods
"""

def checkKeyAxes(barray, axes):
    """
    Checks to see if a list of axes are valid axes to iterate over during a functional operation.
    i.e. map(func, axes=(1,2)) only makes sense if the BoltArray's shape is >= 3
    """
    for axis in axes:
        if (axis > len(barray.shape) - 1) or (axis < 0):
            raise ValueError("Axes not valid for an ndarray of shape: %s" % str(self.shape))

