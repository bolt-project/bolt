from numpy import array
from bolt.local import BoltArrayLocal
from bolt.spark import BoltArraySpark


def barray(input, context=None):

    if context is None:
        return BoltArrayLocal(array(input))

    else:
        try:
            from pyspark import SparkContext

        except ImportError:
            print("Spark is not avaialble")
            return

        return BoltArraySpark.fromarray(input, context)
