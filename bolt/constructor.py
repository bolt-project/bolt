from numpy import asarray
from bolt.local import BoltArrayLocal
from bolt.spark import BoltArraySpark


def barray(input, context=None, split=1):

    if context is None:
        return BoltArrayLocal(asarray(input))

    else:
        try:
            from pyspark import SparkContext

        except ImportError:
            print("Spark is not avaialble")
            return

        return BoltArraySpark.fromarray(asarray(input), context=context, split=split)
