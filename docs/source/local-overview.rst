Overview
========

Overview of local...

Goal is to basically just wrap numpy whereever possible. Most of the core API is based on numpy.

But add functional operators. We chose ``map``, ``filter``, ``reduce``.

Like using Python's built-ins on ndarrays, but can apply them along one or more axes. May cause a reshape.

Also added conversion methods (e.g. to Spark, which just requires a SparkContext)
