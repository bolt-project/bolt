Overview
========

The Spark implementation of the bolt array uses an RDD of ``key,value`` pairs to represent different parts of a multi-dimensional array. We express common ndarray operations through distributed operatiions on this RDD, and keep track of shape parameters and how they are affected by manipulations.

The data structure is defined by which axes are represented in the keys, and which are represented in the values. The ``keys`` are tuples and the ``values`` are arrays. You can think of each key as providing an index along a subset of axes. Ordering doesn't matter, but the RDD will be sorted before converting into a local array to ensure the correct shape.

Most operations are lazy, except where it is neccessary to perform a computation before proceeding, usually because the shape of the resulting object depends on evaluation (e.g. in ``filter``).

As part of the core API, we expose the functional operators ``map``, ``filter``, and ``reduce``, which are like their counterparts on the RDD except with the additional ability to apply along a specified set of axes. In addition, the Spark methods ``cache`` and ``unpersist`` are available to control the cacheing of the underlying RDD, which can speed up certain workflows by storing raw data or intermediate results in distributed memory. We do not expose all other Spark operations so as to ensure manipulations that generate valid bolt arrays. However, the underlying RDD can always be accessed by developers via the ``tordd()`` method.