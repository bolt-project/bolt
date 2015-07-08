Design
=======

Here we delve into the core design underlying the Spark bolt array. Our considerations were based on the following observations having worked with ndarray-like objects in Spark:

- we usually want to parallelize operations across one or more axes, e.g. fitting a model to each of several data points, filtering each of several images, detrending each of several time series
- we want to easily and quickly change which axes we want to parallelize over
- we want to combine and chain basic functional parallel operators with ndarray-style manipulations

In Spark, our primitive is a distributed collection of ``key,value`` pairs, so we designed an object in which the keys and values separately represent different dimensions of a single array.

To make axis swapping efficient, we...

Chunking
--------

Transposing / reshaping
-----------------------

Stacking
--------

When calling ``map`` on a ``BoltArraySpark`` operations are applied in parallel to the ``value`` of each ``key,value`` record, which is a NumPy array representing a subset of axes. For some vectorized operations, it can be more efficient to apply functions to groups of records at once (represented as "stacked" NumPy arrays), and in these situations order tpyically does not matter. 

To support this use case, we provide a stacked representation of a Bolt array in Spark. The ``StackedArray`` combines all records in a single Spark partition. Operations performed on this object can leverage NumPy's single-core performance at the partition level, rather than at the usual ``key,value`` record level. 

A stacked representation can be accessed via the ``BoltArraySpark.stack()`` method. The optional ``size`` parameter describes the maximum number of records that can be incorporated into a stack. Note that the stacking operation will never combine records across partitions (which would incur a shuffle), so the actual stack sizes might be less than ``size``. Calling ``stack`` will return a reference to a ``StackedArray`` object which wraps an underlying bolt array and only provides access to a subset of its original methods. Currently, the only method supported on a `Stacked` object is ``map``, and it only operates over whichever axes were in the values when ``stack`` was called.

Calling ``unstack()`` on a ``StackedArray`` object will return the unstacked ``BoltArraySpark``, potentially transformed by a series of ``map`` operations. This is accomplished by calling Spark's ``flatMap`` on every stack, converting each back into a list of ``key,value`` pairs. 

When should you ``stack``? If you have a core function