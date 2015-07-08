Design
=======

Here we delve into the core design underlying the Spark bolt array. Our considerations were based on our experiences working with ndarray-like objects in Spark.

- we usually want to parallelize operations across one or more axes, e.g. fitting a model to each of several features, filtering each of several images
- we want to easily and quickly change which axes we want to parallelize over
- we want to chain basic functional parallel operators with ndarray-style manipulations

In Spark, our primitive is a distributed collection of ``key,value`` pairs, so we designed an object in which the keys and values separately represent different dimensions of a single array.

To make axis swapping efficient, we...

Describe chunking

Describe transposing / reshaping

Stacking
--------

It is often more efficient to apply vectorized NumPy functions to a distributed set of medium-sized arrays, rather than to each row (a small array) independently. In many operations, such as element-wise multiplication, the ordering in which an operation is applied to an array's rows is not relevant to the correctness of the computation. Using this property, we can cheaply aggregate and stack (using `vstack`) all array records contained in a single Spark partition, returning a "stacked" representation of the original array. Operations performed on this object can leverage NumPy's single-core performance at the partition level, rather than at the usual `key,value` record level. 

A stacked representation of a `BoltArraySpark` can be accessed via the `BoltArraySpark.stack(stack_size=None)` method. The optional `stack_size` parameter describes the maximum number of records that can be incorporated into a stack. Note that the stacking operation will never combine records across partitions (which would incur a shuffle), so the actual stack sizes might be less than `stack_size`. Calling `stack` will return a reference to a `Stacked` object which wraps an underlying `BoltArraySpark` and only provides access to a subset of the array's original methods. Currently, the only methods supported on a `Stacked` object are: 

- `Stacked.map(func)`
- `Stacked.reduce(func)`

Neither map nor reduce currently supports operations over multiple axes (they both operate over axis 0).

Calling `unstack()` on a `Stacked` object will return the unstacked `BoltArraySpark`, potentially transformed by a series of `map` and `reduce` operations. `unstack` works by calling Spark's `flatMap` on every stack, converting each stack back into a list of `key,value` pairs. 
