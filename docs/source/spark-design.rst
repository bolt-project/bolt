Design
=======

Here we delve into the core design underlying the Spark bolt array. Our considerations were based on the following observations having worked with ndarray-like objects in Spark:

- we usually want to parallelize operations across one or more axes, e.g. fitting a model to each of several data points, filtering each of several images, detrending each of several time series
- we want to easily and quickly change which axes we parallelize over
- we want to chain basic functional parallel operators with ndarray-style manipulations

In Spark, our primitive is a distributed collection of records, each of which takes the form of a ``key,value`` pair. We designed a ``BoltArraySpark`` object
in which the keys and values separately represent different dimensions of a single array. The keys are tuples that encode the indices of the "parallelized"
dimensions, while each value is a NumPy ``ndarray`` with one dimension for every "localized" dimension in the array. To help keep track of which dimensions
are "key-dimensions" and which are "value-dimensions", we impose a structure where the the key-dimensions always come before the value-dimensions. While this
choice does limit generality, we find that it is very useful for keeping track of which dimensions are parallelized and which are not -- an important distinction to
always keep in mind, as the efficiency of many operations depends on which type of dimensions they are applied to.. In line with this organization, we define an array's
"split" as the number of key-dimensions. This value is accessible through the ``split`` property.

Swapping
--------

One important operation for working with the ``BoltArraySpark`` is changing which axes are parallelized over. We call this operation "swapping" as it 
involves changing key-dimensions to value dimensions and vice-versa. Not only is this operation useful for users wishing to optimize distributed computations,
but it is also at the heart of transposing and reshaping these arrays. We expose this operation through the ``swap`` function. It takes a set of
key-dimensions that will be collected into value-dimensions, as well as a set of value-dimensions that will be broken up en route to
becoming key-dimensions. As we have decided that key-dimensions always precede valued-dimensions, this operation causes a corresponding transposition.
As a matter of convention, we specify that key-dimensions that are swapped to value-dimensions are placed immediately *after* the split, in the same order that they
appeared in the original array. On the other hand, the value-dimensions that move to the key-dimensions are placed immediately *before* the split (again,
in the same order in which they appeared in the original array).

This swapping operation can be very computational expensive -- changing how the data is distributed across the records can involve moving large volumes of data
across the Spark cluster. Thus, we have built on our experience working with these types of objects to make this operation as efficient as we can. The rationale
behind our swapping algorithm is perhaps best be understood by first thinking about two extreme approaches to solving this problem:

- At one extreme, one could collect every record to a single machine, construct the full array, slice it along the new set of key-dimensions, and finally parallelize these new values to their own records. This approach obviously fails in a dramatic way when the array is too large to fit in memory on a single machine. But even when working with a medium-sized dataset, it fails to leverage the distributed power of the Spark cluster.
- At the other end of the specturm, one could break the value (i.e subarray) in every record into singlton values, each tagged with its indices in the full array. Then these singltons could be shuffled around the cluster, being sorted into new records based on the indices corresponding to the new key-dimensions. Within these new records, the pieces could then be put back together into the correct array. Unlike the previous solution, this method does take advantege of the distributed computing power of the Spark cluster. However, in breaking the data up into singleton values, it generates an extremely large number of packets that must be shuffled around the network -- an strategy that is very costly.

Our algorithm seeks the happy medium between these two extremes. First, we break up the value subarrays into chunks, but only along the value-dimesions that are being
swapped to key-dimensions -- the other axes will remain together in the final result, so there is no need to break them apart only to have to put them
back together again. Second, we choose the number of chunks to strike a balance between the size and number of chunks that will be shuffled across the network.
Recognizing that this optimazation might be specific each computing environment, we also allow user-defined chunking strategies. One nice property of this
algorithm is that it can be implmented in a completely lazy fashion -- a fact that we fully exploit. While the actual computation behind ``swap`` is implemented
lazily, array properties such as ``shape`` and ``split`` will immediately be updated to reflect the outcome of the operation.

Transposing / reshaping
-----------------------

Transposing and reshaping are fundamental array operations. Our stipulation that key-dimensions always come before value dimensions means that we can be
explicit as to when a call to ``transpose`` or ``reshape`` will involving a (potentially costly) call to ``swap`` behind the scenes -- to wit, we distinguish between
two important cases:

1. when the transpose acts independently on the keys and values, we apply a fast ``map`` operation that that does not affect how is distributed across the partitions
2. when the transpose demands that keys and values be interchanged, we cannot avoid redistributing the data, but we every attempt to do this in an efficient manner

As with ``swap``, these operations are lazy, though array properties are immediately updated.

Stacking
--------

When calling ``map`` on a ``BoltArraySpark`` operations are applied in parallel to the ``value`` of each ``key,value`` record, which is a NumPy array representing a subset of axes. For some vectorized operations, it can be more efficient to apply functions to groups of records at once (represented as "stacked" NumPy arrays), and in these situations order tpyically does not matter. 

To support this use case, we provide a stacked representation of a Bolt array in Spark. The ``StackedArray`` combines all records in a single Spark partition. Operations performed on this object can leverage NumPy's single-core performance at the partition level, rather than at the usual ``key,value`` record level. 

A stacked representation can be accessed via the ``BoltArraySpark.stack()`` method. The optional ``size`` parameter describes the maximum number of records that can be incorporated into a stack. Note that the stacking operation will never combine records across partitions (which would incur a shuffle), so the actual stack sizes might be less than ``size``. Calling ``stack`` will return a reference to a ``StackedArray`` object which wraps an underlying bolt array and only provides access to a subset of its original methods. Currently, the only method supported on a `Stacked` object is ``map``, and it only operates over whichever axes were in the values when ``stack`` was called.

Calling ``unstack()`` on a ``StackedArray`` object will return the unstacked ``BoltArraySpark``, potentially transformed by a series of ``map`` operations. This is accomplished by calling Spark's ``flatMap`` on every stack, converting each back into a list of ``key,value`` pairs. 

When should you ``stack``? If you have a core function
