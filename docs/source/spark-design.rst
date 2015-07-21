Design
=======

Our design stems from several observations we've made working with ndarray-like objects in Spark:

- we usually want to parallelize functions across one or more axes, e.g. fitting a model to each of several data points, filtering each of several images, detrending each of several time series
- we want to easily and quickly change which axes we parallelize over
- we want to chain basic functional parallel operators alongside ndarray-style manipulations

How do we achieve this? The primitive object in Spark is a distributed collection of ``key,value`` pair records. The ``BoltArraySpark`` uses keys and values to separately represent axes of a single array. The keys are tuples that encode the indices of the "parallelized"
axes, while each value is a NumPy ``ndarray`` representing all the "localized" axes. For example, in a ``(2, 3, 4)`` array of ones

.. code:: python

 	>>> a = ones((2, 3, 4), sc)
 	>>> a.shape
 	(2, 3, 4)

each key is a tuple

.. code:: python

	>>> a.tordd().keys().collect()
	[(0,), (1,)]

and each value is a ``(3, 4)`` array

.. code:: python

	>>> [v.shape for v in a.tordd().values().collect()]
	[(3, 4), (3, 4)]

By convention, the key axes always come before the value axes, and we define an array's ``split`` as the number of key axes. During construction or loading, you can decide which axes to use as the keys. For example, just the first

.. code:: python

 	>>> a = ones((2, 3, 4), sc, axis=(0,))
 	>>> a.tordd().keys().collect()
 	[(0,), (1,)]
 	>>> a.split
 	1

or the first and second

.. code:: python

 	>>> b = ones((2, 3, 4), sc, axis=(0, 1))
 	>>> b.tordd().keys().collect()
 	[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
 	>>> b.split
 	2

Array methods are invariant to the choice of split, but performance can be strongly affected, especially when performing parallelized operations.

Swapping
--------

An important operation on the ``BoltArraySpark`` is changing which axes are parallelized. We call this "swapping" because it moves key axes to value axes (or vice versa), and it's the core of our ``transpose`` and ``reshape`` implementations.

The ``swap`` function takes a set of key axes that will be moved to value axes, and value axes that will be broken up to become key axes. By convention, the key axes that are swapped to value axes are placed *after* the split, and the value axes that move to the key axes are placed *before* the split. In both cases, the new axes have the same order as in the starting array. As examples,

.. code:: python

 	>>> a = ones((2, 3, 4), sc)
 	>>> a.shape
 	(2, 3, 4)
 	>>> a.swap(0, 1).shape
 	(4, 2, 3)
 	>>> a.swap((0,), (0, 1)).shape
 	(3, 4, 2)

One argument can be empty, for example, to move all axes into the keys. In this case, the shape stays the same
 	
.. code:: python

 	>>> b = a.swap((), (0, 1))
 	>>> a.shape
 	(2, 3, 4)
 	>>> b.shape
 	(2, 3, 4)

but the split has changed

.. code:: python

 	>>> a.split
 	1
 	>>> b.split
 	3

the keys are now three dimensional

.. code:: python

 	>>> b.tordd().keys().take(5)
 	[(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0)]

and there are more records reflecting greater parallelism, as expected

.. code:: python

	>>> a.tordd().count()
	2
	>>> b.tordd().count()
	24

Swapping can be expensive because it incurs a shuffle, but we have leveraged our experience doing these operations at scale to make it as efficient as possible. 

To understand our solution, consider two extremes. One one end, we could collect the entire array locally, reslice locally, and redistribute -- but that will fail on out-of-memory datasets. On the other end, we could break up the values into singletons, tag each with an index, and do a massive and expensive shuffle to put them back together.

Our approach is in the middle. We break up the values into "chunks", only along dimensions that are being moved, and use chunk sizes that minimize the number of objects shuffled while avoiding objects that are too large (this is a configurable parameter, but our default has proven efficient at scale in practice).

The entire process is lazy, which helps when composing it with other lazy operations, and properties like ``shape`` and ``split`` are automatically propagated.

Transposing / reshaping
-----------------------
The user-facing functions ``transpose`` and ``reshape`` are generally special cases of ``swap``, with one small modification: if the desired shape can be achieved by separately and independently manipulating the keys axes or values axes, we can avoid a shuffle, and just apply the neccessary operations via a ``map``. We identify the neccessary steps for any given requested ``transpose`` or ``reshape``, and choose the most efficient execution. As with ``swap``, these operations are lazy, though array properties are immediately updated.

Stacking / chunking
-------------------
For more fine-grained control over applying parallel operations to distributed arrays, we provide methods for both :ref:`stacking` (which groups records together for faster vectorized operations) and :ref:`chunking` (which breaks records apart for increased parallelization).
