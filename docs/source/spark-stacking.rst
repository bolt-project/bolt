.. _stacking:

Stacking
========

A common use case for distributed arrays is applying functions in parallel. In general, when calling ``map``, functions will be applied to the ``value`` of each ``key,value`` record, which is a NumPy array representing a subset of axes. For example, if we have the following Bolt array:

.. code:: python

 	>>> a = ones((100, 5), sc)
 	>>> a.shape
 	(100, 5)

each value is a ``(5,)`` array

.. code:: python

	>>> a.tordd().values().first().shape
	(5,)

For some operations, it can be more efficient to apply functions to groups of records at once. For example, when calling one of the ``partial_fit`` methods from ``scikit-learn``, which take an arbitrary number of data points and estimate model parameters. We provide a method ``stack`` to aggregate records within partitions. The resulting ``StackedArray`` has the same intrinstic shape, but records have been aggregated to leverage faster performance by operating on larger arrays. The only parameter is the ``size``, the number of records aggregated per partition.

.. code:: python

	>>> s = a.stack(size=10)
	>>> s.shape
	(100, 5)
	>>> s.tordd().values().first().shape
	(10, 5)

To ensure proper shape handling, we restrict functionality to ``map``, and the mapped function must return an ``ndarray``. We automatically infer and propagate transformations of shape, and after applying a set of function(s) you can recreate a Bolt array using ``unstack``.

As an example use case, imagine we have one hundred 5-d points we want to cluster

.. code:: python

	>>> a = ones((100, 5), sc)
	>>> from sklearn.cluster import MiniBatchKMeans
	>>> km = MiniBatchKMeans(n_clusters=2)

if we stack into groups of 5 and apply a ``partial_fit``, we end up with 20 fitted models, each a ``(2, 5)`` array

.. code:: python

	>>> fits = a.stack(5).map(lambda x: km.partial_fit(x).cluster_centers_).unstack()
	>>> fits.shape
	(20, 2, 5)

which we can use to estimate a model average

.. code:: python

	>>> fits.mean(axis=0).shape
	(2, 5)
