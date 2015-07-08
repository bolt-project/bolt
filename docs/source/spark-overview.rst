Overview
========

The Spark implementation of the Bolt array uses an RDD of ``key,value`` pairs to represent different parts of a multi-dimensional array. We express common ndarray operations through distributed operatiions on this RDD, and keep track of shape parameters and how they are affected by manipulations.

If we start by creating an array

.. code:: python

  >>> a = blt.ones((2, 3, 4), sc)

we can inspect its properties

.. code:: python

  >>> a.shape
  (2, 3, 4)
  >>> a.dtype
  numpy.float64

compute reductions

.. code:: python

  >>> a.sum()
  24.0
  >>> a.mean(axis=0).shape
  (3, 4)

or move axes around

.. code:: python

   >>> a.transpose(2, 0, 1).shape
   (4, 2, 3)
   >>> a.reshape(3, 2, 4).shape
   (3, 2, 4)
   >>> a.T.shape
   (4, 3, 2)

The data structure is defined by which axes are represented in the keys, and which are represented in the values. The ``keys`` are tuples and the ``values`` are ndarrays. You can think of each key as providing an index along a subset of axes. We call the axis at which we switch from keys to values the ``split``.

It's easy to inspect the "shape" of either part

.. code:: python

	>>> a = ones((2, 3, 4), sc, axis=0)
	>>> a.split
	1
	>>> a.keys.shape
	(2,)
	>>> a.values.shape
	(3, 4)

and change the form of paralleization during construction

.. code:: python

	>>> a = ones((2, 3, 4), sc, axis=(0, 1))
	>>> a.split
	2
	>>> a.keys.shape
	(2, 3)
	>>> a.values.shape
	(4,)

The generic ``swap`` operation moves axes between the keys and the values

.. code:: python

	>>> a = ones((2, 3, 4), sc, axis=0)
	>>> b = a.swap((), (0,))
	>>> a.keys.shape
	(2,)
	>>> b.keys.shape
	(2,3)

in this case the array itself didn't change dimension, we only changed the parallelization

.. code:: python

	>>> b.shape
	(2, 3, 4)

The user-facing ``transpose`` and ``reshape`` rely on ``swap`` to do their reorganization.

Ordering does not matter, but the RDD will be sorted before converting into a local array to ensure correct structure

.. code:: python
	
	>>> a.sum(axis=(0,1)).toarray()
	array([ 6.,  6.,  6.,  6.])

As part of the `core API`_, we expose the functional operators ``map``, ``filter``, and ``reduce``, which are like their counterparts on the RDD except with the additional ability to be applied along a specified set of axes. 

.. _core API: overview-methods.html

.. code:: python

	>>> a = ones((2, 3, 4), sc)
	>>> a.shape
	(2, 3, 4)
	>>> a.map(lambda x: x.sum(), axis=0).shape
	(2, )
	>>> a.map(lambda x: x.sum(), axis=(0, 1)).shape
	(2, 3)

We do not expose other Spark operations in order to ensure that manipulations generate valid Bolt arrays. However, the underlying RDD can always be accessed by developers via the ``tordd()`` method.

.. code:: python

	>>> a = ones((2, 3), sc)
	>>> a.todd().collect()
	[((0,), array([ 1.,  1.,  1.])), ((1,), array([ 1.,  1.,  1.]))]

In addition, the Spark methods ``cache`` and ``unpersist`` are available to control the caching of the underlying RDD, which can speed up certain workflows by storing raw data or intermediate results in distributed memory. 

.. code:: python

	>>> a = ones((2, 3, 4), sc)
	>>> a.cache()
	>>> a.tordd().is_cached
	True

Most operations are lazy, except where it is neccessary to perform a computation before proceeding -- usually because the shape of the resulting object depends on evaluation (e.g. in ``filter``).

For more info, read the design_ section for details on implementation, and see the full `API documentation`_.

.. _design: spark-design.html
.. _API documentation: spark-api.html



