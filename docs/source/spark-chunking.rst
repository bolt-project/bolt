.. _chunking:

Chunking
========

When parallelizing operations over distributed arrays, sometimes you'll want more fine-grained control than just which axis to operate over. We expose a `chunk` method for this use case, which is actually the same operation we use under the hood for distributed transposes.

To chunk an array, just call `chunk` and specify the desidred size of each chunk along each axis of the arrays in the values. For example, starting with a simple array 

.. code:: python

 	>>> a = ones((5, 20, 10), sc)
 	>>> a.shape
 	(5, 20, 10)

We can chunk into subarrays of shape `(5,5)`.

.. code:: python

	>>> c = a.chunk((5, 5))

The "shape" is still the same

.. code:: python

	>>> c.shape
	(5, 20, 10)

But if we look at the first record of the underlying RDD and compare to the original, we see that the shapes are different

.. code:: python

	>>> a.tordd().values().first().shape
	(20, 10)

	>>> c.tordd().values().first().shape
	(5, 5)

While narray is chunked you can perform ``map`` operations which operate in parallel over the chunks (including operations that change shape)


	>>> c.map(lambda x: x * 2)
	>>> c.map(lambda x: x.sum(axis=0))

And you can restore the original with ``unchunk``

	>>> c.map(lambda x: x * 2).unchunk().toarray().shape
	(5, 20, 10)
	>>> c.map(lambda x: x.sum(axis=0)).unchunk().toarray().shape
	(5, 20, 2)

Chunking is especially useful when elements defined by even individual axes or pairs of axes are very large (such as long time series or large images) and you want to operate in parallel over portions (like windows or tiles) and then continue with other array operations.


