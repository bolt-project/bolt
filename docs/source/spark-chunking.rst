Chunking
========

Another common use case for distributed arrays is breaking the values into chunks, and then performing parallized operations over the resulting subarrays (and optionally rebuilding the original array). Bolt uses this kind of chunking under the hood during it's :ref:`swap` operations, but we also expose chunking directly.

To chunk an array, just call `chunk` and specify the desidred size of each chunk along each axis

.. code:: python

 	>>> a = ones((100, 5), sc)
 	>>> a.shape
 	(100, 5)

