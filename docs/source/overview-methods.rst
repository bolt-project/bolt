Methods
=======

.. currentmodule:: bolt.base.BoltArray

The base object in ``bolt`` defines a core set of ndarray functionality, as well as a subset functional operators. Different implementations are provided for different settings. The current ones are:

.. |br| raw:: html

   <br />

``bolt.local`` for local computation with NumPy |br| ``bolt.spark`` for distributed computation with Spark

The core methods currently avaialble on all ``bolt`` arrays are as follows.

Reductions along axes:

.. autosummary::
   sum
   mean
   var
   std
   max
   min

Shaping/transposing:

.. autosummary::
   transpose
   squeeze
   swapaxes
   reshape
   shape
   ndim
   size
   T

Functional operators:

.. autosummary::
   map
   reduce
   filter
 
And also slicing (e.g. ``x[0:10, 0:100]``, ``x[0:10, :]``) and indexing (e.g. ``x[[0, 1, 2], [0, 1, 3]]``) 

We aim to replicate a large fraction of the NumPy API, so if there is something that we are missing that you would be interested in having, or something that you would like to contribute, create an issue.

For further details on the implementations, as well as functionality specific to the different modes, see the documentation for `bolt.spark`_ and `bolt.local`_.

.. _bolt.spark: spark.html
.. _bolt.local: local.html
