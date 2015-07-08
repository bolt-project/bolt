Construction
============

Local
-----

Constructing local bolt arrays is nearly identical to using NumPy's constructors.

.. currentmodule:: bolt.local.construct.ConstructLocal

.. autosummary::
	array
  	ones
  	zeros
  	concatenate


Spark
-----

Constructing bolt arrays in Spark is similar, except the constructors must be provided with a ``SparkContext``. This is normally provided when running Spark interactively, or created at the beginning of a Spark job. In addition, you can specify which axes will be distributed. Briefly, arrays are represented using a subset of axes as the keys. So a five dimensional array specified with ``axis=(0, 1)`` would be represented as ``key,value`` pairs where the keys are two-tuples and the values are three-dimensional arrays. Bolt is designed so that its methods are invariant to the choice of distributed axes, but the choice will affect performance of many operations.

.. currentmodule:: bolt.spark.construct.ConstructSpark

.. autosummary::
	array
  	ones
  	zeros
  	concatenate

Examples
--------

Comparing local and distributed constructors

.. code:: python

  >>> a = blt.ones((2, 3, 4))
  >>> a.shape
  (2, 3, 4)
  >>> a.mode
  local

.. code:: python

  >>> a = blt.ones((2, 3, 4), sc)
  >>> a.shape
  (2, 3, 4)
  >>> a.mode
  spark

Comparing different axis choices

.. code:: python

  >>> x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
  >>> blt.array(x, sc, axis=(0, 1)).shape
  (2, 3, 4)
  >>> blt.array(x, sc, axis=(0, 1, 2)).shape
  (2, 3, 4)

.. code:: python

  >>> blt.ones((2, 3, 4), sc, axis=(0, 1)).shape
  (2, 3, 4)
  >>> blt.zeros((2, 3, 4), sc, axis=(0,)).shape
  (2, 3, 4)



Detailed API
------------

.. currentmodule:: bolt.spark.construct

.. autoclass:: ConstructSpark
	:members:

.. currentmodule:: bolt.local.construct

.. autoclass:: ConstructLocal
	:members:
