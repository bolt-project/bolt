Overview
========

The local implementation of the Bolt array directly extends NumPy's ``ndarray``. As such, it as all the functionality of an ``ndarray``, but adds functionality to complete the `core API`_. Those additions are the functional operators ``map``, ``filter``, and ``reduce``. These behave like calling Python's built-in functional operators on an ``ndarray`` but allow functions to be applied along one or more axes, whereas ordinarily they are only applied along the first axis.

.. _core API: overview-methods.html

