.. meta::
   :description: Official documentation for the Bolt library
   :keywords: arrays, spark, python, big data, science

.. title::
  Bolt

.. header::
	Foo

.. raw:: html

  <img src='_static/header-logo.svg' width=700></img>


Multidimensional arrays are core to a wide variety of applications. Some are well-suited to single machines, especially when datasets fit in memory. Others can benefit from distributed computing, especially for out-of-memory datasets and complex workflows. We see a need for a single interface for using multidimensional arrays across these settings.

Bolt is a Python project currently built on NumPy and Spark. Its primary object exposes array operations through either local implementations (with NumPy) or distributed operations (with Spark), and makes it easy to switch between them. The distributed operations leverage efficient data structures for multidimensional array manipulation, and aim to support most of NumPy's familiar ndarray interface in a distributed setting.

This documentation is split into a general overview, and detailed descriptions of the ``local`` and ``spark`` implementations.


.. toctree::
   :maxdepth: 1

   overview.rst
   spark.rst
   local.rst
