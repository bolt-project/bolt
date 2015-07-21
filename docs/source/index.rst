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

Bolt is an Python project designed to faciliate working with ndarrays in local and distributed settings. It exposes array operations through either local or distributed implementations, and makes it easy to switch between them. The distributed implementation is currently backed by Spark, but will support other backends in the future. The distributed operations leverage efficient data structures for multidimensional array manipulation, and aim to cover most of NumPy's familiar ndarray interface in a distributed setting.

This documentation is split into a general overview_, info on installation_ and array construction_, and detailed descriptions of the local_ and spark_ implementations. 

For more information, visit the GitHub_ repo or join the chatroom_.

.. _overview: overview.html
.. _installation: installation.html
.. _construction: construction.html
.. _local: local.html
.. _spark: spark.html
.. _GitHub: https://github.com/bolt-project/bolt
.. _chatroom: https://gitter.im/bolt-project/bolt

.. toctree::
   :hidden:
   :maxdepth: 1

   overview.rst
   installation.rst
   construction.rst
   spark.rst
   local.rst
   contributing.rst


