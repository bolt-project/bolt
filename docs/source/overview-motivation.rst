Motivation
==========

We and others have worked with multidimensional arrays in both local and distributed environments across multiple languages. New distributed computing platforms, especially Spark, have made it easy to flexibly implement distributed workflows that scale well to potentially massive data sets. But these technologies do not provide much benefit, and may even slow down, workflows on smaller or medium sized data sets. 

We want to be able to build projects on an interface like NumPy's ``ndarray`` and know that we can leverage either local or distributed operations. And we want to target Python because of its rich libaries for scientific computing and machine learning. Specifically, we want an object that:

- implements most of the ``ndarray`` interface
- implements a subset of functional operators (e.g. ``map``, ``filter``, ``reduce``)
- supports a variety of backends (e.g. local, multi-core, distributed)

For distributed computation, we currently target Spark's ``RDD`` (resiliant distributed dataset), which provides an elegant API for functional operations (``map``, ``reduce``, ``join``, ``filter``, etc.) but is not easy to work with as an multidimensional array. Many other projects have invented the neccessary  abstractions (e.g. Thunder_, spylearn_, sparkit-learn_). We hope to solve this problem once well so others can build on it. 

.. _Thunder: https://github.com/thunder-project/thunder
.. _spylearn: https://github.com/ogrisel/spylearn
.. _sparkit-learn: https://github.com/lensacom/sparkit-learn

Bolt aims to implement an object with the properties list above. It currently supports both NumPy (for local computation), and Spark (for distributed computation), but we envision adding other backends in the future.

The project is in its early stages, so we welcome feedback, ideas, and use cases, just join us in the chatroom_.

.. _chatroom: https://gitter.im/bolt-project/bolt
