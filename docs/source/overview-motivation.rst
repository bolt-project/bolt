Motivation
==========

We and others have worked with ndarrays in both local and distributed environments across multiple languages. New distributed computing platforms, especially Spark, have made it easy and flexible to implement distributed workflows, and scale well to potentially massive data sets. But these technologies do not provide much benefit, and may even slow down, workflows on smaller or medium sized data sets. 

We want to be able to build projects on an ndarray-like interface and know that we can leverage either local or distributed operations under the hood. And we want to target Python because of its rich libaries for scientific computing and machine learning.

For local computation, Python already offers a powerful multidimensional array abstraction, numpy's ``ndarray``.

For distributed computation, Spark's ``RDD`` (resiliant distributed dataset) provides an elegant API for functional operations (``map``, ``reduce``, ``join``, ``filter``, etc.), but is not straightforward to work with as an ``ndarray``, leaving many other projects to invent the neccessary abstractions. For example, in working on Thunder_, we implemented ndarray-like manipulations on distributed collections of key-value pairs, for the purpose of image and time series processing. Variants on this idea were also developed as part of spylearn_, and now sparkit-learn_. 

One of the goals of Bolt is to solve this problem once well, so that others can build on top of it. If you have a project that could leverage this functionality, come chat_ with us!

.. _Thunder: https://github.com/thunder-project/thunder
.. _spylearn: https://github.com/ogrisel/spylearn
.. _sparkit-learn: https://github.com/lensacom/sparkit-learn
.. _chat: https://gitter.im/bolt-project/bolt
