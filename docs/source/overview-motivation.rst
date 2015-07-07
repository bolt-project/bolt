Motivation
==========

We and others have spent a lot of time working with ndarrays in both local and distributed environments across multiple languages. New distributed computing platforms, especially Spark, have made it easy and flexible to implement distributed workflows, and scale well to potentially massive data sets. But these technologies do not provide much benefit, and may even slow down, workflows on smaller or medium sized data sets. 

We want to be able to build projects on an ndarray-like object and know that we can make those operations as fast as possible by leveraging either local or distributed operations under the hood.

For local computation, Python already offers a powerful multidimensional array abstraction, numpy's ndarray. 

For distributed computation, Spark's RDD (resiliant distributed dataset) provides an elegant API for functional operations (``map``, ``reduce``, ``join``, ``filter``, etc.), but is not straightforward to work with as an ndarray, leaving many other projects to reinvent the abstraction. 

In the Thunder project, we implemented ndarray-like manipulations of distributed collections of key-value pairs (the RDD, Spark's primary abstraction), for the purpose of image and time series processing. Versions of this idea were also developed as part of the spylearn project, and now the sparkit-learn project. One of our goals with Bolt is to solve this problem once well, so that other projects can build on top of it and help expand it.

We chose Python because we feel the scientific computing stack is better developed than in, say, scala or java, and the rapid prototyping afforded by Python is valuable in scientific computing applications. 

