Design
=======

Here we delve into the core design underlying the Spark bolt array. Our considerations were based on our experiences working with ndarray-like objects in Spark.

- we usually want to parallelize operations across one or more axes, e.g. fitting a model to each of several features, filtering each of several images
- we want to easily and quickly change which axes we want to parallelize over
- we want to chain basic functional parallel operators with ndarray-style manipulations

In Spark, our primitive is a distributed collection of ``key,value`` pairs, so we designed an object in which the keys and values separately represent different dimensions of a single array.

To make axis swapping efficient, we...

Describe chunking

Describe transposing / reshaping

Describe stacking (useful alternate representation for some workflows)