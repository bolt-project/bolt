Bolt
----
Unified interface for local and distributed ndarrays. Backed by numpy or Spark. Optimal performance whether data are small, medium, or very, very large.

Goals
-----
Multidimensional arrays are core to a wide variety of applications. Some of these applications are well-suited to single machines, especially when datasets fit in memory. Other applications can benefit from distributed computing, especially for out-of-memory datasets and complex workflows. We see need for a single interface for using multidimensional arrays across these settings.

Bolt is a Python project currently built on numpy and Spark. Its primary object exposes ndarray operations and can use either local implementations (with numpy) or distributed operations (with Spark), and makes it easy to switch between them. The distributed operations leverage efficient data structures for multi-dimemsional array manipulation, and support most of the ndarray interface in a distributed setting.

Requirements
------------
Bolt supports Python 2.7 and 3.4, and its only primary dependency is numpy.

For Spark functionality, Bolt requires Spark 1.4+ which can be obtained [here](http://spark.apache.org/downloads.html).

Examples
--------

Let's create a `BoltArray` from an existing array (loading from external sources will be added soon).

```
>> from bolt import array
```

A local `BoltArray` is the default, and behaves just like an ndarray.
```
>> a = np.arange(18).reshape(2,3,3)
>> x = array(a)
>> x
BoltArray
mode: local
shape: (2, 3, 3)
```
We can easily turn it into a Spark version.
```
>> y = x.tospark(sc)
>> y
BoltArray
mode: spark
shape: (2, 3, 3)
```
And all ndarray operations will be distributed, including reductions along axes (`mean`, `max`), axis reordering and shaping (`reshape`, `transpose`), and slicing and indexing (`x[:,0:10:20]`).
```
>> y.filter(lambda x: sum(x) > 50)
BoltArray
mode: spark
shape: (1, 3, 3)

>> y.filter(lambda x: sum(x) > 50).squeeze()
BoltArray
mode: spark
shape: (3, 3)

>> y.transpose(1, 0, 2)
BoltArray
mode: spark
shape: (3, 2, 3)

>> y[0,1,0:2]
BoltArray
mode: spark
shape: (2,)
```
We can construct arrays in Spark directly, and control how it's parallelized through a single parameter, which determines which axes are represented by keys or values:
```
>> x = array(a, sc, axis=(0, 1))
>> x
BoltArray
mode: spark
shape: (2, 3, 3)
```
We aim to support sufficient array functionality so that downstream projects can use the bolt array like an `ndarray`
```
>> x = array(a, sc)
>> x.sum()
153
>> x.mean()
8.5
>> x.shape
(2, 3, 3)
```

And it's easy to chain local and distributed methods together!
```
>> array(a, sc, 0).filter(lambda y: np.sum(y) > 50).tolocal().sum(axis=0).tospark(sc, 1)
BoltArray
mode: spark
shape: (3, 3)
```

Running with Spark
-------------------
Using Spark with Bolt just requires that you have a valid `SparkContext` defined, and that you have Bolt installed (by calling `pip install bolt-python` on both the master and workers of your cluster). We cover the basics of starting a SparkContext here, but for details on setting up Spark in either a local environment, or on a cluster, consult the official documentation.

1) Launch Spark through the `pyspark` executable, at which point a `SparkContext` will already be defined as `sc`. If bolt's constructors (`array`, `ones`, `zeros`) are passed `sc`, it will automatically create a distributed array:

```
from bolt import ones
a = ones((100, 20), sc)
```

2) Write your application in a python script and submit it as a job using the `spark-submit` executable. You can then create a SparkContext within your job, and use it alongside bolt:

```
from pyspark import SparkContext
sc = SparkContext(appName='test', master='local')

from bolt import ones
a = ones((100, 20), sc)
```

2) Start `python` or `ipython`, setup Spark with the [`findspark`]() utility and then start a `SparkContext`:

```
import findspark
findspark.init()
sc = SparkContext(appName='test', master='local')

from bolt import ones
a = ones((100, 20), sc)
```
