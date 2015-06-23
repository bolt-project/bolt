Bolt
----
Multidimensional arrays in numpy and Spark via a common interface.

Optimal performance whether data are small, medium, or very, very lage.

Goals
-----
We use multidimensional arrays all the time. When we're working on small or medium sized data we want single-core or single-machine code. But when we're working on large data sets with Spark we want distributed computation.

Bolt combines these two workflows into one. Its primary object exposes numpy array operations and behind the scenes uses either local array operations or distributed operations, and makes it easy to switch between them. The distributed operations are powered by Spark, and leverage efficient data structures for multi-dimemsional array manipulations (e.g. reshaping and transposing).

Examples
--------

Let's create a `BoltArray` from an existing array (loading from external sources will be added soon).

```
>> from bolt import barray
```

A local `BoltArray` is the default, and behaves just like an ndarray.
```
>> a = numpy.arange(18).reshape(2,3,3)
>> x = barray(a)
>> x
BoltArray
mode: local
```
We can easily turn it into a Spark version.
```
>> y = x.tospark(sc)
>> y
BoltArray
mode: spark
```
And immediately start using distributed operations.
```
>> y.map(func)
BoltArray
mode: spark
```
We can construct the Spark version directly, and control how it's parallelized through a single parameter:
```
>> x = barray(a, sc, split=2)
>> x
BoltArray
mode: spark
```
We aim support enough functionality so that downstream projects can use it just like an `ndarray`
```
>> x = barray(a)
>> x.sum()
6
>> x.mean()
2
>> x.shape
(2, 3, 3)

>> x = barray(a, sc)
>> x.sum()
6
>> x.mean()
2
>> x.shape
(2, 3, 3)
```

And it's easy to chain together, fluidly mixing local and distributed computation!
```
>> barray(a).tospark(sc, 2).map(func).tolocal().sum(axis=0).tospark(sc, 1)
BoltArray
mode: spark
```
