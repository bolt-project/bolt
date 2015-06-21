Bolt
----
High-performance multidimensional arrays 

Goals
-----
We use multidimensional arrays all the time. Sometimes we're working on small or medium sized data, and can rely on single-core or single-machine computation. Sometimes we're working on large data sets on a Spark cluster, and want to distribute computation using its powerful executation engine. 

Bolt combines these two workflows into one through a common interface. Its primary object exposes numpy array operations and behind the scenes uses either local array operations or distributed operations, and makes it easy to switch between them.

Examples
--------

A simple constructor let's us create a `BoltArray` from existing arrays. Loading from data sources will be provided.

```
from bolt import barray
```

A local `BoltArray` is the default, and behaves just like a ndarray.
```
x = barray([1,2,3])
x
>> BoltArray
>> mode: local
>> value: [1, 2, 3]
```

To create a `BoltArray` backed by Spark, just pass a `SparkContext`.
```
x = barray([1,2,3], sc)
x
>> BoltArray
>> mode: spark
>> value: [1, 2, 3]
```

While we cannot complete method parity with the `ndarray`, our goal is to support enough that the two can be used interchangably by downstream projects. 
```
x = barray([1,2,3])
x.sum()
>> 6
x.mean()
>> 2

x = barray([1,2,3], sc)
x.sum()
>> 6
x.mean()
>> 2
```