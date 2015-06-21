Bolt
----
High-performance multidimensional arrays 

Goals
-----
We use multidimensional arrays all the time. When we're working on small or medium sized data we want single-core or single-machine code. But when we're working on large data sets with Spark we want distributed computation.

Bolt combines these two workflows into one through a common interface. Its primary object exposes numpy array operations and behind the scenes uses either local array operations or distributed operations, and makes it easy to switch between them.

Examples
--------

A simple constructor let's us create a `BoltArray` from existing arrays. Loading from external data sources will be added soon.

```
>> from bolt import barray
```

A local `BoltArray` is the default, and behaves just like a ndarray.
```
>> x = barray([1,2,3])
>> x
BoltArray
mode: local
value: [1 2 3]
```

To create a `BoltArray` backed by Spark, just pass a `SparkContext`.
```
>> x = barray([1,2,3], sc)
>> x
BoltArray
mode: spark
value: [1 2 3]
```

While we cannot complete method parity with the `ndarray`, our goal is to support enough that the two can be used interchangably by downstream projects. 
```
>> x = barray([1,2,3])
>> x.sum()
6
>> x.mean()
2

>> x = barray([1,2,3], sc)
>> x.sum()
6
>> x.mean()
2
```

It's easy to switch from one mode to the other.
```
>> x = barray([1,2,3])

>> x.tordd(sc)
BoltArray
mode: spark
value: [1 2 3]

>> x.tordd(sc).toarray()
BoltArray
mode: local
value: [1 2 3]
```