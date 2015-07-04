Bolt
----
Multidimensional arrays, backed by numpy or Spark via a common interface.

Optimal performance whether data are small, medium, or very, very lage.

Goals
-----
Multidimensional arrays are core to a wide variety of applications. Some of these applications are suited to single machines, whereas others can benefit from distributed computing. We see need for a single interface for using multidimensional arrays across these settings.

Bolt is a Python project currently built on numpy and Spark. Its primary object exposes numpy operations and can use either local implementations or distributed operations, and makes it easy to switch between them. The distributed operations are powered by Spark, and leverage efficient data structures for multi-dimemsional array manipulations.

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
And all operations will distributed, including both ndarray operations (`mean`, `max`, `squeeze`), functional operators (`map`, `reduce`, `filter`), and slicing / indexing.
```
>> y.filter(lambda x: sum(x) > 50)
BoltArray
mode: spark
shape: (1, 3, 3)

>> y.filter(lambda x: sum(x) > 50).squeeze()
BoltArray
mode: spark
shape: (3, 3)

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

And it's easy to chain local and stributed methods together!
```
>> array(a).tospark(sc, 0).filter(lambda y: np.sum(y) > 50).tolocal().sum(axis=0).tospark(sc, 1)
BoltArray
mode: spark
shape: (3, 3)
```
