Related projects
================

Bolt is related to several other projects in the pydata + big data space, but hopefully fills a complementary role. 

Spark itself has recently introduced a DataFrame API, which similarly enhances the RDD interface to be more data science friendly, but does not provide straightforward ndarray functionality. In some ways, our efforts here are similar but for ndarrays instead of DataFrames.

Dask is a complementary project that offers an alternative task scheduler / parallel execution engine, initially targeting single-machine multi-core workflows, and more recently supporting distributed implementations. It should be possible to provide a Dask backend, alongside local and Spark, for the Bolt array.

Another related Python projet, Blaze, is incredibly ambitious and exciting, aiming to provide a common interface to a wide variety of data structures and backends. But in practice, we found it very difficult to achieve the ndarray-focused functionality we wanted here.

On the JVM, the ND4J project is also trying to provide ndarray like functionality for either Spark or GPUs, but does not target unified local and distributed use. And the spark-timeseries_ project provides time series functionality to RDDs.

.. _spark-timeseries: https://github.com/cloudera/spark-timeseries