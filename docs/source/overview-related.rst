Related work
============

Bolt is related to complimentary projects in the python + big data space, but aims to play a unique role.

Dask_ is a complementary and exciting project that offers an alternative task scheduler / parallel execution engine to Spark. It also provides an ndarray interface through ``dask.array`` that runs on top of that engine. Dask initially targeted single-machine multi-core workflows but has recently added support for clusters. It's more mature, and currently has a lot of functionality that Bolt doesn't. At the same time, as Bolt's distributed array is backed by Spark, it's currently powered by a more mature distributed engine. For now, the answer of "which to use" will surely depend on a combination of compute resources, time constraints, and the nature of the problem. We are working to accurately quantify this space, and it's likely there are use cases in which each have advantages!

.. _Dask: https://github.com/ContinuumIO/dask

Spark_ itself has introduced a DataFrame API with Python bindings, which similarly enhances the RDD interface to be more data science friendly, but does not provide straightforward ndarray functionality. In some ways, our efforts here are similar but for ndarrays instead of DataFrames.

.. _Spark: https://spark-project.org

Blaze_, another related Python project, is more abstract and more ambitious, aiming to provide a common interface to a wide variety of data structures and backends. But in practice, we found it very difficult to achieve the ndarray-focused functionality we wanted here, and Spark integration was limited.

.. _Blaze: https://github.com/ContinuumIO/Blaze

As mentioned in motivation_, several other Python projects have implemented aspects of ndarray-style operations on Spark RDDs, including Thunder_ (for image and time series processing) and spylearn_ and sparkit-learn_ (for machine learning). We aim for a more general interface on which such projects could build.

.. _motivation: overview-motivation.html
.. _Thunder: https://github.com/thunder-project/thunder
.. _spylearn: https://github.com/ogrisel/spylearn
.. _sparkit-learn: https://github.com/lensacom/sparkit-learn

On the JVM, the ND4J_ project is also providing ndarray-like functionality using both Spark and GPUs, and spark-timeseries_ adds time series functionality to RDDs. 

.. _ND4J: https://github.com/deeplearning4j/nd4j
.. _spark-timeseries: https://github.com/cloudera/spark-timeseries
