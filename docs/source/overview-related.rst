Related work
============

Bolt is related to sevelal complimentary projects in the pydata + big data space, but aims to play a unique role.

As mentioned in motivation_, several other projects have implemented aspects of ndarray style operations on Spark RDDs, including Thunder_ (for image and time series processing) and spylearn_ and sparkit-learn_ (for machine learning). We hope that the Bolt array is a more general interface on which these and similar projects could build.

.. _motivation: overview-motivation.html
.. _Thunder: https://github.com/thunder-project/thunder
.. _spylearn: https://github.com/ogrisel/spylearn
.. _sparkit-learn: https://github.com/lensacom/sparkit-learn

Dask_ is a complementary project that offers an alternative task scheduler / parallel execution engine to Spark, initially targeting single-machine multi-core workflows, and more recently supporting distributed implementations. It should be possible to provide a Dask backend, alongside local and Spark, for the Bolt array.

.. _Dask: https://github.com/ContinuumIO/dask

Another related Python projet, Blaze_, is more abstract and more ambitious, aiming to provide a common interface to a wide variety of data structures and backends. But in practice, we found it very difficult to achieve the ndarray-focused functionality we wanted here, and Spark integration was limited.

.. _Blaze: https://github.com/ContinuumIO/Blaze

On the JVM, the ND4J_ project is also trying to provide ndarray-like functionality that incorporates Spark, as well as leveraging GPUs. And the spark-timeseries_ project provides time series functionality to RDDs. And Spark itself has recently introduced a DataFrame API, which similarly enhances the RDD interface to be more data science friendly, but does not provide straightforward ndarray functionality. In some ways, our efforts here are similar but for ndarrays instead of DataFrames.

.. _spark-timeseries: https://github.com/cloudera/spark-timeseries
.. _ND4J: https://github.com/deeplearning4j/nd4j
