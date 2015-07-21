Installation
============

Getting started
---------------

The core of Bolt is pure Python and the only dependency is ``numpy``, so installation is straightforward. Obtain Python 2.7+ or 3.4+ (we strongly recommend using Anaconda_), and then install with ``pip``

.. code:: bash

	$ pip install bolt-python

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

To use Bolt with one of its backends, follow the instructions below.

If you just want to play around with Bolt, try the live notebooks_ which use Docker and tmpnb_ to generate temporary interactive notebook environments with all dependencies loaded. The same notebooks are available in this repo_.

.. _notebooks: http://try.bolt-project.org
.. _tmpnb: https://github.com/jupyter/tmpnb
.. _repo: https://github.com/bolt-project/bolt-notebooks

Backends
--------

Local
^^^^^
The local backend just uses ``numpy``, so nothing special required:

.. code:: python

	>>> from bolt import ones
	>>> a = ones((2, 3, 4))
	>>> a.shape
	(2, 3, 4)

Spark
^^^^^

Bolt offers easy integration with Spark. Rather than make Spark a hard dependency, or add complex custom executables, using Bolt with Spark just requires that an existing valid ``SparkContext`` has been defined, either within an interactive notebook or inside an application. We cover the basics of a local installation here, but consult the official documentation_ for more information, especially for cluster deployment.

.. _deployment: http://spark.apache.org/docs/latest/cluster-overview.html
.. _documentation: http://spark.apache.org/docs/latest/index.html

For local testing, the easiest way to get Spark download a prepackaged version here_ (get version 1.4+, compiled for any version of Hadoop), or if you are on Mac OS X, you can install using homebrew:

.. _here: http://spark.apache.org/downloads.html

.. code:: bash

	$ brew install apache-spark

Then just find and run the ``pyspark`` executable in the ``bin`` folder of your Spark installation. 

With Spark installed and deployed, you can launch through the ``pyspark`` executable at which point a ``SparkContext`` will already be defined as ``sc``. To use it with Bolt pass ``sc`` as a constructor input:

.. code:: python

	>>> from bolt import ones
	>>> a = ones((100, 20), sc)
	>>> a.shape
	(100, 20)

If you write a Spark application in Python and submit it with ``spark-submit``, you would define a ``SparkContext`` within your application and then use it similarly with Bolt

.. code:: python
	
	>>> from pyspark import SparkContext
	>>> sc = SparkContext(appName='test', master='local')

	>>> from bolt import ones
	>>> a = ones((100, 20), sc)
	>>> a.shape
	(100, 20)
 
If you are using Spark on a cluster, you just need to run

.. code:: bash

	$ pip install bolt-python

on all the of the cluster nodes.

Docker image
------------

We provide a Docker image with Bolt and its backends installed and configured, alongside  an example Jupyter notebook. This is a great way to try out Bolt. The docker file is on GitHub_ and the image is hosted on Dockerhub_. To run the image on OS X follow these instructions:

.. _GitHub: http://github.com/bolt-project/bolt-docker
.. _Dockerhub: https://registry.hub.docker.com/u/freemanlab/bolt/

- Download and install boot2docker_ (if you don't have it already)

- Launch the ``boot2docker`` application from your ``Applications`` folder

- Type ``docker run -i -t -p 8888:8888 freemanlab/bolt``

- Point a web browser to ``http://192.168.59.103:8888/``

.. _boot2docker: https://github.com/boot2docker/osx-installer/releases/tag/v1.7.1