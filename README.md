[![Build Status](https://travis-ci.org/bolt-project/bolt.svg?branch=master)](https://travis-ci.org/bolt-project/bolt)
[![Join the chat at https://gitter.im/bolt-project/bolt](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/bolt-project/bolt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Bolt
----
Bolt provides a Python interface to local and distributed multi-dimensional arrays. Aims to optimize performance whether data are small, medium, or very, very large, all through a common and familiar ndarray interface. The core is 100% Python. Currently backed by NumPy (local) or Spark (distributed) and will expand to others in the future.

Read more at [bolt-project.org](http://bolt-project.org)

Try live notebooks at [try.bolt-project.org](http://try.bolt-project.org)

Requirements
------------
Bolt supports Python 2.7+ and Python 3.4+. The core library is 100% Python, the only primary requirement is numpy, and for Spark functionality it requires Spark 1.4+ which can be obtained [here](http://spark.apache.org/downloads.html).

Installation
------------
```
$ pip install bolt-python
```
