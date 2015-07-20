[![Build Status](https://travis-ci.org/bolt-project/bolt.svg?branch=master)](https://travis-ci.org/bolt-project/bolt)
[![Join the chat at https://gitter.im/bolt-project/bolt](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/bolt-project/bolt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Bolt
----
Bolt provides a pure Python interface to local and distributed multi-dimensional arrays. Aims to optimize performance whether data are small, medium, or very, very large, all through a common and familiar ndarray interface. Currently backed by numpy (local) or Spark (distributed) but will expand to other backends in the future.

Read more at [bolt-project.org](http://bolt-project.org)

Requirements
------------
Bolt supports Python 2.7+ and Python 3.4+. The core library is 100% Python, the only primary requirement is numpy, and for Spark functionality it requires Spark 1.4+ which can be obtained [here](http://spark.apache.org/downloads.html).

Installation
------------
```
$ pip install bolt-python
```
