# bolt

[![Latest Version](https://img.shields.io/pypi/v/bolt.svg?style=flat-square)](https://pypi.python.org/pypi/regional)
[![Build Status](https://img.shields.io/travis/bolt-project/bolt/master.svg?style=flat-square)](https://travis-ci.org/freeman-lab/regional) 

> python interface to local and distributed multi-dimensional arrays

The goal of `bolt` is to support array manipulation and computation whether data are small, medium, or very, very large, through a common and familiar ndarray interface. The core is 100% Python. Currently backed by [`numpy`](https://github.com/numpy/nump) (local) or [`spark`](https://github.com/apache/spark) (distributed) and will expand to others in the future.

Read more at [bolt-project.org](http://bolt-project.org)

Try live notebooks at [try.bolt-project.org](http://try.bolt-project.org)

Requirements
------------
Bolt supports Python 2.7+ and Python 3.4+. The core library is 100% Python, the only primary requirement is [`numpy`](https://github.com/numpy/numpy), and for [`spark`] functionality it requires 1.4+ which can be obtained [here](http://spark.apache.org/downloads.html).

Installation
------------
```
$ pip install bolt-python
```
