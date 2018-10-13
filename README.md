# bolt

[![Latest Version](https://img.shields.io/pypi/v/bolt-python.svg?style=flat-square)](https://pypi.python.org/pypi/bolt-python)
[![Build Status](https://img.shields.io/travis/bolt-project/bolt/master.svg?style=flat-square)](https://travis-ci.org/bolt-project/bolt) 

> python interface to local and distributed multi-dimensional arrays

The goal of `bolt` is to support array manipulation and computation whether data are small, medium, or very, very large, through a common and familiar ndarray interface. The core is 100% Python. Currently backed by [`numpy`](https://github.com/numpy/nump) (local) or [`spark`](https://github.com/apache/spark) (distributed) and will expand to others in the future.

View the documentation at [bolt-project.github.io/](http://bolt-project.github.io/)

Requirements
------------
Bolt supports Python 2.7+ and Python 3.4+. The core library is 100% Python, the only primary requirement is [`numpy`](https://github.com/numpy/numpy), and for [`spark`](https://github.com/apache/spark) functionality it requires 1.4+ which can be obtained [here](http://spark.apache.org/downloads.html).

Installation
------------
```
$ pip install bolt-python
```
