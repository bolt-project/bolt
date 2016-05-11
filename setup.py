#!/usr/bin/env python

from setuptools import setup

version = '0.6.0'

setup(
    name='bolt-python',
    version=version,
    description='Unified and scalable interface for multidimensional arrays',
    author='The Freeman Lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/bolt-project/bolt',
    packages=['bolt',
              'bolt.local',
              'bolt.spark'],
    long_description=open('README.rst').read(),
    install_requires=open('requirements.txt').read().split()
)
