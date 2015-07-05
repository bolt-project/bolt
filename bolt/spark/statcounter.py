#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file is ported from spark/util/StatCounter.scala
#
# This code is based on pyspark's statcounter.py and used under the ASF 2.0 license.

import copy
from itertools import chain

from numpy import sqrt


class StatCounter(object):

    REQUIRED_FOR = {
        'mean': ('mu',),
        'sum': ('mu',),
        'variance': ('mu', 'm2'),
        'stdev': ('mu', 'm2'),
        'all': ('mu', 'm2')
    }

    def __init__(self, values=(), stats='all'):
        self.n = 0
        self.mu = 0.0
        self.m2 = 0.0

        if isinstance(stats, str):
            stats = [stats]
        self.required = frozenset(chain().from_iterable([StatCounter.REQUIRED_FOR[stat] for stat in stats]))

        for v in values:
            self.merge(v)

    # add a value into this StatCounter, updating the statistics
    def merge(self, value):
        self.n += 1
        if self.__requires('mu'):
            delta = value - self.mu
            self.mu += delta / self.n
            if self.__requires('m2'):
                self.m2 += delta * (value - self.mu)

        return self

    # checks whether the passed attribute name is required to be updated in order to support the
    # statistics requested in self.requested
    def __requires(self, attrname):
        return attrname in self.required

    # merge another StatCounter into this one, adding up the statistics
    def combine(self, other):
        if not isinstance(other, StatCounter):
            raise Exception("can only merge StatCounters!")

        # reference equality holds
        if other is self:
            # avoid overwriting fields in a weird order
            self.merge(copy.deepcopy(other))
        else:
            # accumulator should only be updated if it's valid in both statcounters
            self.required = set(self.required).intersection(set(other.required))

            if self.n == 0:
                self.n = other.n
                for attrname in ('mu', 'm2'):
                    if self.__requires(attrname):
                        setattr(self, attrname, getattr(other, attrname))

            elif other.n != 0:
                if self.__requires('mu'):
                    delta = other.mu - self.mu
                    if other.n * 10 < self.n:
                        self.mu = self.mu + (delta * other.n) / (self.n + other.n)
                    elif self.n * 10 < other.n:
                        self.mu = other.mu - (delta * self.n) / (self.n + other.n)
                    else:
                        self.mu = (self.mu * self.n + other.mu * other.n) / (self.n + other.n)

                    if self.__requires('m2'):
                        self.m2 += other.m2 + (delta * delta * self.n * other.n) / (self.n + other.n)

                self.n += other.n
        return self

    def count(self):
        return self.n

    def __isavail(self, attrname):
        if not all(attr in self.required for attr in StatCounter.REQUIRED_FOR[attrname]):
            raise ValueError("'%s' stat not available, must be requested at "
                             "StatCounter instantiation" % attrname)

    @property
    def mean(self):
        self.__isavail('mean')
        return self.mu

    @property
    def sum(self):
        self.__isavail('sum')
        return self.n * self.mu

    @property
    def variance(self):
        self.__isavail('variance')
        if self.n == 0:
            return float('nan')
        else:
            return self.m2 / self.n

    @property
    def stdev(self):
        self.__isavail('stdev')
        return sqrt(self.variance)
