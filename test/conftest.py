import os
import sys
import glob
import pytest

spark_home = os.environ['SPARK_HOME']
spark_python = os.path.join(spark_home, 'python')
py4j = glob.glob(os.path.join(spark_python, 'lib', 'py4j-*.zip'))[0]
sys.path[:0] = [spark_python, py4j]

@pytest.fixture(scope="session")
def sc():
    from pyspark import SparkContext
    return SparkContext(appName="bolt-tests")

