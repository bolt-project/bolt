import pytest
import findspark

findspark.init()

@pytest.fixture(scope="session")
def sc():
    from pyspark import SparkContext
    return SparkContext(appName="bolt-tests")

