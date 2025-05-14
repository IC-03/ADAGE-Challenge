import os
from pyspark.sql import SparkSession
from main import main  # import your logic from main.py


#os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"

# for client mode
spark = SparkSession.builder.appName("sPCA").getOrCreate()
"""
spark = SparkSession.builder.config(
    "spark.archives", "env.tar.gz#environment"
).appName("sPCA").getOrCreate()
"""
sc = spark.sparkContext
print(sc)

print("hello")
main(spark)

spark.stop()
