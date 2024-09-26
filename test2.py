from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("SparkUIExample") \
    .getOrCreate()

# Perform some operation to trigger Spark execution
df = spark.range(1000000)
df.count()

print("Spark UI should now be available at http://localhost:4040")
print("Keep this script running and check your browser")

# Keep the session alive to explore the UI
input("Press Enter to exit...")


spark.stop()