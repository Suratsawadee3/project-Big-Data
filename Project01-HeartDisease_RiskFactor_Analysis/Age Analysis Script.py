# ==========================================
# Analysis 2: Age vs Heart Disease
# ==========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder \
    .appName("Age_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",
                    header=True,
                    inferSchema=True)

print("===== Average Age by HeartDisease =====")
df.groupBy("HeartDisease") \
  .agg(avg("Age").alias("Average_Age")) \
  .show()

spark.stop()