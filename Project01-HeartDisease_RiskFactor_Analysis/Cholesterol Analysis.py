# ==========================================
# Analysis 3: Cholesterol vs Heart Disease
# ==========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count

spark = SparkSession.builder \
    .appName("Cholesterol_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",
                    header=True,
                    inferSchema=True)

print("===== Average Cholesterol by HeartDisease =====")
df.groupBy("HeartDisease") \
  .agg(avg("Cholesterol").alias("Avg_Cholesterol")) \
  .show()

print("===== High Cholesterol (>240) =====")
high_chol = df.filter(col("Cholesterol") > 240)

high_chol.groupBy("HeartDisease") \
         .agg(count("*").alias("Count")) \
         .show()

spark.stop()