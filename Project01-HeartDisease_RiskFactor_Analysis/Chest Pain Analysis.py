# ==========================================
# Analysis: ChestPainType vs Heart Disease
# ==========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, desc

spark = SparkSession.builder \
    .appName("ChestPain_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",
                    header=True,
                    inferSchema=True)

# ==========================================
# 1 ตาราง ChestPainType vs HeartDisease
# ==========================================

print("===== ChestPainType vs HeartDisease =====")
df.groupBy("ChestPainType", "HeartDisease") \
  .agg(count("*").alias("Count")) \
  .orderBy("ChestPainType") \
  .show()

# ==========================================
# 2 คำนวณเปอร์เซ็นต์ความเสี่ยงในแต่ละประเภท
# ==========================================

# จำนวนทั้งหมดในแต่ละ ChestPainType
total_cp = df.groupBy("ChestPainType") \
             .agg(count("*").alias("Total"))

# จำนวนที่เป็นโรคหัวใจในแต่ละ ChestPainType
disease_cp = df.filter(col("HeartDisease") == 1) \
               .groupBy("ChestPainType") \
               .agg(count("*").alias("Disease_Count"))

# รวมข้อมูล
result = total_cp.join(disease_cp, on="ChestPainType")

# คำนวณเปอร์เซ็นต์
result = result.withColumn(
    "Disease_Percentage",
    round((col("Disease_Count") / col("Total")) * 100, 2)
)

print("===== Risk Percentage by ChestPainType (High to Low) =====")
result.orderBy(desc("Disease_Percentage")).show()

spark.stop()