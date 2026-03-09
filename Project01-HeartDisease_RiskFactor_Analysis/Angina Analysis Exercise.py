# ==========================================
# Analysis 4: ExerciseAngina vs Heart Disease
# ==========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, sum, round

spark = SparkSession.builder \
    .appName("ExerciseAngina_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",
                    header=True,
                    inferSchema=True)

print("===== ExerciseAngina vs HeartDisease (Count) =====")

grouped = df.groupBy("ExerciseAngina", "HeartDisease") \
            .agg(count("*").alias("Count"))

grouped.show()

# ===============================
# คำนวณ Risk Rate
# ===============================

# รวมจำนวนทั้งหมดในแต่ละกลุ่ม ExerciseAngina
total_per_group = grouped.groupBy("ExerciseAngina") \
    .agg(sum("Count").alias("Total"))

# ดึงเฉพาะจำนวนผู้ที่เป็นโรคหัวใจ (HeartDisease = 1)
heart_cases = grouped.filter(col("HeartDisease") == 1) \
    .select("ExerciseAngina", col("Count").alias("HeartCases"))

# Join เพื่อคำนวณอัตราความเสี่ยง
risk_df = heart_cases.join(total_per_group, "ExerciseAngina") \
    .withColumn("Risk_Rate",
                round(col("HeartCases") / col("Total") * 100, 2))

print("===== Risk Rate by ExerciseAngina =====")
risk_df.show()

spark.stop()