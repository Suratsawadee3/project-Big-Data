# ==========================================
# Analysis 3: Health Metrics Comparison
# ==========================================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Health_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",
                    header=True,
                    inferSchema=True)

# 1️⃣ คำนวณค่าเฉลี่ย
health_stats = df.groupBy("HeartDisease").agg(
    F.round(F.avg("Cholesterol"), 2).alias("Avg_Cholesterol"),
    F.round(F.avg("RestingBP"), 2).alias("Avg_RestingBP"),
    F.round(F.avg("MaxHR"), 2).alias("Avg_MaxHR")
)

print("===== Health Metrics by Heart Disease =====")
health_stats.show()

# 2️⃣ คำนวณความแตกต่าง
hd1 = health_stats.filter(F.col("HeartDisease") == 1).alias("a")
hd0 = health_stats.filter(F.col("HeartDisease") == 0).alias("b")

diff_stats = hd1.crossJoin(hd0).select(
    F.abs(F.col("a.Avg_Cholesterol") - F.col("b.Avg_Cholesterol")).alias("Diff_Cholesterol"),
    F.abs(F.col("a.Avg_RestingBP") - F.col("b.Avg_RestingBP")).alias("Diff_RestingBP"),
    F.abs(F.col("a.Avg_MaxHR") - F.col("b.Avg_MaxHR")).alias("Diff_MaxHR")
)

print("===== Mean Differences =====")
diff_stats.show()

# 3️⃣ Correlation (ใช้ round ของ Python ได้ปกติ)
corr_chol = df.stat.corr("Cholesterol", "HeartDisease")
corr_bp   = df.stat.corr("RestingBP", "HeartDisease")
corr_hr   = df.stat.corr("MaxHR", "HeartDisease")

print("===== Correlation with HeartDisease =====")
print("Correlation (Cholesterol vs HeartDisease):", round(corr_chol, 4))
print("Correlation (RestingBP vs HeartDisease):", round(corr_bp, 4))
print("Correlation (MaxHR vs HeartDisease):", round(corr_hr, 4))

spark.stop()