# ==========================================
# Analysis 1: Gender vs Heart Disease
# ==========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, desc

spark = SparkSession.builder \
    .appName("Gender_Analysis") \
    .getOrCreate()

df = spark.read.csv(r"D:\Big_Data\miniproject\heart.csv",header=True,inferSchema=True)

# ==========================================
# 1️⃣ จำนวนข้อมูลทั้งหมด
# ==========================================

total_all = df.count()
print("===== Total People in Dataset =====")
print("Total:", total_all)

# ==========================================
# 2️⃣ จำนวนคนเป็น / ไม่เป็นโรคหัวใจ (รวมทั้ง dataset)
# ==========================================

overall_hd = df.groupBy("HeartDisease") \
               .agg(count("*").alias("Count"))

overall_hd = overall_hd.withColumn(
    "Percentage",
    round((col("Count") / total_all) * 100, 2)
)

print("===== Overall Heart Disease Distribution =====")
overall_hd.orderBy(desc("Percentage")).show()

# ==========================================
# 3️⃣ วิเคราะห์แยกตามเพศ
# ==========================================

print("===== Total by Gender =====")
df.groupBy("Sex").count().show()

print("===== Gender vs HeartDisease =====")
gender_hd = df.groupBy("Sex", "HeartDisease") \
              .agg(count("*").alias("Count"))

gender_hd.show()

# ==========================================
# 4️⃣ คำนวณเปอร์เซ็นต์ความเสี่ยงตามเพศ
# ==========================================

total_gender = df.groupBy("Sex") \
                 .agg(count("*").alias("Total"))

disease_gender = df.filter(col("HeartDisease") == 1) \
                    .groupBy("Sex") \
                    .agg(count("*").alias("Disease_Count"))

result = total_gender.join(disease_gender, on="Sex")

result = result.withColumn(
    "Disease_Percentage",
    round((col("Disease_Count") / col("Total")) * 100, 2)
)

print("===== Risk Percentage by Gender (High to Low) =====")
result.orderBy(desc("Disease_Percentage")).show()

spark.stop()