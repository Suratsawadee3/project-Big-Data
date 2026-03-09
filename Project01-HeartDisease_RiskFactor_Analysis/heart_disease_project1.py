# =========================================
# Project 1 : Heart Disease Analysis
# Using PySpark
# =========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, count, when, sum

# -----------------------------------------
# 1) สร้าง Spark Session
# ใช้เริ่มต้นการทำงานของ PySpark
# -----------------------------------------
spark = SparkSession.builder \
    .appName("HeartDiseaseProject") \
    .getOrCreate()

# ปิด Spark log เพื่อให้ terminal แสดงเฉพาะตาราง
spark.sparkContext.setLogLevel("OFF")

# -----------------------------------------
# 2) โหลด Dataset
# อ่านไฟล์ CSV และให้ Spark ตรวจสอบชนิดข้อมูลอัตโนมัติ
# -----------------------------------------
df = spark.read.csv(
    "Heart Failure Prediction Dataset.csv",
    header=True,
    inferSchema=True
)

# -----------------------------------------
# ตารางที่ 1 : แสดงตัวอย่างข้อมูล
# เพื่อดูรูปแบบข้อมูลเบื้องต้น
# -----------------------------------------
df.show()

# -----------------------------------------
# ตารางที่ 2 : โครงสร้างข้อมูล (Schema)
# แสดงประเภทข้อมูลของแต่ละคอลัมน์
# -----------------------------------------
df.printSchema()

# -----------------------------------------
# ตารางที่ 3 : จำนวนข้อมูลทั้งหมด
# แสดงจำนวน record ทั้งหมดใน dataset
# -----------------------------------------
print("Total Rows:", df.count())

# -----------------------------------------
# ตารางที่ 4 : ตรวจสอบ Missing Values
# นับจำนวนค่าที่เป็น NULL ในแต่ละคอลัมน์
# -----------------------------------------
df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()

# -----------------------------------------
# ตารางที่ 5 : จำนวนผู้ป่วย vs ไม่ป่วย
# HeartDisease
# 1 = มีโรคหัวใจ
# 0 = ไม่มีโรคหัวใจ
# -----------------------------------------
df.groupBy("HeartDisease") \
  .count() \
  .show()

# -----------------------------------------
# ตารางที่ 6 : จำนวนผู้ป่วยแยกตามเพศ
# แสดงจำนวน
# - คนที่ไม่เป็นโรคหัวใจ
# - คนที่เป็นโรคหัวใจ
# -----------------------------------------
gender_disease = df.groupBy("Sex").agg(
    count(when(col("HeartDisease") == 0, True)).alias("No_HeartDisease"),
    count(when(col("HeartDisease") == 1, True)).alias("HeartDisease")
)

gender_disease.show()

# -----------------------------------------
# ตารางที่ 7 : คำนวณจำนวนคนทั้งหมดในแต่ละเพศ
# Total_People = คนป่วย + คนไม่ป่วย
# -----------------------------------------
gender_disease = gender_disease.withColumn(
    "Total_People",
    col("HeartDisease") + col("No_HeartDisease")
)

gender_disease.show()

# -----------------------------------------
# ตารางที่ 8 : คำนวณเปอร์เซ็นต์ผู้ป่วยโรคหัวใจ
# Disease_Percentage = (HeartDisease / Total_People) * 100
# -----------------------------------------
gender_disease = gender_disease.withColumn(
    "Disease_Percentage",
    round((col("HeartDisease") / col("Total_People")) * 100, 2)
)

gender_disease.show()

# -----------------------------------------
# ปิด Spark Session
# -----------------------------------------
spark.stop()