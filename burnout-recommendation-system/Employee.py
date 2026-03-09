import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler, StandardScaler


spark = SparkSession.builder \
    .appName("BurnoutBehaviorRecommender") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


df = spark.read.csv("work_from_home_burnout_dataset.csv", header=True, inferSchema=True)

print(f"Total Records : {df.count()}")

print("\n=== DATA SCHEMA ===")
df.printSchema()

print("\n=== RAW DATA PREVIEW (Top 5 Rows) ===")
df.show(5, truncate=False)


# เลือกเฉพาะคอลัมน์พฤติกรรมที่จะนำมาวิเคราะห์
features = [
    "work_hours", 
    "meetings_count", 
    "breaks_taken", 
    "screen_time_hours", 
    "sleep_hours"
]

# นำคอลัมน์พฤติกรรมมามัดรวมกันเป็น Vector 
assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
df_assembled = assembler.transform(df)

# ปรับสเกลข้อมูล (Standardization) เพื่อลดความเหลื่อมล้ำของหน่วยวัด
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)


# กำหนด ID ของพนักงานที่เราต้องการหาระบบแนะนำให้
TARGET_USER_ID = 9

# ดึงข้อมูลของพนักงานเป้าหมาย
target_row = df_scaled.filter(F.col("user_id") == TARGET_USER_ID).first()

# แปลงคุณสมบัติของเป้าหมายให้เป็น Numpy Array เพื่อเตรียมคำนวณ
target_vector = target_row["scaled_features"].toArray()
target_burnout = target_row["burnout_score"]

print("\n" + "="*75)
print(f"TARGET EMPLOYEE (User ID: {TARGET_USER_ID})")
print("="*75)
df.filter(F.col("user_id") == TARGET_USER_ID).select(["user_id"] + features + ["burnout_score"]).show()


# ฟังก์ชันคำนวณความคล้ายคลึงของเวกเตอร์ (Cosine Similarity)
def calculate_cosine_sim(vec):
    v = vec.toArray()
    norm = (np.linalg.norm(target_vector) * np.linalg.norm(v))
    if norm == 0:
        return 0.0
    return float(np.dot(target_vector, v) / norm)

# แปลงฟังก์ชัน Python ให้เป็น PySpark UDF (User Defined Function)
cosine_sim_udf = F.udf(calculate_cosine_sim, FloatType())

# คำนวณความคล้ายคลึงระหว่างเป้าหมายกับทุกคนในบริษัท และสร้างเป็นคอลัมน์ใหม่
df_sim = df_scaled.withColumn("similarity", cosine_sim_udf(F.col("scaled_features")))


# กรองเอาตัวเองออก และเรียงลำดับคนที่คล้ายเราที่สุด 5 อันดับแรก
df_similar = df_sim.filter(F.col("user_id") != TARGET_USER_ID)
top5_df = df_similar.orderBy(F.col("similarity").desc()).limit(5)

print("\n" + "="*75)
print("TOP 5 SIMILAR EMPLOYEES")
print("="*75)

display_cols = ["user_id", "similarity", "burnout_score"] + features
top5_df.select(display_cols).show()


# จาก 5 คนที่คล้ายเราที่สุด ให้เลือกคนที่มีคะแนนความหมดไฟ (Burnout Score) "ต่ำที่สุด" มาเป็นต้นแบบ
best_employee = top5_df.orderBy(F.col("burnout_score").asc()).first()

print("\n" + "="*75)
print("CURRENT VS RECOMMENDED BEHAVIOR")
print("="*75)
print(f"Role Model User ID        : {best_employee['user_id']}")
print(f"Current Burnout Score     : {target_burnout:.2f}")
print(f"Recommended Burnout Score : {best_employee['burnout_score']:.2f}")
print("-" * 75)
print(f"{'Feature':<20} | {'Current':<10} | {'Recommended':<15} | {'Action'}")
print("-" * 75)


# เปรียบเทียบค่าพฤติกรรมทีละตัว เพื่อให้คำแนะนำ
for f in features:
    cur_val = target_row[f]
    rec_val = best_employee[f]
    
    if rec_val > cur_val:
        action = "Increase"
    elif rec_val < cur_val:
        action = "Reduce"
    else:
        action = "Keep"
        
    print(f"{f:<20} | {cur_val:<10.2f} | {rec_val:<15.2f} | {action}")

print("\n" + "="*75)
print("RECOMMENDATION COMPLETE")
print("="*75 + "\n")

# ปิด Spark Session 
spark.stop()