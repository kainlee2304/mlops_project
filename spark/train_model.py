import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("MLOps Training Model with MLflow") \
    .getOrCreate()

# Tạo một DataFrame mẫu nếu bạn không có tệp CSV
# Thay thế phần này bằng đoạn mã đọc dữ liệu từ file nếu bạn có tệp "data.csv"
data = spark.createDataFrame([
    (1.0, 2.0, 3.0, 10.0),
    (2.0, 3.0, 4.0, 20.0),
    (3.0, 4.0, 5.0, 30.0),
    (4.0, 5.0, 6.0, 40.0),
    (5.0, 6.0, 7.0, 50.0)
], ["feature1", "feature2", "feature3", "label"])

# Nếu có tệp CSV, thay thế đoạn trên bằng đoạn sau:
# data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Bắt đầu theo dõi thí nghiệm với MLflow
mlflow.set_experiment("MLOps Training Model Experiment")  # Đặt tên cho thí nghiệm
with mlflow.start_run():
    # Tiền xử lý dữ liệu: chuyển đổi các cột thành vector tính năng
    assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")
    assembled_data = assembler.transform(data)

    # Chia dữ liệu thành training và test set
    train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

    # Huấn luyện mô hình Linear Regression
    lr = LinearRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_data)

    # Đánh giá mô hình trên dữ liệu test
    test_results = lr_model.evaluate(test_data)

    # Ghi kết quả vào MLflow
    mlflow.log_metric("RMSE", test_results.rootMeanSquaredError)
    mlflow.log_metric("R2", test_results.r2)
    mlflow.log_param("features", ["feature1", "feature2", "feature3"])
    mlflow.spark.log_model(lr_model, "linear_regression_model")

    # In kết quả
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)
    print("Test RMSE:", test_results.rootMeanSquaredError)
    print("Test R2:", test_results.r2)

# Dừng SparkSession
spark.stop()
