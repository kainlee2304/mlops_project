from airflow import DAG
from airflow.operators.dummy import DummyOperator  # Cách import cập nhật cho DummyOperator
from datetime import datetime

# Định nghĩa các tham số mặc định cho DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 1),  # Thời điểm bắt đầu thực thi DAG
    'retries': 1,  # Số lần retry nếu gặp lỗi
}

# Định nghĩa DAG với context manager
with DAG(
    'mlops_pipeline',  # Tên DAG
    default_args=default_args,  # Tham số mặc định
    schedule_interval='@daily',  # Lịch chạy hàng ngày
    catchup=False,  # Không cần chạy lại các ngày trong quá khứ
) as dag:

    # Định nghĩa các task
    start = DummyOperator(task_id='start')  # Task bắt đầu
    end = DummyOperator(task_id='end')      # Task kết thúc

    # Kết nối các task với nhau
    start >> end  # Thiết lập thứ tự task: start -> end

