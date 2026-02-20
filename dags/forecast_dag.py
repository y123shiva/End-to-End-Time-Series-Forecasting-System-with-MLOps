from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "shivani",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="financial_forecast_training",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",   # weekly retraining
    catchup=False,
    default_args=default_args,
    tags=["ml", "forecast"]
) as dag:

    train = BashOperator(
        task_id="train_models",
        bash_command="python /workspaces/Time-Series-Forecasting/src/train.py"
    )

    train
