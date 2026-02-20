from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "shivani",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="financial_forecast_training",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args=default_args,
    tags=["ml", "forecast"]
) as dag:

    train = BashOperator(
        task_id="train_models",

        # ✅ portable + safe
        bash_command="python -m src.pipelines.train",

        # ✅ ensures imports like `from src.data_loader import ...` work
        env={"PYTHONPATH": "."}
    )

    train
