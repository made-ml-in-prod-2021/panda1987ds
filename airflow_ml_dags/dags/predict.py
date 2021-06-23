from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor
import os

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_file(path):
    print(path)
    print(os.path.exists(path))
    print(os.path.abspath(os.curdir))
    return os.path.exists(path)


with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
        tags=['hw']
) as dag:
    wait = PythonSensor(
        task_id="wait_for_file",
        python_callable=_wait_for_file,
        op_kwargs={'path':'/opt/airflow/data/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )
    predict = DockerOperator(
        image="predict",
        command="--data_path /data/processed/{{ ds }} --model_path /data/models/{{ ds }} --output_path /data/predictions/{{ds }}",
        task_id="predict",
        do_xcom_push=False,
        volumes=["/home/dasha/PycharmProjects/ml_prod/panda1987ds/airflow/data:/data"]
    )

    wait >> predict
