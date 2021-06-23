import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["pankratova.dasha@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "generate_date",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
        tags=['hw']
) as dag:
    generate = DockerOperator(
        image="generate-data",
        command="--path /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="airflow_generate_data",
        do_xcom_push=False,
        volumes=["/home/dasha/PycharmProjects/ml_prod/panda1987ds/airflow/data:/data"]
    )

    generate
