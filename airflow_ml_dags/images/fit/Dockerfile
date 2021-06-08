FROM airflow-ml-base:latest

COPY fit.py /fit.py
COPY train_config.yaml /train_config.yaml
COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

ENTRYPOINT ["python", "fit.py"]
