FROM airflow-ml-base:latest

COPY validate.py /validate.py
COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

ENTRYPOINT ["python", "validate.py"]
