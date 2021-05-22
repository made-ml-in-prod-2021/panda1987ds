FROM python:3.8-slim

ARG NAME=online_inference
ARG APP_HOME=/opt/$NAME

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt && \
    mkdir -p $APP_HOME/logs

COPY configs/logging.ini $APP_HOME/configs/logging.ini
COPY models/model.pkl $APP_HOME/models/model.pkl
COPY src/ $APP_HOME/src/
COPY data/request_example.json $APP_HOME/data/request_example.json
COPY tests/ $APP_HOME/tests/

WORKDIR $APP_HOME/src

ENV PYTHONPATH "${PYTHONPATH}:.."

EXPOSE 8080/tcp

CMD ["python", "app.py"]
