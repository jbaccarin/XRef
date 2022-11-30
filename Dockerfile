#FROM python:3.8.12-buster
FROM tensorflow/tensorflow:2.10.0
COPY api /api
COPY requirements.txt /requirements.txt
COPY scripts /scripts
# COPY requirements_prod.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
