FROM python:3.8.6-buster
#FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0

COPY circor /circor
#COPY model.joblib /model.joblib
COPY requirements_prod.txt /requirements_prod.txt
COPY setup.py /setup.py

#RUN apt-get update
#RUN apt-get install libsndfile1-dev -y

RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt
CMD uvicorn circor.api.api:app --host 0.0.0.0 --port $PORT
