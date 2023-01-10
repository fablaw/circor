FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0
#FROM tensorflow/tensorflow:2.10.0

COPY circor /circor
COPY requirements_prod.txt /requirements_prod.txt


RUN apt-get update
RUN apt-get install libsndfile1-dev -y

RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt
CMD uvicorn circor.api.api:app --host 0.0.0.0 --port $PORT
