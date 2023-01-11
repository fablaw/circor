
FROM tensorflow/tensorflow:2.10.0

COPY requirements_prod.txt /requirements_prod.txt


RUN apt-get update
RUN apt-get install libsndfile1-dev -y

RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt

COPY circor /circor

CMD uvicorn circor.api.api:app --host 0.0.0.0 --port $PORT
