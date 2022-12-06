FROM python:3.8.6-buster
COPY circor /circor
#COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN apt-get update
RUN apt-get install libsndfile1-dev -y
RUN pip install -r requirements.txt
CMD uvicorn circor.api.api:app --host 0.0.0.0 --port $PORT
