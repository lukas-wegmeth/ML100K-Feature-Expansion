FROM python:3.9.7 as app

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
