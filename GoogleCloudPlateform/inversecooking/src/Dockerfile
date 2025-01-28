# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM ubuntu:18.04

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

ENV PYTHONIOENCODING=utf-8

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update -y && \
    apt-get install -y python3.7 python3-pip python3.7-dev  && \
    apt-get install -y wget
    


RUN apt-get install -y python-qt4
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN ["wget", "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt"]

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app