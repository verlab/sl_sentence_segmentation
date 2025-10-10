FROM tensorflow/tensorflow:2.14.0-gpu

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /requirements.txt

RUN apt-get update && \
    apt-get install --no-install-recommends -y g++ ffmpeg git

RUN pip install -r requirements.txt

WORKDIR /

RUN rm requirements.txt
