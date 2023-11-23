
FROM ubuntu:latest

LABEL AUTHOR="simone.balducci00@gmail.com"
LABEL VERSION="1.0"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib scikit-learn
WORKDIR /app

