
FROM ubuntu:latest

LABEL AUTHOR="simone.balducci00@gmail.com"
LABEL VERSION="1.0"

SHELL ["/bin/bash", "-c"]

# copy library source files
COPY ./src/nnhep.hpp /usr/local/include/
COPY ./src/nnhep/ /usr/local/include/

# copy preprocessing scripts
COPY ./example/higgs_ml/preprocessing_scripts/ /app
# copy main files for training and validation
COPY ./example/higgs_ml/training_and_validation/ /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib scikit-learn
WORKDIR /app

