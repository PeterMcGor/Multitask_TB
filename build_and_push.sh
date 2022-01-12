#!/bin/bash
echo "build/push script"
DOCKER_NAME=$1  #oni:11500/pedro/py3-tf1_13_1
sudo docker build . -t $DOCKER_NAME
sudo docker push $DOCKER_NAME

