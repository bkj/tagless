#!/bin/bash

# run.sh
#
# Script for installing and launching tagless docker container

# --
# Prelim

sudo apt-get update
sudo apt-get install -y git wget jq

# --
# Install docker

sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update

sudo apt-get install -y docker-ce

# --
# Install elasticsearch

sudo apt-get install -y default-jdk

mkdir ~/software
cd software
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.2.0.tar.gz
tar -xzvf elasticsearch-5.2.0.tar.gz
rm elasticsearch-5.2.0.tar.gz
mv elasticsearch-5.2.0 elasticsearch

echo "transport.host: localhost" > elasticsearch/config/elasticsearch.yml
echo "transport.tcp.port: 9300" >> elasticsearch/config/elasticsearch.yml
echo "http.port: 9200" >> elasticsearch/config/elasticsearch.yml
echo "network.host: 0.0.0.0" >> elasticsearch/config/elasticsearch.yml

# screen -S es
# ./elasticsearch/bin/elasticsearch
# 
# !! This has to be run on the host, because the Docker images get deleted after use

# --
# Download tagless

mkdir projects
cd projects
git clone https://github.com/bkj/tagless -b simple.auth
cd tagless/docker

# --
# Build image

sudo docker build -t tagless .

# --
# Run image

# Path to list of filenames (absolute)
FILENAMES="/home/ubuntu/fnames"
# Prefix for image filenames (absolute path)
IMG_DIR="/home/ubuntu/"
# Host IP address
HOST_IP="172.31.1.39"

# Some parameter explanations:
#
# -p => opens port 5000
# --mount => mounts filenames
# --sampler => sets elasticsearch sampler (as opposed to simple_las)
# --filenames => path to file w/ list of filenames
# --img-dir => prefix for filenames
# --labels => possible classes (up to 4 right now)
# --es-host => IP of ES server to use
# --es-index => name of ES index to write to
# --require-authentication => require authentication?
# --allow-relabel => allow labels to be changed -- some algorithms shouldn't allow this

sudo docker run -d \
    -p 5050:5050 \
    -v $IMG_DIR:/img_dir/ \
    --mount type=bind,src=$FILENAMES,target=/filenames \
    tagless python -m tagless \
        --sampler elasticsearch \
        --filenames /filenames \
        --img-dir /img_dir/ \
        --labels 'LA,NY,UNK' \
        --es-host $HOST_IP \
        --es-index tagless-docker-prod-v1 \
        --require-authentication \
        --allow-relabel

# --
# Check to see if annotations got saved

curl -XPOST http://localhost:9200/tagless-docker-prod-v1/_search -d '{
    "query" : {
        "term" : {
            "annotated" : true
        }
    }
}' | jq .