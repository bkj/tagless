#!/bin/bash

# run.sh
# 
# Run Docker container w/ annotation interface

# --
# Start elasticsearch on host 
# !! Important that this doesn't run inside Docker, because 
#   the Docker image gets blown away

# ... ./bin/elasticsearch ...

# --
# Run container

sudo docker build -t tagless .

# List of image filenames (absolute path)
FILENAMES="/srv/e2/instagram/trickle/tagless/fnames"
# Prefix for image filenames (absolute path)
IMG_DIR="/srv/e2/instagram/trickle/images/"

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

sudo docker run -it \
    -p 5050:5050 \
    -v $IMG_DIR:/img_dir/ \
    --mount type=bind,src=$FILENAMES,target=/filenames \
    tagless python -m tagless \
        --sampler elasticsearch \
        --filenames /filenames \
        --img-dir /img_dir/ \
        --labels 'LA,NY,NONE' \
        --es-host 10.1.90.130 \
        --es-index tagless-docker-v0

# --
# Check to see if annotations got saved

curl -XPOST http://localhost:9200/tagless-docker-v0/_search -d '{
    "query" : {
        "term" : {
            "annotated" : true
        }
    }
}' | jq .