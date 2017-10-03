#!/bin/bash

# --
# Start elasticsearch on host 
# !! Important that this doesn't run inside Docker, because 
#   the Docker image gets blown away

# ... ./bin/elasticsearch ...

# --
# Run container

sudo docker build -t tagless .

IMG_DIR="/srv/e2/instagram/trickle/images/"
FILENAMES="/srv/e2/instagram/trickle/tagless/fnames"

sudo docker run -it \
    -p 5000:5000 \
    -v $IMG_DIR:/img_dir/ \
    --mount type=bind,src=$FILENAMES,target=/filenames \
    tagless python -m tagless \
        --sampler elasticsearch \
        --filenames /filenames \
        --img-dir /img_dir/ \
        --labels 'LA,NY' \
        --es-host 10.1.90.130 \
        --es-index tagless-docker-v0

# --
# Make sure some annotations took

curl -XPOST http://localhost:9200/tagless-docker-v0/_search -d '{
    "query" : {
        "term" : {
            "annotated" : true
        }
    }
}' | jq .