#!/bin/bash

IMG_DIR="/srv/e2/instagram/trickle/images/"

sudo docker run -it \
    -p 5000:5000 \
    -v $IMG_DIR:/images/ \
    -v fnames:/fnames \
    tagless python -m tagless \
        --sampler elasticsearch \
        --filenames /fnames \
        --img-dir /images/ \
        --labels 'LA,NY'
