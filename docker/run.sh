#!/bin/bash

IMG_DIR="/srv/e2/instagram/trickle/images/"
FILENAMES="fnames"

sudo docker run -it \
    -p 5000:5000 \
    -v $IMG_DIR:/img_dir/ \
    -v $FILENAMES:/filenames \
    tagless python -m tagless \
        --sampler elasticsearch \
        --filenames /filenames \
        --img-dir /img_dir/ \
        --labels 'LA,NY'
