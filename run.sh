#!/bin/bash

docker build -t tagless .

# --
# Run CLIP featurizer

mkdir feats
docker run \
    --gpus all --ipc=host \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless.featurize.clip_featurize

# --
# run labeling interface

docker run \
    --gpus all --ipc=host -p 5000:5000 \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless.label_server

# connect on localhost:5000
# label images (click for yes, ctrl+click for no)
# classifier is trained every so often

# --
# run inference w/ exported model

docker run \
    --gpus all --ipc=host -p 6000:6000 \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless.inference --clf_path /feats/test/gun.joblib

# test
curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./gun.jpg'
curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./forest.jpg'
curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./deer.jpg'