#!/bin/bash

conda activate tagless_env

python clip_featurize.py \
    --indir  twimgs \
    --outdir twimgs_out
    

# --
# Run CLIP featurizer

mkdir feats
docker build -t tagless .

# precompute features
docker build -t tagless . && docker run \
    --gpus all --ipc=host \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless.featurize.clip_featurize

# --
# run labeling interface

docker build -t tagless . && docker run \
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

docker build -t tagless . && docker run \
    --gpus all --ipc=host -p 6000:6000 \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless.inference --clf_path /feats/test/gun.joblib

curl -X POST localhost:6000/inference \
    -H "Content-Type: application/octet-stream" \
    --data-binary @'./test.jpg'