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

# run app
docker build -t tagless . && docker run \
    --gpus all --ipc=host -p 5000:5000 \
    --mount type=bind,source=/home/ubuntu/_data0,target=/imgs \
    --mount type=bind,source=$(pwd)/feats,target=/feats \
    -it tagless \
        python -m tagless

# connect on localhost:5000