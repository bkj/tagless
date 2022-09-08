#!/bin/bash

mkdir -p ./{data,results}

find ./imgs/ -type f | python -m tdesc --model vgg16 --crow > ./data/crow-feats
find ./seeds/ -type f | python -m tdesc --model vgg16 --crow > ./data/crow-seed-feats

python ~/projects/tagless/tagless/prep.py --inpath ./data/crow-feats ./data/crow-feats.h5
python ~/projects/tagless/tagless/prep.py --inpath ./data/crow-seed-feats ./data/crow-seed-feats.h5
    
# Run server
python -m tagless \
    --outpath ./results/run-v0 \
    --crow ./data/crow-feats.h5 \
    --seeds ./data/crow-seed-feats.h5 \
    --img-dir $(pwd)/imgs \
    --n-las 10

# Connect to localhost:5000