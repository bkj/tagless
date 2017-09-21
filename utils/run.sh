#!/bin/bash

mkdir -p ./{data,results}

find ./imgs/ -type f | python -m tdesc --model vgg16 --crow > ./data/crow
python ~/projects/tagless/tagless/prep.py --inpath ../data/crow-feats ./data/crow
    
# Run server
python -m tagless --outpath ./results/run-v0 \
    --crow ./data/crow \
    --seeds ./data/crow-seeds \
    --img-dir $(pwd)/imgs \
    --n-las 10

# Connect to localhost:5000