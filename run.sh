#!/bin/bash

conda activate tagless_env

python clip_featurize.py \
    --indir twimgs \
    --outdir twimgs_out