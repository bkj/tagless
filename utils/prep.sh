#!/usr/bin/env

# Featurize images
find ./imgs/ -type f | python -m tdesc --model vgg16 --crow > .crow
python prep.py --inpath ../data/crow-feats ./data/crow

# Featurize seeds
find ./seeds/ -type f | python -m tdesc --model vgg16 --crow > .crow-seeds
python prep.py .crow-seeds