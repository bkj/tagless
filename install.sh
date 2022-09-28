#!/bin/bash

# run.sh

# --
# Create env

conda create -y -n tagless_env python=3.9
conda activate tagless_env

conda install -y -c pytorch pytorch torchvision cudatoolkit=11.3

pip install pandas
pip install bcolz-zipline
pip install ftfy
pip install regex
pip install tqdm
pip install git+https://github.com/openai/CLIP.git
pip install arrow

pip install -e .

# --
# Run

python -m tagless