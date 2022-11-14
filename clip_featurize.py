#!/usr/bin/env python

"""
    clip_featurize.py
"""

import os
import sys
import bcolz
import argparse
import numpy as np
from tqdm import tqdm
from shutil import rmtree

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda

import clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',   type=str, default='data_dedup')
    parser.add_argument('--outdir',  type=str, default='out')
    parser.add_argument('--model',   type=str, default="ViT-L/14@336px")
    args = parser.parse_args()
    return args

args = parse_args()
os.makedirs(args.outdir, exist_ok=True)

# --
# Load model

print('clip_featurize: loading CLIP', file=sys.stderr)

model, preprocess = clip.load(args.model, device='cpu')

# --
# Load data

print('clip_featurize: prepping data', file=sys.stderr)

if os.path.exists('_tmp'):
    rmtree('_tmp')

os.makedirs('_tmp', exist_ok=True)
os.symlink(os.path.abspath(args.indir), '_tmp/data')

dataset    = ImageFolder('_tmp', transform=Lambda(preprocess))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

fnames = [
    os.path.abspath(
        os.path.join(
            args.indir,
            os.path.basename(fname)
        )
    ) for fname, _ in dataset.imgs
]

# --
# Featurize

print('clip_featurize: featurizing', file=sys.stderr)

n_imgs = len(dataset)
dim    = model.encode_image(dataset[0][0][None].cpu()).shape[1]
feats  = bcolz.zeros((n_imgs, dim), rootdir=os.path.join(args.outdir, 'feats.bcolz'), mode='w')

offset = 0
with torch.no_grad():
    for (x, _) in tqdm(dataloader):
        x      = x.cpu()
        _feats = model.encode_image(x)
        _feats = _feats.detach().cpu().numpy()
        
        feats[offset:offset + _feats.shape[0]] = _feats
        offset += _feats.shape[0]

# --
# Cleanup

feats.flush()
np.save(os.path.join(args.outdir, 'fnames.npy'), fnames)
rmtree('_tmp')