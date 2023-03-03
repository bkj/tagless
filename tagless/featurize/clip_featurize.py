#!/usr/bin/env python

"""
    clip_featurize.py
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from rich import print

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda

import clip

from tagless.directory_dataset import DirectoryDataset

assert torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',  type=str, default='test')
    parser.add_argument('--indir',   type=str, default='/imgs')
    parser.add_argument('--outdir',  type=str, default='/feats')
    parser.add_argument('--model',   type=str, default="ViT-L/14@336px")
    args = parser.parse_args()
    
    args.outdir = os.path.join(args.outdir, args.run_id)
    
    return args

args = parse_args()
os.makedirs(args.outdir, exist_ok=True)

# --
# Load model

print('clip_featurize: loading CLIP', file=sys.stderr)

model, preprocess = clip.load(args.model, device='cuda')

# model.visual.proj = None # ?? Remove .proj layer

# --
# Load data

print('clip_featurize: prepping data', file=sys.stderr)

dataset    = DirectoryDataset(args.indir, transform=Lambda(preprocess))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# --
# Featurize

print('clip_featurize: featurizing', file=sys.stderr)

n_imgs = len(dataset)
dim    = model.encode_image(dataset[0][None].cuda()).shape[1]
feats  = np.zeros((n_imgs, dim), dtype=np.float32)

print(f'-> ({n_imgs}, {dim}) feature matrix')

offset = 0
with torch.no_grad():
    for x in tqdm(dataloader):
        x      = x.cuda()
        _feats = model.encode_image(x)
        _feats = _feats.detach().cpu().numpy()
        
        feats[offset:offset + _feats.shape[0]] = _feats
        offset += _feats.shape[0]


# --
# Drop files that didn't work

fnames     = np.array(dataset.fnames)
bad_fnames = np.array(list(dataset.bad_fnames))

drop       = np.in1d(fnames, bad_fnames)
feats      = feats[~drop]
fnames     = fnames[~drop]

# --
# Cleanup

np.save(os.path.join(args.outdir, 'feats.npy'), feats)
np.save(os.path.join(args.outdir, 'fnames.npy'), fnames)