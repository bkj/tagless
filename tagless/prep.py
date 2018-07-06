#!/usr/bin/env python

"""
    prep.py
"""

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--max-records', type=int, default=-1)
    parser.add_argument('--inseed', type=str, default=None)
    parser.add_argument('--outseed', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    print >> sys.stderr, 'loading %s' % args.inpath
    if args.max_records > 0:
        df = pd.read_csv(args.inpath, sep='\t', header=None, nrows=args.max_records)
    else:
        df = pd.read_csv(args.inpath, sep='\t', header=None)
    
    print >> sys.stderr, 'prepping feats'
    feats = np.array(df[range(1, df.shape[1])])
    feats /= np.sqrt((feats ** 2).sum(axis=1, keepdims=True))
    
    print >> sys.stderr, 'prepping labs'
    labs = np.array(df[0])
    
    print >> sys.stderr, 'saving'
    outfile = h5py.File(args.outpath)
    outfile['feats'] = feats
    outfile['labs'] = labs.astype(str)
    outfile.close()
    
    print >> sys.stderr, 'written to %s' % args.outpath

    if args.inseed:
        seeds = set(map(lambda x: x.strip(),open(args.inseed).readlines()))
        seed_df = df[df[0].isin(seeds)]
        seed_feats = np.array(seed_df[range(1, seed_df.shape[1])])
        seed_feats /= np.sqrt((seed_feats ** 2).sum(axis=1, keepdims=True))
        seed_labs = np.array(seed_df[0])
        seedfile = h5py.File(args.outseed)
        seedfile['feats'] = seed_feats
        seedfile['labs'] = seed_labs.astype(str)
        seedfile.close()
