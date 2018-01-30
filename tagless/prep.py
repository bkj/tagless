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