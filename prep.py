#!/usr/bin/env python

"""
    prep.py
"""

import os
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    inpath = sys.argv[1]
    
    print >> sys.stderr, 'loading %s' % inpath
    df = pd.read_csv(inpath, sep='\t', header=None)
    
    print >> sys.stderr, 'prepping feats'
    feats = np.array(df[range(1, df.shape[1])])
    feats /= np.sqrt((feats ** 2).sum(axis=1, keepdims=True))
    
    print >> sys.stderr, 'prepping labs'
    labs = np.array(df[0])
    labs = np.array(map(os.path.basename, labs))
    
    np.save('%s.feats' % inpath, feats)
    np.save('%s.labs' % inpath, labs)
    print >> sys.stderr, 'written to %s' % os.path.join(inpath, '.{feats,labs}')