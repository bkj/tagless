#!/usr/bin/env python

"""
    simple_las_sample.py
"""

import sys
import numpy as np
from simple_las import SimpleLAS

class SimpleLASSampler(SimpleLAS):
    def __init__(self, crow, seeds=None, pi=0.05, eta=0.5, alpha=1e-6, n=10, verbose=False):
        if seeds:
            print >> sys.stderr, 'init_las: seeds'
            feats = np.vstack([
                np.load('%s.feats.npy' % seeds),
                np.load('%s.feats.npy' % crow),
            ])
            
            seed_labs = np.load('%s.labs.npy' % seeds)
            self.labs = np.hstack([
                seed_labs, 
                np.load('%s.labs.npy' % crow)
            ])
            
            init_labels = {i:1 for i in range(seed_labs.shape[0])}
        else:
            print >> sys.stderr, 'init_las: no seeds'
            feats = np.load('%s.feats.npy' % crow)
            self.labs = np.load('%s.labs.npy' % crow)
            rand_idx = np.random.choice(len(self.labs))
            init_labels = {rand_idx:0}
        
        print >> sys.stderr, 'SimpleLASSampler: initializing w/ %s' % ('no seeds' if seeds is None else 'seeds')
        super(SimpleLASSampler, self).__init__(
            feats,
            init_labels=init_labels,
            pi=pi,
            eta=eta,
            alpha=alpha,
            n=n,
            verbose=verbose
        )
