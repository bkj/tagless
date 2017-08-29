#!/usr/bin/env python

"""
    simple_las_sample.py
"""

import sys
import h5py
import numpy as np
from simple_las import SimpleLAS

class SimpleLASSampler(SimpleLAS):
    def __init__(self, crow, seeds=None, pi=0.05, eta=0.5, alpha=1e-6, n=10, verbose=False):
        init_labels = {}
        
        crow = h5py.File(crow)
        
        if seeds:
            print >> sys.stderr, 'init_las: seeds'
            seeds = h5py.File(seeds)
            init_labels = {i:1 for i in range(seeds['labs'].value.shape[0])}
            
            feats = np.vstack([seeds['feats'].value, crow['feats'].value])
            self.labs = np.hstack([seeds['labs'].value, crow['labs'].value])
        else:
            print >> sys.stderr, 'init_las: no seeds'
            feats = crow['feats'].value
            self.labs = crow['labs'].value
        
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
    
    def get_next(self):
        return self.next_message
    
    def set_label(self, idx, lab):
        self.setLabel(idx, lab)
    
    def get_data(self):
        return self.X.T, self.labels
    
    def n_hits(self):
        return sum(self.hits)
    
    def n_labeled(self):
        return len(self.labeled_idxs)
    
    def is_labeled(self, idx):
        print idx
        return idx in self.labeled_idxs

