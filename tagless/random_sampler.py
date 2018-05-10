#!/usr/bin/env python

"""
    random_sampler.py
"""

import os
import sys
import h5py
from datetime import datetime
import numpy as np

class RandomSampler(object):
    
    def __init__(self, crow, n=2, prefix=None):
        
        crow = h5py.File(crow)
        self.labs = crow['labs'].value
        if prefix is not None:
            self.labs = np.array([os.path.join(prefix, l) for l in self.labs])
        
        self.order = np.random.permutation(self.labs.shape[0])
        
        self.y = np.zeros(self.labs.shape[0]) - 1
        self.validation_idx = []
        
        self._counter = 0
        self.n = n
    
    def get_next(self):
        out = []
        for _ in range(self.n):
            out.append(self.order[self._counter])
            self._counter += 1
        
        return np.array(out)
    
    def set_label(self, idx, label):
        self.y[idx] = label
        self.validation_idx.append(idx)
    
    def get_data(self):
        raise NotImplemented
    
    def n_hits(self):
        return (self.y == 1).sum()
    
    def n_labeled(self):
        return (self.y >= 0).sum()
    
    def is_labeled(self, idx):
        return idx in np.where(self.y >= 0)[0]
    
    def save(self, outpath):
        f = h5py.File('%s-%s-%s.h5' % (outpath, 'validation', datetime.now().strftime('%Y%m%d_%H%M%S')))
        f['y']              = self.y
        f['labs']           = self.labs
        f['validation_idx'] = np.array(self.validation_idx)
        f.close()

