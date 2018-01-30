#!/usr/bin/env python

"""
    validation_sampler.py
"""

import sys
import h5py
from datetime import datetime
import numpy as np

class ValidationSampler(object):
    
    def __init__(self, preds, labs, y):
        
        self.order = np.argsort(preds)[::-1]
        self._counter = 0
        
        self.preds = preds
        self.labs = labs
        self.y = y
    
    def get_next(self):
        out = self.order[self._counter]
        self._counter += 1
        return np.array([out])
    
    def set_label(self, idx, label):
        self.y[idx] = label
    
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
        f['y'] = self.y
        f['labs'] = self.labs
        f.close()

