#!/usr/bin/env python

"""
    simple_las_sample.py
"""

import sys
import h5py
from datetime import datetime
import numpy as np

from libact.base.dataset import Dataset
from libact.models import LinearSVC
from libact.query_strategies import UncertaintySampling

class UncertaintySampler(object):
    
    def __init__(self, X, y, labs, n=2):
        
        y = [yy if yy >= 0 else None for yy in y]
        
        self.dataset = Dataset(X, y)
        self.labs = labs
        
        self.uc = UncertaintySampling(self.dataset, method='lc', model=LinearSVC())
        self.n = n
    
    def get_next(self, session_id=None):
        print >> sys.stderr, 'get_next: start'
        out = self.uc.make_query(n=self.n)
        print >> sys.stderr, 'get_next: done'
        return out
    
    def set_label(self, idx, label, session_id=None):
        print >> sys.stderr, 'set_label: start'
        out = self.dataset.update(idx, label)
        print >> sys.stderr, 'set_label: done'
        return out
    
    def get_data(self):
        X, y = zip(*self.dataset.get_entries())
        X, y = np.vstack(X), np.array([yy if yy is not None else -1 for yy in y])
        return X, y
    
    def n_hits(self):
        labels = np.array(zip(*self.dataset.get_entries())[1])
        return (labels == 1).sum()
    
    def n_labeled(self):
        return self.dataset.len_labeled()
    
    def is_labeled(self, idx):
        return idx in np.where(zip(*self.dataset.get_entries())[1])[0]
    
    def save(self, outpath):
        """ !! This should be updated to save in same format as simple_las """
        X, y = self.get_data()
    
        f = h5py.File('%s-%s-%s.h5' % (outpath, 'uncertainty', datetime.now().strftime('%Y%m%d_%H%M%S')))
        f['X'] = X
        f['y'] = y
        f['labs'] = self.labs
        f.close()

