#!/usr/bin/env python

"""
    simple_las_sample.py
"""

import sys
import libact
import numpy as np

from libact.base.dataset import Dataset
from libact.models import SVM
from libact.query_strategies import UncertaintySampling

from sklearn.svm import LinearSVC

class UncertaintySampler(object):
    
    def __init__(self, X, y, labs):
        
        y = [yy if yy >= 0 else None for yy in y]
        
        self.dataset = Dataset(X, y)
        self.labs = labs
        
        self.uc = UncertaintySampling(self.dataset, method='lc', model=SVM(kernel='linear'))
        print 'uc ok'
    
    def next_messages(self):
        return np.array([self.uc.make_query()])
    
    def set_label(self, idx, label):
        self.dataset.update(idx, label)
    
    def get_data(self):
        return zip(*self.dataset.get_entries())
    
    def n_hits(self):
        labels = np.array(zip(*self.dataset.get_entries())[1])
        return (labels == 1).sum()
    
    def n_labeled(self):
        return self.dataset.len_labeled()
    
    def is_labeled(self, idx):
        return idx in np.where(zip(*self.dataset.get_entries())[1])[0]
