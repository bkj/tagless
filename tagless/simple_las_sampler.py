#!/usr/bin/env python

"""
    simple_las_sample.py
"""

import os
import sys
import bcolz
import numpy as np
import pandas as pd
from simple_las import SimpleLAS

# class SimpleLASSampler(SimpleLAS):
#     def __init__(self, feat_path, meta_path, seeds=None, pi=1e-3, eta=0.5, alpha=0, n=3, verbose=False, prefix=None):
#         init_labels = {}
        
#         feat_path = '/home/bjohnson/projects/moco/data/naip/feats/houston'
#         feats     = bcolz.open(feat_path)[:]
#         feats     = feats / np.sqrt((feats ** 2).sum(axis=-1, keepdims=True))
        
#         meta_path       = '/home/bjohnson/projects/moco/data/naip/feats/houston_meta.jl'
#         meta            = pd.read_json(meta_path, lines=True)
        
#         # feats = feats[0::8]
#         # meta  = meta.iloc[0::8]
#         # assert feats.shape[0] == meta.shape[0]
        
#         meta['geohash'] = meta.img_name.apply(lambda x: os.path.basename(x).split('.')[0])
#         meta['idx']     = meta.geohash + '_' + meta.patch_idx.astype(str)
#         meta            = meta.set_index('idx', drop=False)
        
#         assert feats.shape[0] == meta.shape[0]
        
#         print(meta.head())
        
#         self.idx2idx = dict(zip(meta.idx, range(meta.shape[0])))
#         self.meta    = meta
        
#         # if seeds:
#         #     print('init_las: seeds', file=sys.stderr)
#         #     seeds = h5py.File(seeds)
#         #     init_labels = {i:1 for i in range(seeds['labs'].value.shape[0])}
            
#         #     feats     = np.vstack([seeds['feats'].value, feat['feats'].value])
#         #     self.labs = np.hstack([seeds['labs'].value, feat['labs'].value])
        
#         # baseball
#         init_labels[self.idx2idx['9vhr0x_0']] = 1
#         init_labels[self.idx2idx['9vk227_0']] = 1
        
#         print(init_labels)
        
#         # track
#         # init_labels[self.idx2idx['9v5z5g_0']] = 1
        
#         super().__init__(
#             feats,
#             init_labels=init_labels,
#             pi=pi,
#             eta=eta,
#             alpha=alpha,
#             n=n,
#             verbose=verbose,
#         )
        
#     def get_next(self):
#         return self.next_message
    
#     def set_label(self, idx, lab):
#         self.setLabel(self.idx2idx[idx], lab)
    
#     def get_data(self):
#         return self.X.T, self.labels
    
#     def n_hits(self):
#         return sum(self.hits)
    
#     def n_labeled(self):
#         return len(self.labeled_idxs)
    
#     def is_labeled(self, idx):
#         print(idx)
#         return idx in self.labeled_idxs


from sklearn.svm import LinearSVC
class SimpleLASSampler:
    def __init__(self, feat_path, meta_path, seeds=None, prefix=None):
        init_labels = {}
        
        feat_path = '/home/bjohnson/projects/moco/data/naip/feats/houston'
        feats     = bcolz.open(feat_path)[:]
        feats     = feats / np.sqrt((feats ** 2).sum(axis=-1, keepdims=True))
        
        meta_path       = '/home/bjohnson/projects/moco/data/naip/feats/houston_meta.jl'
        meta            = pd.read_json(meta_path, lines=True)
        
        # feats = feats[0::8]
        # meta  = meta.iloc[0::8]
        # assert feats.shape[0] == meta.shape[0]
        
        meta['geohash'] = meta.img_name.apply(lambda x: os.path.basename(x).split('.')[0])
        meta['idx']     = meta.geohash + '_' + meta.patch_idx.astype(str)
        meta            = meta.set_index('idx', drop=False)
        
        self.idx2idx = dict(zip(meta.idx, range(meta.shape[0])))
        self.meta    = meta
        self.feats   = feats
        
        n_obs  = self.meta.shape[0]
        self.labels = np.zeros(n_obs) - 1
        
        # baseball
        self.labels[self.idx2idx['9vhr0x_0']] = 1
        self.labels[self.idx2idx['9vk227_0']] = 1
        
        # track
        # self.labels[self.idx2idx['9v5z5g_0']] = 1
        # self.labels[self.idx2idx['9vhruw_7']] = 1
        
        # planes
        # self.labels[self.idx2idx['9vk0xu_2']] = 1
        
        self.labeled_idxs = set([])
        self.hits         = []
    
    def get_next(self):
        X = np.row_stack([
            self.feats[self.labels == 1],
            self.feats[self.labels == 0],
            self.feats[np.random.choice(np.where(self.labels == -1)[0], 5_000, replace=False)]
        ])
        y = np.hstack([
            1 * np.ones((self.labels == 1).sum()),
            0 * np.ones((self.labels == 0).sum()),
            0 * np.ones(5_000),
        ])
        sample_weight = np.hstack([
            np.ones((self.labels == 1).sum()),
            np.ones((self.labels == 0).sum()),
            np.ones(5_000) * 0.001,
        ])
        self.model = LinearSVC(C=2).fit(X, y, sample_weight=sample_weight)
        
        self.labels[self.model.decision_function(self.feats).argsort()[-5:][::-1]]
        
        unlabeled = np.where(self.labels == -1)[0]
        scores    = self.model.decision_function(self.feats[unlabeled])
        return unlabeled[np.argsort(scores)[-3:][::-1]]
    
    def set_label(self, idx, lab):
        self.hits.append(lab)
        self.labeled_idxs.add(idx)
        self.labels[self.idx2idx[idx]] = lab
    
    # def get_data(self):
    #     return self.X.T, self.labels
    
    def n_hits(self):
        return int((self.labels == 1).sum())
    
    def n_labeled(self):
        return int((self.labels != 1).sum())
    
    def is_labeled(self, idx):
        return idx in self.labeled_idxs

