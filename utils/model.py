#!/usr/bin/env python

"""
    model.py
    
    Train model on the labels
"""

import h5py
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics

from rsub import *
from matplotlib import pyplot as plt

inpath = './results/run-v0-uncertainty-20170828_120852.h5'

f = h5py.File(inpath)
# paths = np.load('./data/crow.labs.npy')
paths = f['labs'].value

X, y = f['X'].value, f['y'].value
labeled = y >= 0

X_l, y_l = X[labeled], y[labeled].astype('bool')
X_u = X[~labeled]

paths_l, paths_u = paths[labeled], paths[~labeled]

svc = LinearSVC().fit(X_l, y_l)

u_preds = svc.decision_function(X_u)
_ = plt.hist(u_preds[u_preds > -1], 250, alpha=0.25)
show_plot()
# ^^ Ideally, nice and bimodal

# Negative eaxmples
for idx in np.random.choice(np.where(u_preds < 0)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])

# Positive examples
for idx in np.random.choice(np.where(u_preds > 0)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])


for p in paths_u[np.where((u_preds > -1) & (u_preds < 0))[0]]:
    shutil.copy(p, './tmp2')


# Marginal examples
for idx in np.random.choice(np.where((u_preds > -1) & (u_preds < 0))[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])


pd.value_counts(u_preds > -1)