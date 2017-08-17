#!/usr/bin/env python

"""
    model.py
    
    Train model on the labels
"""

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics

from rsub import *
from matplotlib import pyplot as plt

f = h5py.File('./results/beds-v1-uncertainty-20170817_153956.h5')
paths = np.load('./data/crow.labs.npy')

X, y = f['X'].value, f['y'].value
labeled = y >= 0

X_l, y_l = X[labeled], y[labeled].astype('bool')
X_u = X[~labeled]

paths_l, paths_u = paths[labeled], paths[~labeled]


svc = LinearSVC().fit(X_l, y_l)

u_preds = svc.decision_function(X_u)

_ = plt.hist(u_preds, 250, alpha=0.25)
show_plot()
# ^^ Nice and bimodal

for idx in np.random.choice(np.where(u_preds < 1)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])


for idx in np.random.choice(np.where(u_preds < 0)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])

