#!/usr/bin/env python

"""
    model.py
    
    Train model on the labels
"""

import h5py
import shutil
import cPickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics

from rsub import *
from matplotlib import pyplot as plt


inpath = 'results/run-gun-v1-uncertainty-20180510_125633.h5'

f = h5py.File(inpath)
paths = f['labs'].value

X, y = f['X'].value, f['y'].value
labeled = y >= 0

X_l, y_l = X[labeled], y[labeled].astype('bool')
X_u = X[~labeled]

paths_l, paths_u = paths[labeled], paths[~labeled]

svc = LinearSVC(C=1).fit(X_l, y_l)
l_preds = svc.decision_function(X_l)

model = {
    "coefs" : svc.coef_.squeeze(),
    "intercept" : svc.intercept_[0],
}

assert (model['intercept'] + X_l.dot(model['coefs']) == l_preds).all()
cPickle.dump(model, open('results/hotel-20180130-v2.pkl', 'w'))

u_preds = svc.decision_function(X_u)
l_preds = svc.decision_function(X_l)

_ = plt.hist(u_preds, 250, alpha=0.25)
_ = plt.axvline(0, c='grey')
show_plot()

_ = plt.hist(l_preds, 250, alpha=0.25) # Bimodal?
show_plot()

# >>
_ = plt.hist(u_preds[u_preds > -2], 250, alpha=0.25)
_ = plt.axvline(0, c='grey')
show_plot()
# <<

# --
# Examples

# Negative examples
for idx in np.random.choice(np.where(u_preds < 0)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])


# Positive examples
for idx in np.random.choice(np.where(u_preds > 0)[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])

# Marginal examples
for idx in np.random.choice(np.where((u_preds > -1) & (u_preds < 0))[0], 10):
    print idx, u_preds[idx], paths_u[idx]
    rsub(paths_u[idx])


all_preds = svc.decision_function(X)
all_preds[labeled] = -np.inf
f['preds'] = all_preds
f.close()
