#!/usr/bin/env python

"""
    validate.py
"""

import h5py
from sklearn import metrics
from rsub import *
from matplotlib import pyplot as plt

inpath = 'results/run-val-validation-20180130_192915.h5'

f = h5py.File(inpath)

preds = f['preds'].value
y = f['y'].value
validation_idx = f['validation_idx'].value

y_val = y[validation_idx]
preds_val = preds[validation_idx]

metrics.roc_auc_score(y_val, preds_val)

sel = np.argsort(preds_val)
_ = plt.plot(np.cumsum(y_val[sel[::-1]]))
show_plot()


_ = plt.hist(preds_val[y_val])