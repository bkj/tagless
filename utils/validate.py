#!/usr/bin/env python

"""
    validate.py
"""

import h5py
import numpy as np
from sklearn import metrics

from rsub import *
from matplotlib import pyplot as plt

inpath = 'results/run-gun-v1-val2-validation-20180510_132245.h5'

f = h5py.File(inpath)

preds = f['preds'].value
y = f['y'].value
validation_idx = f['validation_idx'].value

y_val = y[validation_idx]
preds_val = preds[validation_idx]

metrics.roc_auc_score(y_val, preds_val)

sel = np.argsort(-preds_val)
_ = plt.plot(np.cumsum(y_val[sel]))
_ = plt.plot(np.arange(y_val[sel].sum()), np.arange(y_val[sel].sum()), c='grey')
show_plot()

_ = plt.hist(preds_val[y_val.astype(np.bool)], 100)
show_plot()