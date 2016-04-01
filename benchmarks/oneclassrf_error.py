"""
==========================================
OneClassRF benchmark
==========================================

A test of OneClassRF on classical anomaly detection datasets.

"""
print(__doc__)
import sys
sys.path.append('~/Bureau/OCRF')

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, OneClassRF
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.utils import shuffle as sh
from sklearn import grid_search


rng = np.random.RandomState(42)

datasets = ['shuttle']

for dat in datasets:
    # loading and vectorization
    print('loading data')

    dataset = fetch_mldata('shuttle')
    X = dataset.data
    y = dataset.target
    sh(X, y, random_state=rng)
    # we remove data with label 4
    # normal data are then those of class 1
    s = (y != 4)
    X = X[s, :]
    y = y[s]
    y = (y != 1).astype(int)

    n_samples, n_features = np.shape(X)
    n_samples_train =  n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X = X.astype(float)
    X = scale(X)
    # remove useless features:
    useless_features = (X.max(axis=0) - X.min(axis=0))==0
    for j in reversed(range(n_features)):
        if useless_features[j]:
            X = np.delete(X, (j), axis=1)

    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    ### cross-val: ####
    parameters = {'max_depth':[1000] , 'max_samples':[.1], 'max_features':[min(10, n_features)], 'n_estimators':[50]}
    model = OneClassRF()
    clf = grid_search.GridSearchCV(model, parameters, refit=False,cv=2)
    clf.fit(X_train, y_train)
    print 'clf.best_params_', clf.best_params_
    model.set_params(**clf.best_params_)

    
    print('OneClassRF processing...')
    # weird: without CV but with same parameters, no error:
    #model = OneClassRF(max_depth=1000, max_samples=.1, max_features=min(10, n_features), n_estimators=50, random_state=rng,  n_jobs=-1)  #commented since cross val
    tstart = time()

    ### training only on normal data:
    X_train = X_train[y_train==0]
    y_train = y_train[y_train==0]

    model.fit(X_train)
    fit_time = time() - tstart
    tstart = time()

    scoring = model.predict(X_test)  # the lower, the more normal
    predict_time = time() - tstart
    fpr, tpr, thresholds = roc_curve(y_test, scoring)
    AUC = auc(fpr, tpr)
