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

# ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
# continuous datasets: http, smtp, shuttle, forescover
datasets = ['shuttle']#['http', 'smtp', 'shuttle', 'forestcover'] 

for dat in datasets:
    # loading and vectorization
    print('loading data')
    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=False, random_state=rng)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
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

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True, random_state=rng)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dat == 'SF':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != 'normal.').astype(int)

    if dat == 'SA':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        lb.fit(X[:, 2])
        x2 = lb.transform(X[:, 2])
        lb.fit(X[:, 3])
        x3 = lb.transform(X[:, 3])
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != 'normal.').astype(int)

    if dat == 'http' or dat == 'smtp':
        y = (y != 'normal.').astype(int)

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

    
    # ### cross-val: ####
    # #parameters = {'max_samples':[.05, .1, .2, .3], 'max_features':[2, 5, 8, 10, 15], 'n_estimators':[5, 10, 20]}   #too much parameters yields segmentation error
    # parameters = {'max_depth':['auto']# ['auto' , 5, 10 , 100, 1000]
    #               , 'max_samples':[0.01, .05, .1, 'auto'], 'max_features':[min(10, n_features), 3], 'n_estimators':[20, 50]}
    # # good param: .05,10,20  (-> auc:0.977)  aussi: 0.02,10,70 aussi: max_depth=1000 et 0.5,8,70
    # model = OneClassRF()
    # clf = grid_search.GridSearchCV(model, parameters, refit=False,cv=2)
    # clf.fit(X_train, y_train)
    # print 'clf.best_params_', clf.best_params_
    # model.set_params(**clf.best_params_)

    ################### bench results:
    #http: {'max_features': 3, 'max_samples': 'auto', 'n_estimators': 50, 'max_depth': 5} -> AUC 0.998 (alpha=1), 0.998 (alpha=0.1) (opt parameters unchanged)
    #smtp: {'max_features': 3, 'max_samples': 0.1, 'n_estimators': 20, 'max_depth': 100} -> AUC 0.981 (alpha=1), 0.98 (alpha=0.1)
    #forestcover: {'max_features': 3, 'max_samples': 'auto', 'n_estimators': 50, 'max_depth': 100} ->AUC 0.979 (alpha=1.)
    #shuttle: (segm error when too large max_depth): {'max_features': 3, 'max_samples': 0.1, 'n_estimators': 20, 'max_depth': 5} --> AUC 1.000
    ###################

    print('OneClassRF processing...')
    model = OneClassRF(max_depth='auto', max_samples=0.1, max_features=3, n_estimators=20, random_state=rng)  # n_jobs=-1)  #commented since cross val
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
    if model.max_depth=='auto':
        plt.plot(fpr, tpr, lw=1, label='ROC for %s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs), (%0.2f, %0.2f, %0.2f, %s)' % (dat, AUC, fit_time, predict_time, model.max_features, model.max_samples, model.n_estimators, model.max_depth))
    else:
        plt.plot(fpr, tpr, lw=1, label='ROC for %s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs), (%0.2f, %0.2f, %0.2f, %0.2f)' % (dat, AUC, fit_time, predict_time, model.max_features, model.max_samples, model.n_estimators, model.max_depth))
        

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
