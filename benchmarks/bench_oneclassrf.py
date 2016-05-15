"""
==========================================
OneClassRF benchmark
==========================================

A test of OneClassRF on classical anomaly detection datasets.

"""
print(__doc__)
import pdb
from time import time
import numpy as np
# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.ensemble import OneClassRF
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.datasets import one_class_data
from sklearn.preprocessing import scale
from sklearn import grid_search  # let it for cv
from scipy.interpolate import interp1d

from sklearn.utils import shuffle as sh

np.random.seed(1)

nb_exp = 10

# NOTE: to be launch from master: bench_lof, bench_isolation_forest,
#       bench_ocsvm

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'internet_ads', 'adult']

# continuous datasets:
datasets = ['http', 'smtp', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt', 'adult']
# new: ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#       'pima', 'wilt', 'adult']

# datasets = ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']
# datasets = ['wilt']
plt.figure(figsize=(25, 17))

for dat in datasets:
    # loading and vectorization
    X, y = one_class_data(dat)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    # de-comment to rm useless features:
    # useless_features = (X.max(axis=0) - X.min(axis=0)) == 0
    # for j in reversed(range(n_features)):
    #     if useless_features[j]:
    #         X = np.delete(X, (j), axis=1)

    n_axis = 1000
    x_axis = np.linspace(0, 1, n_axis)
    tpr = np.zeros(n_axis)
    precision = np.zeros(n_axis)
    fit_time = 0
    predict_time = 0

    for ne in range(nb_exp):
        print 'exp num:', ne
        X, y = sh(X, y)

        X_train = X[:n_samples_train, :]
        X_test = X[n_samples_train:, :]
        y_train = y[:n_samples_train]
        y_test = y[n_samples_train:]

        print('OneClassRF processing...')
        model = OneClassRF()

        # training only on normal data: (not supported in cv)
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

        tstart = time()
        model.fit(X_train)
        fit_time += time() - tstart
        tstart = time()

        scoring = model.predict(X_test)  # the lower, the more normal
        # scoring = scale(scoring)
        predict_time += time() - tstart
        fpr_, tpr_ = roc_curve(y_test, scoring)[:2]
        f = interp1d(fpr_, tpr_)
        tpr += f(x_axis)
        tpr[0] = 0.

        precision_, recall_ = precision_recall_curve(y_test, scoring)[:2]

        # cluster: old version of scipy -> interpol1d needs sorted x_input
        arg_sorted = recall_.argsort()
        recall_ = recall_[arg_sorted]
        precision_ = precision_[arg_sorted]

        f = interp1d(recall_, precision_)
        precision += f(x_axis)

    tpr /= float(nb_exp)
    fit_time /= float(nb_exp)
    predict_time /= float(nb_exp)
    AUC = auc(x_axis, tpr)
    precision /= float(nb_exp)
    precision[0] = 1.
    AUPR = auc(x_axis, precision)

    plt.subplot(121)
    plt.plot(x_axis, tpr, lw=1, label='%s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs)' % (dat, AUC, fit_time, predict_time, ))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('Receiver operating characteristic for OneClassRF', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 15})

    plt.subplot(122)
    plt.plot(x_axis, precision, lw=1, label='%s (area = %0.3f)'
             % (dat, AUPR))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('Precision-Recall curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 15})

plt.savefig('bench_oneclassrf_roc_pr_supervised_factorized_sqrt')
