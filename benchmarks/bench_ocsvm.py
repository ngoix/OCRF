"""
==========================================
OneClassSVM benchmark
==========================================

A test of OneClassSVM on classical anomaly detection datasets.

"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.datasets import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from sklearn.datasets import fetch_pendigits, fetch_pima, fetch_wilt
from sklearn.datasets import fetch_internet_ads
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle as sh
from scipy.interpolate import interp1d

np.random.seed(1)

nb_exp = 1

# TODO: CV for OCSVM!


# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'internet_ads']

# continuous datasets:
datasets = ['http', 'smtp', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt']

# # new datasets:
# datasets = ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#             'pima', 'wilt']

for dat in datasets:
    # loading and vectorization
    print('loading data')

    if dat == 'internet_ads':  # not adapted to oneclassrf
        dataset = fetch_internet_ads(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'ad.').astype(int)

    if dat == 'wilt':
        dataset = fetch_wilt(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'w').astype(int)

    if dat == 'pima':
        dataset = fetch_pima(shuffle=True)
        X = dataset.data
        y = dataset.target
        # anomalies = class 4

    if dat == 'pendigits':
        dataset = fetch_pendigits(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 4).astype(int)
        # anomalies = class 4

    if dat == 'arrhythmia':
        dataset = fetch_arrhythmia(shuffle=True)
        X = dataset.data
        y = dataset.target
        # rm 5 features containing some '?' (XXX to be mentionned in paper)
        X = np.delete(X, [10, 11, 12, 13, 14], axis=1)
        y = (y != 1).astype(int)
        # normal data are then those of class 1

    if dat == 'annthyroid':
        dataset = fetch_annthyroid(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y != 3).astype(int)
        # normal data are then those of class 3

    if dat == 'spambase':
        dataset = fetch_spambase(shuffle=True)
        X = dataset.data
        y = dataset.target

    if dat == 'ionosphere':
        dataset = fetch_mldata('ionosphere')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y)
        y = (y != 1).astype(int)

    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=True)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        sh(X, y)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True)
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
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train
    X = X.astype(float)

    n_axis = 1000
    x_axis = np.linspace(0, 1, n_axis)
    tpr = np.zeros(n_axis)
    precision = np.zeros(n_axis)
    fit_time = 0
    predict_time = 0

    for ne in range(nb_exp):
        print 'exp num:', ne
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)  # shuffle the dataset
        X = X[indices]
        y = y[indices]

        X_train = X[:n_samples_train, :]
        X_test = X[n_samples_train:, :]
        y_train = y[:n_samples_train]
        y_test = y[n_samples_train:]

        # training only on normal data:
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

        print('OneClassSVM processing...')
        model = OneClassSVM()
        tstart = time()
        model.fit(X_train)
        fit_time += time() - tstart
        tstart = time()

        scoring = -model.decision_function(X_test)  # the lower,the more normal
        predict_time += time() - tstart
        fpr_, tpr_, thresholds_ = roc_curve(y_test, scoring)

        f = interp1d(fpr_, tpr_)
        tpr += f(x_axis)
        tpr[0] = 0.

        precision_, recall_ = precision_recall_curve(y_test, scoring)[:2]
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
    plt.plot(x_axis, tpr, lw=1, label='ROC for %s (area = %0.3f, train-time: %0.2fs, test-time: %0.2fs)' % (dat, AUC, fit_time, predict_time))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for OneClassSVM')
    plt.legend(loc="lower right")

    plt.subplot(122)
    plt.plot(x_axis, precision, lw=1, label='AUPR for %s (area = %0.3f)'
             % (dat, AUPR))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

plt.show()
