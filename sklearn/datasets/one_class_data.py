
import numpy as np

from .mldata import fetch_mldata
from .fetch_ml_mieux import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from .fetch_ml_mieux import fetch_pendigits, fetch_pima, fetch_wilt
from .fetch_ml_mieux import fetch_internet_ads, fetch_adult
from .kddcup99 import fetch_kddcup99
from .covtype import fetch_covtype

from ..preprocessing import LabelBinarizer, scale
from ..utils import shuffle as sh

__all__ = ["one_class_data"]


def one_class_data(dat, anomaly_max=0.1, percent10_kdd=False, scaling=True,
                   shuffle=True, continuous=True):
    '''
    Parameters
    ----------

    dat : string, dataset to return
        -datasets available:
            'http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt','internet_ads', 'adult'
        -continuous datasets:
            'http', 'smtp', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt', 'adult'

    anomaly_max : float in (0, 1), default=0.1
        max proportion of anomalies.

    percent10_kdd : bool, default=False
        Whether to load only 10 percent of the kdd data.

    scale : bool, default=True
        Whether to scale dataset.

    shuffle : bool, default=True
        Whether to shuffle dataset.

    continuous: bool, default=True
        Whether to remove discontinuous attributes.
    '''

    print('loading data' + dat)

    if dat == 'adult':
        dataset = fetch_adult(shuffle=False)
        X = dataset.data
        y = dataset.target
        # anormal data are those with label >50K:
        y = np.all((y != ' <=50K', y != ' <=50K.'), axis=0).astype(int)

    if dat == 'internet_ads':  # not adapted to oneclassrf
        dataset = fetch_internet_ads(shuffle=False)
        X = dataset.data
        y = dataset.target
        y = (y == 'ad.').astype(int)

    if dat == 'wilt':
        dataset = fetch_wilt(shuffle=False)
        X = dataset.data
        y = dataset.target
        y = (y == 'w').astype(int)

    if dat == 'pima':
        dataset = fetch_pima(shuffle=False)
        X = dataset.data
        y = dataset.target

    if dat == 'pendigits':
        dataset = fetch_pendigits(shuffle=False)
        X = dataset.data
        y = dataset.target
        y = (y == 4).astype(int)
        # anomalies = class 4

    if dat == 'arrhythmia':
        dataset = fetch_arrhythmia(shuffle=False)
        X = dataset.data
        y = dataset.target
        # rm 5 features containing some '?' (XXX to be mentionned in paper)
        X = np.delete(X, [10, 11, 12, 13, 14], axis=1)
        # rm non-continuous features:
        if continuous is True:
            l = []
            for j in range(X.shape[1]):
                if len(set(X[:, j])) < 10:
                    l += [j]
            X = np.delete(X, l, axis=1)
        y = (y != 1).astype(int)
        # normal data are then those of class 1

    if dat == 'annthyroid':
        dataset = fetch_annthyroid(shuffle=False)
        X = dataset.data
        y = dataset.target
        # rm 1-15 features taking only 2 values:
        if continuous is True:
            X = np.delete(X, range(1, 16), axis=1)
        y = (y != 3).astype(int)
        # normal data are then those of class 3

    if dat == 'spambase':
        dataset = fetch_spambase(shuffle=False)
        X = dataset.data
        y = dataset.target

    if dat == 'ionosphere':
        dataset = fetch_mldata('ionosphere')
        X = dataset.data
        y = dataset.target
        # rm first two features which are not continuous (take only 2 values):
        if continuous is True:
            X = np.delete(X, [0, 1], axis=1)
        y = (y != 1).astype(int)

    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=False,
                                 percent10=percent10_kdd)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=False)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        # rm discontinnuous features:
        if continuous is True:
            l = []
            for j in range(X.shape[1]):
                if len(set(X[:, j])) < 10:
                    l += [j]
            X = np.delete(X, l, axis=1)
        # X = np.delete(X, [28, 50], axis=1)
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

    # take max 10 % of abnormal data:
    if anomaly_max is not None:
        index_normal = (y == 0)
        index_abnormal = (y == 1)
        if index_abnormal.sum() > anomaly_max * index_normal.sum():
            X_normal = X[index_normal]
            X_abnormal = X[index_abnormal]
            n_anomalies = X_abnormal.shape[0]
            n_anomalies_max = int(0.1 * index_normal.sum())
            r = sh(np.arange(n_anomalies))[:n_anomalies_max]
            X = np.r_[X_normal, X_abnormal[r]]
            y = np.array([0] * X_normal.shape[0] + [1] * n_anomalies_max)

    X = X.astype(float)

    # scale dataset:
    if scaling:
        X = scale(X)

    # shuffle dataset:
    if shuffle is True:
        X, y = sh(X, y)

    return X, y
