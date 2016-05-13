"""OCRF algorithm."""

import os
import inspect
import glob
import numpy as np

from subprocess import call

from .arff import dumps

# ./ocrf_open/ocrf_v0.6/RELEASE/ocrf


def _p(e):
    if e == 1:
        return 'outlier'
    else:
        return 'target'

def to_fucking_arff(X, y, filename):
    y_s = np.array([_p(e) for e in y])
    data = np.hstack((X, y_s.reshape((y.shape[0], 1))))
    attr = [(u'' + str(i), u'REAL') for i in xrange(X.shape[1])]
    attr.append((u'class', [u'outlier', u'target']))
    arff_struct = {
        u'attributes': attr,
        u'data': data,
        # u'description': u'',
        u'relation': u'tmp_data'}

    f = open(filename, 'w')
    print >> f, dumps(arff_struct)

class OCRF:

    def __init__(self, krsm=-1, krfs=-1, nbTree=100,
                 method=1, rejectOutOfBounds=0, optimize=1,
                 alpha=1.2, beta=10.):
        self.krsm = krsm
        self.krfs = krfs
        self.nbTree = nbTree
        self.method = method
        self.rejectOutOfBounds = rejectOutOfBounds
        self.optimize = optimize
        self.alpha = alpha
        self.beta = beta

    def fit_predict(self, X_train, y_train, X_test, y_test):
        install_dir = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))  # script directory

        call(["mkdir", "tmp_data"])
        to_fucking_arff(X_train, y_train, 'tmp_data/learning_set.arff')
        to_fucking_arff(X_test, y_test, 'tmp_data/test_set.arff')
        call(["mkdir", "results"])

        call([install_dir + "/ocrf_open/ocrf_v0.6/ocrf",
        # call([install_dir + "/../../.skdata/ocrf/ocrf",
              "-path_learning", "tmp_data/learning_set.arff",
              "-path_test", "tmp_data/test_set.arff",
              "-dimension", str(X_train.shape[1]),
              "-db", "tmp_data",
              "-strat", str(0),
              "-fold", str(0),
              "-krsm", str(self.krsm),
              "-krfs", str(self.krfs),
              "-nbTree", str(self.nbTree),
              "-method", str(self.method),
              "-rejectOutOfBounds", str(self.rejectOutOfBounds),
              "-optimize", str(self.optimize),
              "-beta", str(self.beta),
              "-alpha", str(self.alpha)
              ])
        filename = \
            glob.glob('results/res/tmp_data/results_decision_tmp_data*.txt')[0]
        f = open(filename)
        i = 0
        j = 0
        pred = np.empty(y_test.shape[0])
        for line in f.readlines():
            if i > 0 and i < X_test.shape[0]:
                sline = line.split('\t')
                v1 = float(sline[2])
                v2 = float(sline[3])
                pred[j] = v1 / (v1 + v2)
                j = j + 1
            i = i + 1
        call(["rm", "-r", "tmp_data"])
        call(["rm", "-r", "results"])

        return pred
